"""Servidor HTTP mínimo para desplegar el pipeline como servicio.

    python -m ragb.api --host 0.0.0.0 --port 8080

Tres rutas, y nada más:

  GET  /healthz  liveness:  200 siempre que el proceso conteste.
  GET  /readyz   readiness: 200 solo si el pod PUEDE atender una consulta:
                 configuración completa (base, clave del modelo y token de la
                 API) y, si se pidió calentar el modelo de embeddings, modelo
                 ya cargado. 503 con el motivo si no. Kubernetes saca el pod
                 del Service, no lo mata.
  POST /query    {"question": str, "tenant": str, "top_k"?: int}
                 con `Authorization: Bearer <RAGB_API_TOKEN>`.

Decisiones que conviene no deshacer sin pensarlo:

1. **Librería estándar, sin FastAPI.** El job `guards` de CI instala solo
   pytest y dos librerías puras; un framework web habría dejado esta capa sin
   probar justo donde más importa (la autenticación). `http.server` alcanza
   para tres rutas y se prueba en un hilo, sin red ni credenciales.

2. **Cerrado por defecto.** Sin `RAGB_API_TOKEN` el endpoint de consulta
   responde 503, no "abierto mientras nadie lo configure". Un servicio que
   arranca sin credencial y atiende igual es un servicio público por accidente,
   y este pipeline tiene detrás una base con datos por tenant y una API de
   pago. Y por lo mismo `/readyz` también exige el token: un pod que responde
   503 al 100 % de las consultas no está listo, y dejarlo dentro del Service
   solo reparte errores entre réplicas.

3. **La configuración se lee en cada request, no al importar.** `ragb.config`
   congela las variables de entorno en el momento del import; para un proceso
   de CLI está bien, pero `/readyz` existe para reflejar el estado ACTUAL, y
   los tests necesitan cambiar el entorno con el servidor ya levantado. Por eso
   acá se consulta `os.environ` directo.

4. **`/readyz` no abre conexión a la base.** Una readiness que pega a Postgres
   en cada sonda se convierte en carga, y un parpadeo de la base sacaría todas
   las réplicas del Service a la vez —cuando lo que está mal no es el pod—. Se
   verifica que la configuración esté completa; la salud de la base se mide
   aparte, con las herramientas de la base.

5. **El modelo de embeddings se calienta al arrancar, en un hilo, y se carga
   UNA sola vez.** Sin esto cada réplica nueva recibía tráfico fría: la
   primera consulta pagaba descarga y carga del modelo (decenas de segundos)
   y, peor, N primeras consultas simultáneas lo cargaban N veces
   (`functools.lru_cache` no serializa la primera ejecución) hasta el OOM.
   Con `RAGB_WARMUP=1` el proceso arranca, `/healthz` contesta desde el primer
   segundo, y `/readyz` dice "cargando" hasta que el modelo esté en memoria.
   Sin la variable (tests, CI) nada de esto corre y el pipeline carga el
   modelo perezoso, pero siempre bajo el mismo candado.

6. **Nunca se registra el cuerpo, el token ni la query string.** El cuerpo
   trae la pregunta (dato del usuario) y la respuesta trae fragmentos del
   corpus; el header trae la credencial; la query string es texto libre del
   cliente. El log dice método, ruta sin query y código de estado, con los
   caracteres de control escapados: una línea de log que un cliente puede
   fabricar es una traza de auditoría que ya no vale.

7. **El pipeline se importa recién cuando hace falta.** Importarlo carga el
   stack de ML (torch, sentence-transformers) y exige credenciales; acá se
   posterga para que `--help`, `/healthz` y los tests corran sin nada de eso.
   La función de consulta es inyectable justamente para que los tests la
   reemplacen por un doble y prueben el cableado HTTP sin el pipeline.

8. **El apagado espera a los requests en vuelo, con tope.** Ante SIGTERM se
   deja de aceptar conexiones y se esperan los handlers vivos hasta
   `RAGB_SHUTDOWN_TIMEOUT` segundos (que tiene que quedar por debajo del
   `terminationGracePeriodSeconds` del Deployment, menos el `preStop`). Los
   hilos son daemon: una consulta colgada no retiene el proceso más allá del
   tope, y el tope se registra en el log cuando se alcanza.

9. **Hay un tope de conexiones simultáneas.** `ThreadingHTTPServer` abre un
   hilo por conexión sin límite; con `RAGB_MAX_CONNECTIONS` el servidor
   contesta 503 en el propio hilo de aceptación cuando se pasa, en vez de
   crecer hasta que la liveness lo reinicie. Es un fusible, no un rate limit:
   ese va en el ingress, delante.

Sobre el modelo de confianza: el token es UNA credencial de servicio, y quien
la tiene elige el `tenant` del body. La API separa a los tenants entre sí
(LLM08: la búsqueda filtra por tenant en `store.py`), no protege a un tenant
de quien posee el token. Si distintos clientes necesitan tokens distintos, va
un gateway delante; esta capa no lo reemplaza.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import itertools
import json
import logging
import os
import re
import signal
import threading
import time
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

logger = logging.getLogger("ragb.api")

# 16 KiB alcanza para cualquier pregunta razonable y deja fuera un cuerpo
# arbitrariamente grande antes de leerlo: el límite se decide con el
# Content-Length, no después de haber cargado el body en memoria.
MAX_BODY_BYTES = 16 * 1024

# Cuando se rechaza un request sin leer su cuerpo, cerrar el socket con datos
# sin leer hace que el kernel mande RST en vez de FIN, y hay stacks TCP que
# descartan la respuesta ya enviada al recibir el RST: el cliente ve
# "connection reset" en lugar del 413. Se drena un tope acotado para que el
# rechazo llegue legible; más allá del tope, el RST es el precio de no leer.
# Para un cliente SIN credencial válida el tope es el del cuerpo legítimo
# (16 KiB): drenar más es trabajo gratis para quien no se autenticó.
MAX_DRAIN_BYTES = 64 * 1024

# Un top_k desmedido es una consulta cara a pgvector y un prompt enorme al
# modelo; el techo es generoso frente al default (5) pero finito.
MAX_TOP_K = 50
MAX_TENANT_CHARS = 64
# El tenant es una clave de aislamiento y un valor de filtro en SQL, no texto
# libre: identificador plano. Deja fuera saltos de línea (inyección en el
# log), espacios y cualquier cosa que un día pudiera confundir a un filtro.
TENANT_RE = re.compile(r"^[A-Za-z0-9._-]{1,%d}$" % MAX_TENANT_CHARS)

# Defaults operativos; el ConfigMap del Deployment los fija explícitamente.
DEFAULT_MAX_CONNECTIONS = 64
DEFAULT_SHUTDOWN_TIMEOUT = 20.0
DEFAULT_WARMUP_RETRY_SECONDS = 15.0

# Escape de caracteres de control para el log, equivalente al que aplica la
# librería estándar en su `log_message` (que acá se reemplaza). Se define
# propio para no depender de un atributo privado del handler.
_TABLA_CONTROL = str.maketrans(
    {c: rf"\x{c:02x}" for c in itertools.chain(range(0x20), range(0x7F, 0xA0))} | {ord("\\"): r"\\"}
)

QueryFn = Callable[..., dict[str, Any]]


def _escapar(texto: str) -> str:
    return texto.translate(_TABLA_CONTROL)


# --- Configuración ----------------------------------------------------------


VARIABLES_OBLIGATORIAS = ("DATABASE_URL", "ANTHROPIC_API_KEY", "RAGB_API_TOKEN")


def config_faltante() -> list[str]:
    """Nombres de las variables de entorno obligatorias que están vacías."""
    return [nombre for nombre in VARIABLES_OBLIGATORIAS if not os.getenv(nombre, "").strip()]


def _token_configurado() -> str:
    return os.getenv("RAGB_API_TOKEN", "").strip()


def _activado(nombre: str) -> bool:
    return os.getenv(nombre, "").strip().lower() in {"1", "true", "yes", "on"}


def _token_valido(presentado: str | None, esperado: str) -> bool:
    """Comparación en tiempo constante.

    `compare_digest` es constante solo entre cadenas del mismo largo; se
    comparan los SHA-256 de ambas para que el largo del token real no se
    filtre por el tiempo de respuesta.
    """
    if presentado is None:
        return False
    a = hashlib.sha256(presentado.encode("utf-8")).digest()
    b = hashlib.sha256(esperado.encode("utf-8")).digest()
    return hmac.compare_digest(a, b)


def _bearer(cabecera: str | None) -> str | None:
    if not cabecera:
        return None
    esquema, _, valor = cabecera.strip().partition(" ")
    if esquema.lower() != "bearer" or not valor.strip():
        return None
    return valor.strip()


# --- Modelo de embeddings: carga única y calentamiento ----------------------


def _cargar_embeddings() -> None:
    """Fuerza la carga del modelo. Import perezoso: ver la decisión 7."""
    from . import embeddings

    embeddings._model()


class EstadoModelo:
    """Una sola carga del modelo por proceso, visible desde `/readyz`.

    `lru_cache` no protege la PRIMERA ejecución: dos hilos que llegan antes
    de que termine la carga la ejecutan los dos. Acá el candado se toma solo
    hasta que `listo` es True; después ningún request lo toca.
    """

    def __init__(self, cargar: Callable[[], None] | None = None):
        self.requerido = False  # True cuando /readyz debe esperar la carga
        self.listo = False
        self.error: str | None = None
        self._lock = threading.Lock()
        self._cargar = cargar or _cargar_embeddings

    def asegurar(self) -> None:
        if self.listo:
            return
        with self._lock:
            if self.listo:
                return
            try:
                self._cargar()
            except Exception as exc:
                # Se guarda el tipo para el log; el detalle no sale por /readyz.
                self.error = f"{type(exc).__name__}: {exc}"
                raise
            self.error = None
            self.listo = True


# Único por proceso: lo comparten el calentamiento y las consultas.
MODELO = EstadoModelo()


def calentar(
    modelo: EstadoModelo,
    *,
    reintento_s: float = DEFAULT_WARMUP_RETRY_SECONDS,
    dormir: Callable[[float], None] = time.sleep,
    intentos: int | None = None,
) -> bool:
    """Carga el modelo con reintentos. Pensado para correr en un hilo.

    Una descarga que falla (red, proxy, Hugging Face caído) no tumba el pod:
    `/healthz` sigue en 200, `/readyz` sigue en 503 diciendo que carga, y el
    motivo va al log. Reintenta hasta lograrlo o hasta `intentos` (None =
    para siempre, que es lo que quiere un pod: seguir intentando hasta que
    alguien lo mate a propósito).
    """
    fallos = 0
    while True:
        try:
            modelo.asegurar()
        except Exception:
            fallos += 1
            logger.exception("el modelo de embeddings no cargó (intento %d); /readyz sigue en 503", fallos)
            if intentos is not None and fallos >= intentos:
                return False
            dormir(reintento_s)
            continue
        logger.info("modelo de embeddings listo")
        return True


# --- Pipeline por defecto ---------------------------------------------------


def query_por_defecto(*, question: str, tenant: str, top_k: int | None) -> dict[str, Any]:
    """Consulta real. Import perezoso: ver la decisión 7 del módulo."""
    from .guards import Permissions
    from . import pipeline

    # Aunque no haya calentamiento, la primera carga pasa por el candado.
    MODELO.asegurar()
    perms = Permissions(tenant=tenant, can_read=True)
    return pipeline.query(perms, question, top_k=top_k)


# --- Validación del body ----------------------------------------------------


def validar_body(body: bytes) -> tuple[dict[str, Any] | None, str | None]:
    """Devuelve (parámetros, None) o (None, motivo). Estricto a propósito.

    Un JSON con claves de más se rechaza: si mañana alguien agrega un campo
    "collection" del lado del cliente, más vale que se entere hoy con un 400
    que descubrir en producción que se ignoraba en silencio.
    """
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, "el cuerpo debe ser JSON UTF-8 válido"
    if not isinstance(data, dict):
        return None, "el cuerpo debe ser un objeto JSON"

    extras = set(data) - {"question", "tenant", "top_k"}
    if extras:
        return None, f"claves no reconocidas: {', '.join(sorted(extras))}"

    question = data.get("question")
    if not isinstance(question, str) or not question.strip():
        return None, "'question' debe ser una cadena no vacía"

    tenant = data.get("tenant")
    if not isinstance(tenant, str) or not tenant.strip():
        return None, "'tenant' debe ser una cadena no vacía"
    tenant = tenant.strip()
    if len(tenant) > MAX_TENANT_CHARS:
        return None, f"'tenant' supera {MAX_TENANT_CHARS} caracteres"
    if not TENANT_RE.match(tenant):
        return None, "'tenant' debe ser un identificador: letras, dígitos, punto, guion o guion bajo"

    top_k = data.get("top_k")
    # `bool` es subclase de `int`: sin esta exclusión, `true` pasaría como 1.
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int)):
        return None, "'top_k' debe ser un entero"
    if top_k is not None and not 1 <= top_k <= MAX_TOP_K:
        return None, f"'top_k' debe estar entre 1 y {MAX_TOP_K}"

    return {"question": question.strip(), "tenant": tenant, "top_k": top_k}, None


# --- Servidor ---------------------------------------------------------------


def _respuesta_cruda(estado: HTTPStatus, cuerpo: dict[str, Any]) -> bytes:
    """Una respuesta HTTP/1.0 completa, para escribir sin pasar por el handler."""
    datos = json.dumps(cuerpo, ensure_ascii=False).encode("utf-8")
    cabeceras = (
        f"HTTP/1.0 {estado.value} {estado.phrase}\r\n"
        "Server: ragb\r\n"
        "Content-Type: application/json; charset=utf-8\r\n"
        f"Content-Length: {len(datos)}\r\n"
        "Cache-Control: no-store\r\n"
        "Connection: close\r\n\r\n"
    )
    return cabeceras.encode("ascii") + datos


class RagbServer(ThreadingHTTPServer):
    """Lleva la función de consulta y el estado del modelo, para que el
    handler no dependa de globales, y controla sus propios hilos.

    `ThreadingMixIn` no registra los hilos daemon, así que su `server_close()`
    no espera a ninguno; acá se llevan aparte para poder esperar CON TOPE
    (decisión 8) y para contar cuántos hay (decisión 9).
    """

    # Los hilos siguen siendo daemon: una consulta colgada no retiene el
    # proceso más allá del tope de apagado. La espera la hace `server_close`.
    daemon_threads = True
    # Reutilizar el puerto evita el TIME_WAIT al reiniciar el pod.
    allow_reuse_address = True

    def __init__(
        self,
        direccion: tuple[str, int],
        query_fn: QueryFn | None = None,
        *,
        modelo: EstadoModelo | None = None,
        max_conexiones: int = DEFAULT_MAX_CONNECTIONS,
        shutdown_timeout: float = DEFAULT_SHUTDOWN_TIMEOUT,
    ):
        super().__init__(direccion, Handler)
        self.query_fn: QueryFn = query_fn or query_por_defecto
        self.modelo = modelo or MODELO
        self.max_conexiones = max_conexiones
        self.shutdown_timeout = shutdown_timeout
        self._en_vuelo: set[threading.Thread] = set()
        self._en_vuelo_lock = threading.Lock()

    # -- hilos --

    def conexiones_activas(self) -> int:
        with self._en_vuelo_lock:
            return len(self._en_vuelo)

    def process_request(self, request, client_address) -> None:
        """Un hilo por conexión, hasta el tope; por encima, 503 sin hilo."""
        with self._en_vuelo_lock:
            activas = len(self._en_vuelo)
            saturado = activas >= self.max_conexiones
            if not saturado:
                hilo = threading.Thread(
                    target=self._atender,
                    args=(request, client_address),
                    daemon=True,
                    name="ragb-conn",
                )
                self._en_vuelo.add(hilo)
        if saturado:
            self._rechazar_saturado(request, client_address, activas)
            return
        hilo.start()

    def _atender(self, request, client_address) -> None:
        try:
            self.process_request_thread(request, client_address)
        finally:
            with self._en_vuelo_lock:
                self._en_vuelo.discard(threading.current_thread())

    def _rechazar_saturado(self, request, client_address, activas: int) -> None:
        # Se escribe en el hilo de aceptación: son ~200 bytes, entran en el
        # buffer de envío sin bloquear; el timeout es el seguro por si no.
        logger.warning("%s rechazado: %d conexiones activas (tope %d)", client_address[0], activas, self.max_conexiones)
        try:
            request.settimeout(1.0)
            request.sendall(_respuesta_cruda(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "servidor saturado"}))
        except OSError:
            pass
        finally:
            self.shutdown_request(request)

    def server_close(self) -> None:
        """Cierra el socket de escucha y espera a los requests en vuelo, con tope."""
        super().server_close()
        limite = time.monotonic() + self.shutdown_timeout
        with self._en_vuelo_lock:
            hilos = list(self._en_vuelo)
        for hilo in hilos:
            restante = limite - time.monotonic()
            if restante <= 0:
                break
            hilo.join(timeout=restante)
        vivos = sum(1 for h in hilos if h.is_alive())
        if vivos:
            logger.warning("%d requests no terminaron en %.0f s; se cortan", vivos, self.shutdown_timeout)


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.0 cierra la conexión tras cada respuesta. Con keep-alive, un
    # cuerpo rechazado sin leer quedaría en el socket y se interpretaría como
    # el request siguiente; cerrar siempre es lo más simple y lo más seguro.
    protocol_version = "HTTP/1.0"
    # Timeout de LECTURA del socket, no por request: protege contra un cliente
    # que declara Content-Length y se calla, y contra uno que abre y no manda
    # nada. NO protege contra slowloris (un byte cada <30 s lo mantiene vivo):
    # eso lo cierra el tope de conexiones del servidor como fusible y el
    # ingress como defensa real (client_header_timeout, límite por IP).
    timeout = 30
    server: RagbServer

    # Sin banner de versión de Python en el header `Server`.
    def version_string(self) -> str:
        return "ragb"

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        # Método, ruta SIN query string y código: la query string es texto
        # libre del cliente y no tiene por qué quedar en el log.
        ruta = (self.path or "").split("?", 1)[0]
        if isinstance(code, HTTPStatus):
            code = code.value
        self.log_message('"%s %s" %s', self.command, ruta, code)

    def log_message(self, formato: str, *args: Any) -> None:
        # Se reemplaza el log por defecto (que escribe a stderr) por `logging`,
        # con los caracteres de control escapados: `\n` o ESC dentro de la
        # línea de request no pueden fabricar líneas de log ni colorear una
        # terminal. Nunca headers, nunca body.
        logger.info("%s %s", self.address_string(), _escapar(formato % args))

    # -- helpers --

    def _json(self, estado: HTTPStatus, cuerpo: dict[str, Any]) -> None:
        datos = json.dumps(cuerpo, ensure_ascii=False).encode("utf-8")
        self.send_response(estado)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(datos)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(datos)

    def _drenar(self, largo: int, tope: int) -> None:
        restante = min(largo, tope)
        while restante > 0:
            trozo = self.rfile.read(min(restante, 8192))
            if not trozo:
                break
            restante -= len(trozo)

    def _content_length(self) -> int | None:
        crudo = self.headers.get("Content-Length")
        if crudo is None:
            return None
        # RFC 9110: 1*DIGIT. `int()` aceptaría "+16", "1_6" o " 1 6"; el
        # header que decide el 413 no puede ser más laxo que el body.
        crudo = crudo.strip()
        if not re.fullmatch(r"[0-9]+", crudo):
            return -1
        return int(crudo)

    def _rechazar(self, estado: HTTPStatus, error: str, *, tope: int = MAX_BODY_BYTES) -> None:
        """Responde un error y drena (hasta `tope`) el cuerpo que no se va a leer."""
        largo = self._content_length()
        self._json(estado, {"error": error})
        if largo and largo > 0:
            self._drenar(largo, tope)

    # -- rutas --

    def do_GET(self) -> None:
        ruta = self.path.split("?", 1)[0]
        if ruta == "/healthz":
            self._json(HTTPStatus.OK, {"ok": True})
        elif ruta == "/readyz":
            faltan = config_faltante()
            modelo = self.server.modelo
            if faltan:
                self._json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"ok": False, "reason": f"faltan variables: {', '.join(faltan)}"},
                )
            elif modelo.requerido and not modelo.listo:
                # Sin detalle del error hacia afuera: /readyz no exige token y
                # el mensaje de una descarga fallida puede nombrar hosts
                # internos. El detalle está en el log del pod.
                self._json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"ok": False, "reason": "modelo de embeddings cargando"},
                )
            else:
                self._json(HTTPStatus.OK, {"ok": True})
        elif ruta == "/query":
            self._json(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "usar POST"})
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "ruta desconocida"})

    def do_POST(self) -> None:
        ruta = self.path.split("?", 1)[0]
        if ruta != "/query":
            self._rechazar(HTTPStatus.NOT_FOUND, "ruta desconocida")
            return

        # 1. Cerrado por defecto: sin token configurado no hay endpoint.
        esperado = _token_configurado()
        if not esperado:
            self._rechazar(HTTPStatus.SERVICE_UNAVAILABLE, "RAGB_API_TOKEN no configurado")
            return

        # 2. Autenticación antes de mirar el cuerpo: un cliente sin credencial
        #    no consigue que el servidor parsee nada suyo, y solo se le drena
        #    lo que un cuerpo legítimo podría medir.
        if not _token_valido(_bearer(self.headers.get("Authorization")), esperado):
            self._rechazar(HTTPStatus.UNAUTHORIZED, "token inválido")
            return

        # 3. Tamaño, decidido por la cabecera antes de leer un byte del cuerpo.
        largo = self._content_length()
        if largo is None:
            self._json(HTTPStatus.LENGTH_REQUIRED, {"error": "falta Content-Length"})
            return
        if largo < 0:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "Content-Length inválido"})
            return
        if largo > MAX_BODY_BYTES:
            self._rechazar(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"el cuerpo supera {MAX_BODY_BYTES} bytes",
                tope=MAX_DRAIN_BYTES,
            )
            return

        # 4. Validación estricta del contenido.
        params, motivo = validar_body(self.rfile.read(largo))
        if motivo:
            self._json(HTTPStatus.BAD_REQUEST, {"error": motivo})
            return
        assert params is not None

        # 5. La consulta. Un PermissionError es la guarda LLM06 hablando y se
        #    traduce a 403; cualquier otra falla es un 500 sin detalle hacia
        #    afuera (el detalle va al log del proceso, sin el body). El tenant
        #    ya pasó por TENANT_RE, así que puede ir al log tal cual.
        try:
            resultado = self.server.query_fn(**params)
        except PermissionError as exc:
            self._json(HTTPStatus.FORBIDDEN, {"error": str(exc)})
            return
        except Exception:
            logger.exception("fallo interno atendiendo /query (tenant=%s)", params["tenant"])
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "fallo interno"})
            return

        self._json(HTTPStatus.OK, resultado)


# --- Arranque ---------------------------------------------------------------


def make_server(
    host: str,
    port: int,
    query_fn: QueryFn | None = None,
    *,
    modelo: EstadoModelo | None = None,
    max_conexiones: int = DEFAULT_MAX_CONNECTIONS,
    shutdown_timeout: float = DEFAULT_SHUTDOWN_TIMEOUT,
) -> RagbServer:
    """Crea el servidor sin arrancarlo. `port=0` pide un puerto libre."""
    return RagbServer(
        (host, port),
        query_fn,
        modelo=modelo,
        max_conexiones=max_conexiones,
        shutdown_timeout=shutdown_timeout,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragb.api", description=__doc__.split("\n\n")[0])
    # 127.0.0.1 por defecto: escuchar en todas las interfaces es una decisión
    # de quien despliega (el Dockerfile la toma explícitamente), no un default.
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--log-level", default="INFO")
    return parser


def _entero_env(nombre: str, default: int) -> int:
    crudo = os.getenv(nombre, "").strip()
    return int(crudo) if crudo.isdigit() else default


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    server = make_server(
        args.host,
        args.port,
        max_conexiones=_entero_env("RAGB_MAX_CONNECTIONS", DEFAULT_MAX_CONNECTIONS),
        shutdown_timeout=float(_entero_env("RAGB_SHUTDOWN_TIMEOUT", int(DEFAULT_SHUTDOWN_TIMEOUT))),
    )

    # Kubernetes manda SIGTERM y espera `terminationGracePeriodSeconds`.
    # `shutdown()` bloquea hasta que `serve_forever` suelta, así que se llama
    # desde otro hilo. Los nuevos requests ya no entran; los que están en
    # vuelo los espera `server_close()` hasta `shutdown_timeout` (decisión 8).
    def _apagar(*_: Any) -> None:
        logger.info("señal de término recibida: dejo de aceptar y espero hasta %.0f s", server.shutdown_timeout)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _apagar)
    signal.signal(signal.SIGINT, _apagar)

    faltan = config_faltante()
    if faltan:
        logger.warning("arrancando sin configuración completa: faltan %s (/readyz responderá 503)", ", ".join(faltan))

    # Decisión 5: con RAGB_WARMUP el modelo se carga en un hilo desde ya y
    # /readyz no da 200 hasta que esté. Sin la variable, carga perezoso.
    if _activado("RAGB_WARMUP"):
        server.modelo.requerido = True
        threading.Thread(target=calentar, args=(server.modelo,), daemon=True, name="ragb-warmup").start()
        logger.info("calentando el modelo de embeddings en segundo plano")

    logger.info("escuchando en http://%s:%s", args.host, server.server_address[1])
    try:
        server.serve_forever()
    finally:
        server.server_close()
    logger.info("servidor cerrado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
