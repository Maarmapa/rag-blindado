"""El servidor HTTP: sondas, autenticación cerrada por defecto y cableado.

Se levanta el servidor real en un hilo, en un puerto libre, y se le habla por
`http.client`. Lo que se prueba es el borde HTTP —qué código devuelve cada
situación, qué llega y qué no llega al pipeline— con la función de consulta
reemplazada por un doble. Corre sin credenciales, sin base y sin red externa:
es lo que el job `guards` de CI puede ejecutar.
"""

from __future__ import annotations

import http.client
import json
import logging
import pathlib
import socket
import subprocess
import sys
import threading
import time

import pytest

from ragb import api

TOKEN = "token-de-prueba-suficientemente-largo-0123456789"
RAIZ = pathlib.Path(__file__).resolve().parent.parent


class Doble:
    """Registra con qué se la llamó y devuelve una traza mínima."""

    def __init__(self):
        self.llamadas: list[dict] = []
        self.falla: Exception | None = None

    def __call__(self, *, question, tenant, top_k):
        self.llamadas.append({"question": question, "tenant": tenant, "top_k": top_k})
        if self.falla is not None:
            raise self.falla
        return {"text": "respuesta", "sources": ["sla.md"], "question": question}


def _levantar(doble, **kwargs):
    # Cada servidor lleva su propio EstadoModelo: el singleton `api.MODELO`
    # es compartido por proceso y un test que lo marcara "requerido" se lo
    # dejaría al siguiente.
    srv = api.make_server("127.0.0.1", 0, query_fn=doble, modelo=api.EstadoModelo(), **kwargs)
    # `shutdown()` espera un tick de `serve_forever`; con el poll por defecto
    # (0.5 s) cada test pagaba medio segundo de teardown sin hacer nada.
    hilo = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    hilo.start()
    return srv, hilo


@pytest.fixture
def servidor():
    doble = Doble()
    srv, hilo = _levantar(doble)
    try:
        yield srv, doble
    finally:
        srv.shutdown()
        srv.server_close()
        hilo.join(timeout=5)


@pytest.fixture
def configurado(monkeypatch):
    """Configuración completa: token, base y clave. Los tests la desarman a gusto."""
    monkeypatch.setenv("RAGB_API_TOKEN", TOKEN)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db:5432/ragb")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-de-prueba")


def _pedir(srv, metodo, ruta, cuerpo=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
    try:
        conn.request(metodo, ruta, body=cuerpo, headers=headers or {})
        resp = conn.getresponse()
        datos = resp.read()
        return resp.status, dict(resp.getheaders()), datos
    finally:
        conn.close()


def _post(srv, cuerpo, token=TOKEN, headers=None):
    h = {"Content-Type": "application/json"}
    if token is not None:
        h["Authorization"] = f"Bearer {token}"
    h.update(headers or {})
    if isinstance(cuerpo, (dict, list)):
        cuerpo = json.dumps(cuerpo).encode("utf-8")
    return _pedir(srv, "POST", "/query", cuerpo, h)


# --- Sondas -----------------------------------------------------------------


def test_healthz_responde_200_sin_configuracion(servidor, monkeypatch):
    """Liveness: el proceso vive aunque no tenga ni una variable. Matarlo no lo
    arreglaría; por eso no mira la configuración."""
    for nombre in ("RAGB_API_TOKEN", "DATABASE_URL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(nombre, raising=False)
    srv, _ = servidor
    estado, headers, cuerpo = _pedir(srv, "GET", "/healthz")
    assert estado == 200
    assert json.loads(cuerpo) == {"ok": True}
    assert headers["Content-Type"] == "application/json; charset=utf-8"


def test_readyz_200_con_configuracion_completa(servidor, configurado):
    srv, _ = servidor
    estado, _, cuerpo = _pedir(srv, "GET", "/readyz")
    assert estado == 200
    assert json.loads(cuerpo) == {"ok": True}


def test_readyz_503_y_nombra_lo_que_falta(servidor, configurado, monkeypatch):
    """Readiness cerrada: sin base no se atiende, y el motivo se dice."""
    monkeypatch.setenv("DATABASE_URL", "")
    srv, _ = servidor
    estado, _, cuerpo = _pedir(srv, "GET", "/readyz")
    assert estado == 503
    assert json.loads(cuerpo) == {"ok": False, "reason": "faltan variables: DATABASE_URL"}


def test_readyz_refleja_el_entorno_actual_no_el_del_import(servidor, configurado, monkeypatch):
    """`ragb.config.settings` congela el entorno al importar; la sonda no puede
    heredar ese vicio, porque su trabajo es decir cómo está el pod AHORA."""
    srv, _ = servidor
    assert _pedir(srv, "GET", "/readyz")[0] == 200
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    estado, _, cuerpo = _pedir(srv, "GET", "/readyz")
    assert estado == 503
    assert "ANTHROPIC_API_KEY" in json.loads(cuerpo)["reason"]


def test_readyz_503_sin_token_porque_query_daria_503_a_todo(servidor, configurado, monkeypatch):
    """Un pod sin RAGB_API_TOKEN rechaza el 100 % de las consultas: decir
    "listo" sería mentir, y con maxUnavailable 0 un rollout lo daría por bueno."""
    monkeypatch.delenv("RAGB_API_TOKEN")
    srv, _ = servidor
    estado, _, cuerpo = _pedir(srv, "GET", "/readyz")
    assert estado == 503
    assert json.loads(cuerpo) == {"ok": False, "reason": "faltan variables: RAGB_API_TOKEN"}


def test_readyz_503_mientras_el_modelo_carga_y_200_cuando_esta(servidor, configurado):
    """Con el calentamiento pedido, la readiness espera al modelo. El motivo es
    genérico a propósito: /readyz no exige token y el error de una descarga
    puede nombrar hosts internos; ese detalle vive en el log."""
    srv, _ = servidor
    srv.modelo.requerido = True
    srv.modelo.error = "OSError: proxy-interno.corp:3128 rechazó la conexión"
    estado, _, cuerpo = _pedir(srv, "GET", "/readyz")
    assert estado == 503
    assert json.loads(cuerpo) == {"ok": False, "reason": "modelo de embeddings cargando"}
    assert b"proxy-interno" not in cuerpo

    srv.modelo.listo = True
    assert _pedir(srv, "GET", "/readyz")[0] == 200
    # /healthz nunca espera al modelo: reiniciar el pod no lo cargaría más rápido.
    srv.modelo.listo = False
    assert _pedir(srv, "GET", "/healthz")[0] == 200


def test_ruta_desconocida_404_y_get_a_query_405(servidor):
    srv, _ = servidor
    assert _pedir(srv, "GET", "/otra")[0] == 404
    assert _pedir(srv, "GET", "/query")[0] == 405
    assert _pedir(srv, "POST", "/otra", b"{}", {"Content-Length": "2"})[0] == 404


# --- Autenticación ----------------------------------------------------------


def test_sin_token_configurado_query_es_503_no_abierto(servidor, configurado, monkeypatch):
    """La regla central: sin credencial configurada el endpoint NO existe.
    Ni siquiera con un token cualquiera en el header."""
    monkeypatch.delenv("RAGB_API_TOKEN")
    srv, doble = servidor
    estado, _, cuerpo = _post(srv, {"question": "¿plazo?", "tenant": "acme"})
    assert estado == 503
    assert json.loads(cuerpo) == {"error": "RAGB_API_TOKEN no configurado"}
    assert doble.llamadas == []


def test_token_incorrecto_401_y_no_llega_al_pipeline(servidor, configurado):
    srv, doble = servidor
    estado, _, cuerpo = _post(srv, {"question": "¿plazo?", "tenant": "acme"}, token="otro")
    assert estado == 401
    assert json.loads(cuerpo) == {"error": "token inválido"}
    assert doble.llamadas == []


def test_sin_header_o_con_esquema_distinto_401(servidor, configurado):
    srv, doble = servidor
    assert _post(srv, {"question": "q", "tenant": "t"}, token=None)[0] == 401
    estado, _, _ = _post(srv, {"question": "q", "tenant": "t"}, token=None, headers={"Authorization": f"Basic {TOKEN}"})
    assert estado == 401
    assert doble.llamadas == []


def test_comparacion_de_token_es_en_tiempo_constante():
    """No se puede medir el tiempo en un test determinista; sí se puede afirmar
    que el largo del token no decide el resultado por un atajo, y que la
    comparación pasa por `hmac.compare_digest` (test de cableado)."""
    import inspect

    assert api._token_valido(TOKEN, TOKEN) is True
    assert api._token_valido(TOKEN[:-1], TOKEN) is False
    assert api._token_valido(TOKEN + "x", TOKEN) is False
    assert api._token_valido(None, TOKEN) is False
    assert "hmac.compare_digest" in inspect.getsource(api._token_valido)


# --- Validación del cuerpo --------------------------------------------------


@pytest.mark.parametrize(
    "cuerpo, motivo",
    [
        (b"no es json", "el cuerpo debe ser JSON UTF-8 válido"),
        (b"[1, 2]", "el cuerpo debe ser un objeto JSON"),
        ({"tenant": "acme"}, "'question' debe ser una cadena no vacía"),
        ({"question": "  ", "tenant": "acme"}, "'question' debe ser una cadena no vacía"),
        ({"question": "q"}, "'tenant' debe ser una cadena no vacía"),
        ({"question": "q", "tenant": 7}, "'tenant' debe ser una cadena no vacía"),
        ({"question": "q", "tenant": "t", "top_k": "3"}, "'top_k' debe ser un entero"),
        ({"question": "q", "tenant": "t", "top_k": True}, "'top_k' debe ser un entero"),
        ({"question": "q", "tenant": "t", "top_k": 0}, "'top_k' debe estar entre 1 y 50"),
        ({"question": "q", "tenant": "t", "top_k": 51}, "'top_k' debe estar entre 1 y 50"),
        ({"question": "q", "tenant": "t", "collection": "x"}, "claves no reconocidas: collection"),
        # El tenant es clave de aislamiento y valor de filtro SQL: identificador
        # plano. Un salto de línea adentro fabricaba líneas de log completas.
        (
            {"question": "q", "tenant": "acme\nERROR ragb.api: FORJADO tenant=otro"},
            "'tenant' debe ser un identificador: letras, dígitos, punto, guion o guion bajo",
        ),
        (
            {"question": "q", "tenant": "acme corp"},
            "'tenant' debe ser un identificador: letras, dígitos, punto, guion o guion bajo",
        ),
        (
            {"question": "q", "tenant": "acme;drop"},
            "'tenant' debe ser un identificador: letras, dígitos, punto, guion o guion bajo",
        ),
    ],
)
def test_body_invalido_400_con_motivo(servidor, configurado, cuerpo, motivo):
    srv, doble = servidor
    estado, _, respuesta = _post(srv, cuerpo)
    assert estado == 400
    assert json.loads(respuesta) == {"error": motivo}
    assert doble.llamadas == []


def test_tenant_identificador_valido_pasa_tal_cual(servidor, configurado):
    srv, doble = servidor
    assert _post(srv, {"question": "q", "tenant": " Cliente_2.prod-eu "})[0] == 200
    assert doble.llamadas[-1]["tenant"] == "Cliente_2.prod-eu"


@pytest.mark.parametrize("valor", ["1_6", "+16", "0x10", "16.0", "1 6"])
def test_content_length_que_no_es_solo_digitos_es_400(servidor, configurado, valor):
    """RFC 9110: 1*DIGIT. `int()` acepta '1_6' y '+16'; la cabecera que decide
    el 413 no puede ser más laxa que el cuerpo que valida."""
    srv, doble = servidor
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
    try:
        # Sin cuerpo a propósito: el rechazo tiene que salir de la cabecera.
        conn.putrequest("POST", "/query")
        conn.putheader("Authorization", f"Bearer {TOKEN}")
        conn.putheader("Content-Length", valor)
        conn.endheaders()
        resp = conn.getresponse()
        assert resp.status == 400
        assert json.loads(resp.read()) == {"error": "Content-Length inválido"}
    finally:
        conn.close()
    assert doble.llamadas == []


def test_body_sobre_16_kib_es_413_sin_llamar_al_pipeline(servidor, configurado):
    """El límite se decide por Content-Length. El cuerpo es JSON válido a
    propósito: lo que se rechaza es el tamaño, no el contenido."""
    srv, doble = servidor
    relleno = "x" * (16 * 1024)
    estado, _, cuerpo = _post(srv, {"question": relleno, "tenant": "acme"})
    assert estado == 413
    assert json.loads(cuerpo) == {"error": f"el cuerpo supera {api.MAX_BODY_BYTES} bytes"}
    assert doble.llamadas == []


def test_exactamente_16_kib_pasa(servidor, configurado):
    srv, doble = servidor
    base = json.dumps({"question": "", "tenant": "acme"}).encode()
    cuerpo = json.dumps({"question": "q" * (api.MAX_BODY_BYTES - len(base)), "tenant": "acme"}).encode()
    assert len(cuerpo) == api.MAX_BODY_BYTES
    assert _post(srv, cuerpo)[0] == 200
    assert len(doble.llamadas) == 1


def test_sin_content_length_411(servidor, configurado):
    srv, _ = servidor
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
    try:
        conn.putrequest("POST", "/query")
        conn.putheader("Authorization", f"Bearer {TOKEN}")
        conn.endheaders()
        resp = conn.getresponse()
        assert resp.status == 411
    finally:
        conn.close()


# --- Cableado con el pipeline -----------------------------------------------


def test_consulta_valida_200_y_el_doble_recibe_tenant_y_question(servidor, configurado):
    srv, doble = servidor
    estado, headers, cuerpo = _post(
        srv, {"question": "¿Cuál es el plazo de severidad 1?", "tenant": "acme", "top_k": 3}
    )
    assert estado == 200
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    assert headers["Cache-Control"] == "no-store"
    assert json.loads(cuerpo) == {
        "text": "respuesta",
        "sources": ["sla.md"],
        "question": "¿Cuál es el plazo de severidad 1?",
    }
    assert doble.llamadas == [{"question": "¿Cuál es el plazo de severidad 1?", "tenant": "acme", "top_k": 3}]


def test_top_k_ausente_llega_como_none(servidor, configurado):
    """None deja que el pipeline aplique su default (`settings.top_k`)."""
    srv, doble = servidor
    assert _post(srv, {"question": "q", "tenant": "acme"})[0] == 200
    assert doble.llamadas == [{"question": "q", "tenant": "acme", "top_k": None}]


def test_permission_error_del_pipeline_es_403(servidor, configurado):
    """La guarda LLM06 hablando: se traduce, no se esconde en un 500."""
    srv, doble = servidor
    doble.falla = PermissionError("lectura no autorizada para este principal")
    estado, _, cuerpo = _post(srv, {"question": "q", "tenant": "acme"})
    assert estado == 403
    assert json.loads(cuerpo) == {"error": "lectura no autorizada para este principal"}


def test_fallo_interno_500_sin_detalle_hacia_afuera(servidor, configurado, caplog):
    srv, doble = servidor
    doble.falla = RuntimeError("psycopg: connection refused host=db-secreta")
    with caplog.at_level(logging.ERROR, logger="ragb.api"):
        estado, _, cuerpo = _post(srv, {"question": "q", "tenant": "acme"})
    assert estado == 500
    assert json.loads(cuerpo) == {"error": "fallo interno"}
    # El detalle sí queda en el log del proceso, que es donde se depura.
    assert "db-secreta" in caplog.text


def test_el_log_no_contiene_ni_el_token_ni_el_cuerpo(servidor, configurado, caplog):
    """Lo que se afirma es la ausencia: la pregunta y la credencial no pueden
    aparecer en el log, ni en el request exitoso ni en el fallido."""
    srv, doble = servidor
    pregunta = "PREGUNTA-CONFIDENCIAL-7f3a"
    with caplog.at_level(logging.DEBUG, logger="ragb.api"):
        assert _post(srv, {"question": pregunta, "tenant": "acme"})[0] == 200
        doble.falla = RuntimeError("boom")
        assert _post(srv, {"question": pregunta, "tenant": "acme"})[0] == 500
        assert _post(srv, {"question": pregunta, "tenant": "acme"}, token="malo")[0] == 401
    assert caplog.text  # el control: algo se registró
    assert "POST /query" in caplog.text
    assert TOKEN not in caplog.text
    assert pregunta not in caplog.text


def test_el_log_no_lleva_query_string_ni_caracteres_de_control(servidor, configurado, caplog):
    """La query string es texto libre del cliente; ESC y saltos de línea en la
    línea de request podían colorear una terminal o fabricar líneas enteras."""
    srv, _ = servidor
    with caplog.at_level(logging.DEBUG, logger="ragb.api"):
        assert (
            _pedir(
                srv,
                "POST",
                "/query?pregunta=SECRETO-EN-QUERYSTRING",
                b"{}",
                {
                    "Authorization": f"Bearer {TOKEN}",
                    "Content-Length": "2",
                },
            )[0]
            == 400
        )
        # Una request line con secuencias de control, cruda por el socket.
        with socket.create_connection(("127.0.0.1", srv.server_address[1]), timeout=5) as s:
            s.sendall(b"GET /healthz\x1b[31mROJO\x1b[0m\x07 HTTP/1.0\r\n\r\n")
            respuesta = s.recv(4096)
        assert respuesta.startswith(b"HTTP/1.0 404")
    assert "POST /query" in caplog.text
    assert "SECRETO-EN-QUERYSTRING" not in caplog.text
    assert "\x1b" not in caplog.text and "\x07" not in caplog.text
    assert r"\x1b" in caplog.text  # escapado, no borrado: la traza sigue diciendo qué llegó


# --- Apagado y saturación ---------------------------------------------------


class Lenta:
    """Doble que tarda: para ver qué pasa con un request en vuelo."""

    def __init__(self, segundos):
        self.segundos = segundos
        self.llamadas = 0

    def __call__(self, *, question, tenant, top_k):
        self.llamadas += 1
        time.sleep(self.segundos)
        return {"text": "tarde"}


def _post_en_hilo(srv, resultado):
    def cliente():
        try:
            resultado["status"] = _post(srv, {"question": "q", "tenant": "acme"})[0]
        except Exception as exc:  # pragma: no cover - solo para diagnosticar
            resultado["error"] = repr(exc)

    hilo = threading.Thread(target=cliente, daemon=True)
    hilo.start()
    return hilo


def test_el_apagado_espera_al_request_en_vuelo(configurado):
    """La regresión que motiva este test: con hilos daemon sin registrar,
    `server_close()` volvía en 0,03 s y el cliente veía RemoteDisconnected
    con la consulta ya pagada a Anthropic."""
    lenta = Lenta(0.6)
    srv, hilo_srv = _levantar(lenta, shutdown_timeout=5.0)
    resultado = {}
    cliente = _post_en_hilo(srv, resultado)
    for _ in range(100):  # hasta que el request esté dentro del doble
        if lenta.llamadas:
            break
        time.sleep(0.01)
    assert lenta.llamadas == 1

    t0 = time.monotonic()
    srv.shutdown()
    srv.server_close()
    espera = time.monotonic() - t0
    hilo_srv.join(timeout=5)
    cliente.join(timeout=5)

    assert resultado == {"status": 200}, resultado
    assert espera >= 0.3, f"server_close volvió en {espera:.2f}s sin esperar al handler"


def test_el_apagado_no_espera_mas_que_el_tope(configurado, caplog):
    """El tope existe para quedar por debajo del terminationGracePeriodSeconds:
    una consulta colgada no puede convertir el SIGTERM en un SIGKILL mudo."""
    lenta = Lenta(3.0)
    srv, hilo_srv = _levantar(lenta, shutdown_timeout=0.3)
    resultado = {}
    _post_en_hilo(srv, resultado)
    for _ in range(100):
        if lenta.llamadas:
            break
        time.sleep(0.01)

    t0 = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="ragb.api"):
        srv.shutdown()
        srv.server_close()
    espera = time.monotonic() - t0
    hilo_srv.join(timeout=5)

    assert 0.3 <= espera < 2.0, espera
    assert "1 requests no terminaron" in caplog.text


def test_por_encima_del_tope_de_conexiones_responde_503_sin_abrir_hilo(configurado, caplog):
    """Dos sockets que no mandan nada ocupan los dos hilos permitidos; el
    tercero recibe 503 desde el hilo de aceptación. Cuando los dos primeros
    se van, vuelve a atender."""
    doble = Doble()
    srv, hilo_srv = _levantar(doble, max_conexiones=2)
    ociosos = []
    try:
        for _ in range(2):
            ociosos.append(socket.create_connection(("127.0.0.1", srv.server_address[1]), timeout=5))
        for _ in range(100):
            if srv.conexiones_activas() == 2:
                break
            time.sleep(0.01)
        assert srv.conexiones_activas() == 2

        with caplog.at_level(logging.WARNING, logger="ragb.api"):
            estado, headers, cuerpo = _pedir(srv, "GET", "/healthz")
        assert estado == 503
        assert json.loads(cuerpo) == {"error": "servidor saturado"}
        assert headers["Content-Type"] == "application/json; charset=utf-8"
        assert "2 conexiones activas (tope 2)" in caplog.text
        assert srv.conexiones_activas() == 2  # el rechazo no abrió un hilo

        for s in ociosos:
            s.close()
        for _ in range(200):
            if srv.conexiones_activas() == 0:
                break
            time.sleep(0.01)
        assert _pedir(srv, "GET", "/healthz")[0] == 200
    finally:
        for s in ociosos:
            s.close()
        srv.shutdown()
        srv.server_close()
        hilo_srv.join(timeout=5)


# --- Carga del modelo -------------------------------------------------------


def test_el_modelo_se_carga_una_sola_vez_aunque_lleguen_hilos_a_la_vez():
    """`lru_cache` no serializa la primera ejecución: cuatro hilos en frío
    cargaban el modelo cuatro veces (medido). Con el candado, una."""
    cargas = []
    barrera = threading.Barrier(4)

    def cargar():
        cargas.append(threading.current_thread().name)
        time.sleep(0.1)

    modelo = api.EstadoModelo(cargar=cargar)

    def hilo():
        barrera.wait()
        modelo.asegurar()

    hilos = [threading.Thread(target=hilo, name=f"req-{i}") for i in range(4)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join(timeout=5)
    assert len(cargas) == 1, cargas
    assert modelo.listo is True and modelo.error is None


def test_calentar_reintenta_y_termina_listo_sin_tumbar_el_proceso(caplog):
    intentos = []

    def cargar():
        intentos.append(1)
        if len(intentos) < 3:
            raise OSError("huggingface.co: 403 por el proxy")

    modelo = api.EstadoModelo(cargar=cargar)
    modelo.requerido = True
    dormido = []
    with caplog.at_level(logging.INFO, logger="ragb.api"):
        assert api.calentar(modelo, reintento_s=7, dormir=dormido.append) is True
    assert len(intentos) == 3
    assert dormido == [7, 7]
    assert modelo.listo is True and modelo.error is None
    assert caplog.text.count("no cargó") == 2
    assert "modelo de embeddings listo" in caplog.text


def test_calentar_con_tope_de_intentos_devuelve_false_y_deja_el_error():
    def cargar():
        raise RuntimeError("sin espacio en /tmp")

    modelo = api.EstadoModelo(cargar=cargar)
    assert api.calentar(modelo, reintento_s=0, dormir=lambda s: None, intentos=2) is False
    assert modelo.listo is False
    assert modelo.error == "RuntimeError: sin espacio en /tmp"


# --- Arranque ---------------------------------------------------------------


def test_importar_api_no_carga_el_pipeline():
    """El import perezoso es lo que permite que /healthz y estos tests corran
    sin torch ni credenciales. Se comprueba en un intérprete limpio, porque en
    este proceso otro test podría haber importado el pipeline antes."""
    codigo = (
        "import sys, ragb.api; "
        "assert 'ragb.pipeline' not in sys.modules, 'pipeline importado al cargar api'; "
        "assert 'ragb.embeddings' not in sys.modules"
    )
    # `cwd` explícito: el subproceso no hereda el rootdir de pytest, y sin esto
    # el test solo pasaba cuando pytest se invocaba desde la raíz del repo.
    proc = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, cwd=str(RAIZ))
    assert proc.returncode == 0, proc.stderr


def test_cli_acepta_host_y_port_y_no_escucha_en_todas_las_interfaces_por_defecto():
    args = api.build_parser().parse_args(["--host", "0.0.0.0", "--port", "8080"])
    assert (args.host, args.port) == ("0.0.0.0", 8080)
    assert api.build_parser().parse_args([]).host == "127.0.0.1"


def test_query_por_defecto_construye_permissions_de_solo_lectura(monkeypatch):
    """Cableado del default: tenant del body, lectura sí, escritura no, y
    `top_k` viaja al pipeline. Se inyecta un `ragb.pipeline` falso en
    `sys.modules` para no cargar el stack de ML."""
    import types

    from ragb.guards import Permissions

    recibido = {}

    def query(perms, question, *, collection=None, top_k=None):
        recibido.update(perms=perms, question=question, collection=collection, top_k=top_k)
        return {"text": "ok"}

    falso = types.ModuleType("ragb.pipeline")
    falso.query = query
    monkeypatch.setitem(sys.modules, "ragb.pipeline", falso)
    # El default pasa por el candado del modelo antes de consultar; se
    # reemplaza el cargador para no importar sentence-transformers.
    cargas = []
    monkeypatch.setattr(api, "MODELO", api.EstadoModelo(cargar=lambda: cargas.append(1)))

    assert api.query_por_defecto(question="q", tenant="acme", top_k=4) == {"text": "ok"}
    assert api.query_por_defecto(question="q2", tenant="acme", top_k=4) == {"text": "ok"}
    assert cargas == [1], "el modelo se asegura una vez, no en cada consulta"
    assert recibido["perms"] == Permissions(tenant="acme", can_read=True)
    assert recibido["perms"].can_write is False
    assert recibido["perms"].dry_run is True
    assert (recibido["question"], recibido["top_k"], recibido["collection"]) == ("q2", 4, None)
