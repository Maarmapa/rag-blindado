# Desplegar rag-blindado en Kubernetes

El servicio es `ragb/api.py`: un servidor HTTP de librería estándar con tres
rutas (`/healthz`, `/readyz`, `POST /query`), cerrado por defecto —sin
`RAGB_API_TOKEN` el endpoint de consulta responde 503 y `/readyz` también—.
Este directorio trae la imagen (`Dockerfile`) y los manifiestos (`k8s/`),
sin Helm.

## 1. Construir la imagen

Desde la **raíz del repo** (el contexto incluye `ragb/` y `requirements.txt`):

```bash
docker build -f deploy/Dockerfile -t ghcr.io/maarmapa/rag-blindado:0.1.0 .
docker push ghcr.io/maarmapa/rag-blindado:0.1.0
```

El tag es explícito y coincide con `image:` en `k8s/deployment.yaml`. Nunca
`latest`: el test `tests/test_k8s_manifests.py` lo rechaza. Mientras la
referencia sea un tag, el Deployment lleva `imagePullPolicy: Always` (un tag
es mutable; sin eso dos nodos pueden correr imágenes distintas bajo el mismo
nombre). Para producción, fijar además el digest (`0.1.0@sha256:…`).

La base (`python:3.12-slim`) va con tag **y** digest en el `FROM`, y torch se
instala primero desde el índice de CPU (`download.pytorch.org/whl/cpu`): la
rueda de PyPI para Linux pesa ~530 MB y arrastra las librerías CUDA, varios
GB que un pod sin GPU no usa.

Probar la imagen en local antes de subirla:

```bash
docker run --rm -p 8080:8080 ghcr.io/maarmapa/rag-blindado:0.1.0
curl -s localhost:8080/healthz   # {"ok": true}
curl -s localhost:8080/readyz    # 503: faltan variables — correcto, no hay Secret
```

### Pendiente conocido: pines exactos

`pip install -r requirements.txt` resuelve versiones en el momento del build;
dos builds del mismo commit pueden no ser idénticos. Cuando se quiera
reproducibilidad byte a byte: `pip-compile --generate-hashes` sobre
`requirements.txt` → `requirements.lock`, y `pip install --require-hashes -r
requirements.lock` en el Dockerfile. No se hizo acá porque `requirements.txt`
no se toca desde este directorio y un lock que no acompaña a su fuente
envejece en silencio.

### Imagen más liviana (opcional)

`requirements.txt` incluye `ragas`/`langchain`, que solo usa `evals/`. La
imagen no las necesita: un `requirements-runtime.txt` con las cinco primeras
líneas (`anthropic`, `psycopg`, `pgvector`, `sentence-transformers`,
`pyyaml`) más `langgraph` y `networkx` baja cientos de MB. No se hizo acá para
no tocar `requirements.txt` ni tener dos listas que diverjan.

### Modelo de embeddings pre-descargado (opcional, recomendado)

Con `RAGB_WARMUP=1` (ConfigMap) cada pod descarga y carga el modelo **al
arrancar**, en un hilo, y `/readyz` responde 503 hasta que esté: la réplica
nueva no recibe tráfico fría, y el modelo se carga una sola vez aunque lleguen
consultas simultáneas. Lo que sigue costando es la descarga en sí (a
`/tmp/hf`, que es `emptyDir`, así que cada réplica nueva la repite) y el
egreso 443 hacia Hugging Face que la NetworkPolicy tiene que permitir.
Hornearlo en la imagen evita las dos cosas:

```dockerfile
# después del COPY ragb/, antes del USER:
ENV HF_HOME=/opt/hf
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('intfloat/multilingual-e5-small')"
ENV HF_HUB_OFFLINE=1
```

y quitar `HF_HOME` del ConfigMap. Requiere red hacia huggingface.co en el
build. Con el modelo horneado, el 443 de la NetworkPolicy puede cerrarse a
la API de Anthropic sola (ver "Lo que este directorio NO resuelve").

## 2. Crear el Secret (fuera del repo)

```bash
kubectl apply -f deploy/k8s/namespace.yaml
kubectl -n rag-blindado create secret generic rag-blindado-secrets \
  --from-literal=DATABASE_URL='postgresql://usuario:password@host:5432/ragb' \
  --from-literal=ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  --from-literal=RAGB_API_TOKEN="$(openssl rand -hex 32)"
```

`k8s/secret.example.yaml` documenta las claves con valores vacíos y **no**
está en la kustomization: no se aplica. En un clúster real, mejor un gestor
de secretos (External Secrets, Sealed Secrets, Vault) que materialice un
Secret con ese mismo nombre y esas claves.

Sin el Secret, el pod no arranca (`envFrom` con `secretRef` obligatorio). Con
el Secret pero sin alguna de las tres claves, arranca, `/healthz` da 200 y
`/readyz` da 503 nombrando lo que falta: el pod queda fuera del Service sin
morir, y `kubectl describe` muestra el motivo. `RAGB_API_TOKEN` cuenta como
obligatoria a propósito: sin él `/query` rechaza el 100 % de las consultas, y
un pod así dentro del Service solo reparte errores.

## 3. Aplicar

```bash
kubectl apply -k deploy/k8s/
kubectl -n rag-blindado rollout status deployment/rag-blindado
kubectl -n rag-blindado get pods,svc,hpa,pdb,networkpolicy
```

`rollout status` espera a que cada réplica nueva pase readiness, y readiness
espera al modelo: en un clúster sin el modelo horneado, el primer rollout
tarda lo que tarde la descarga.

Antes de aplicar, tres parámetros del clúster que los manifiestos asumen:

- **Namespace del ingress**: `networkpolicy.yaml` solo admite tráfico desde
  `ingress-nginx`. Con Gateway API, Traefik u otro, cambiar ese nombre.
- **CIDR de Postgres**: el egreso 5432 va a `10.0.0.0/8` (base administrada
  dentro de la VPC). Reemplazar por el `/32` de la base, o por un
  `namespaceSelector` si corre en el clúster. Nunca `0.0.0.0/0`: el test lo
  rechaza.
- **CNI con NetworkPolicy**: sin Calico/Cilium o equivalente, la política se
  crea y no hace nada. Verificar con un pod de prueba (`kubectl run … --rm -it
  --image=curlimages/curl -- curl rag-blindado.rag-blindado.svc`), que debería
  fallar desde un namespace que no sea el del ingress.

El Deployment exige el perfil `restricted` de Pod Security (lo declara el
Namespace): no root, sin escalada, sin capabilities, seccomp, sistema de
archivos de solo lectura con `/tmp` en `emptyDir`.

### Apagado

Ante SIGTERM el servidor deja de aceptar conexiones y **espera a los requests
en vuelo** hasta `RAGB_SHUTDOWN_TIMEOUT` (20 s); pasado el tope los corta y lo
registra. El `preStop` de 5 s cubre la propagación del Endpoint (el SIGTERM
llega antes de que el ingress deje de enrutar al pod). 5 + 20 < 30 =
`terminationGracePeriodSeconds`; `tests/test_k8s_manifests.py` afirma la suma
y `tests/test_api.py` prueba la espera con un request lento.

## 4. Probar el servicio

Desde dentro del clúster (o con `kubectl port-forward svc/rag-blindado 8080:80`):

```bash
curl -s localhost:8080/readyz
curl -s -X POST localhost:8080/query \
  -H "Authorization: Bearer $RAGB_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cuál es el plazo de severidad 1?", "tenant": "acme", "top_k": 5}'
```

| situación | código |
|---|---|
| `/readyz` sin `DATABASE_URL`, `ANTHROPIC_API_KEY` o `RAGB_API_TOKEN` | 503 nombrando la variable |
| `/readyz` con el modelo de embeddings todavía cargando (`RAGB_WARMUP=1`) | 503 `modelo de embeddings cargando` |
| `/query` sin `RAGB_API_TOKEN` en el pod | 503 |
| token ausente o incorrecto | 401 |
| cuerpo que no es JSON, falta `question`/`tenant`, `tenant` que no es identificador (`[A-Za-z0-9._-]{1,64}`), `top_k` fuera de 1–50, claves extra, `Content-Length` que no es solo dígitos | 400 |
| cuerpo mayor a 16 KiB (por `Content-Length`, antes de leerlo) | 413 |
| `PermissionError` de las guardas (LLM06) | 403 |
| cualquier otro fallo del pipeline | 500 sin detalle (el detalle va al log del pod, sin el body) |
| más de `RAGB_MAX_CONNECTIONS` conexiones simultáneas en el pod | 503 `servidor saturado`, sin abrir hilo |

El log del servicio registra método, ruta **sin query string** y código, con
los caracteres de control escapados. Nunca el cuerpo ni el token;
`TRACE_INCLUDE_TEXT=0` en el ConfigMap mantiene lo mismo para las trazas de
`ragb/observe.py`.

## 5. Verificar sin clúster

Lo que corre en CI (job `guards`, sin credenciales ni red):

```bash
python -m pytest tests/test_api.py tests/test_k8s_manifests.py -q
```

`test_k8s_manifests.py` es un test de cableado: afirma que ningún contenedor
usa `latest`, que todos tienen `requests`/`limits`, `runAsNonRoot`,
`readOnlyRootFilesystem`, sondas, `envFrom` al ConfigMap y al Secret, que el
HPA no baja de 2 réplicas, que toda regla de egreso de la NetworkPolicy tiene
destino, que `preStop + RAGB_SHUTDOWN_TIMEOUT` cabe en el grace period y que
`OMP_NUM_THREADS` coincide con el límite de CPU. Un refactor que borre
cualquiera de esas líneas rompe el test, no producción.

Validación de esquema contra la API de Kubernetes, con
[kubeconform](https://github.com/yannh/kubeconform):

```bash
curl -sSL https://github.com/yannh/kubeconform/releases/latest/download/kubeconform-linux-amd64.tar.gz | tar xz
./kubeconform -strict -summary -ignore-filename-pattern kustomization.yaml deploy/k8s/
```

`-strict` rechaza campos desconocidos (un typo en `readOnlyRootFilesystem`
pasaría inadvertido sin eso). `secret.example.yaml` sí se valida: es un
Secret sintácticamente correcto, solo que vacío.

## Lo que este directorio NO resuelve

- **TLS, rate limit, límite de conexiones y WAF**: van en el ingress/gateway,
  delante del Service. El tope de conexiones del servidor
  (`RAGB_MAX_CONNECTIONS`) es un fusible, no una defensa: un cliente que gotea
  un byte cada 29 s mantiene un hilo vivo (el `timeout` del handler es por
  lectura, no por request). Con ingress-nginx, como mínimo:

  ```yaml
  metadata:
    annotations:
      nginx.ingress.kubernetes.io/limit-rps: "10"
      nginx.ingress.kubernetes.io/limit-connections: "20"
      nginx.ingress.kubernetes.io/proxy-body-size: "16k"
      nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
  ```

  y `client-header-timeout` / `client-body-timeout` bajos en el ConfigMap del
  controlador.
- **Egreso por dominio**: la NetworkPolicy abre el 443 a direcciones públicas
  (excluye lo privado y la metadata del nodo), no a `api.anthropic.com` en
  particular: una NetworkPolicy no filtra por nombre. Para cerrarlo de verdad
  hace falta un proxy de salida con lista de dominios o una política por FQDN
  del CNI (`CiliumNetworkPolicy` con `toFQDNs`), y hornear el modelo en la
  imagen para no depender de huggingface.co en runtime.
- **Un token por cliente**: `RAGB_API_TOKEN` es una credencial de servicio;
  quien la tiene elige el `tenant`. Si distintos clientes necesitan
  credenciales distintas, va un gateway con autenticación propia delante.
- **Ingesta**: la API solo consulta. Indexar el corpus sigue siendo
  `python -m ragb.cli --tenant … ingest … --write` (un Job de Kubernetes con
  la misma imagen y el mismo Secret sirve).
- **Persistencia de trazas**: `TRACE_LOG_PATH` va vacío; con el sistema de
  archivos de solo lectura, lo que corresponde es stdout hacia el colector de
  logs del clúster.
