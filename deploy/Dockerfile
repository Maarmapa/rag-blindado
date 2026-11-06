# Imagen del servicio HTTP de rag-blindado (ragb/api.py).
#
#   docker build -f deploy/Dockerfile -t ghcr.io/maarmapa/rag-blindado:0.1.0 .
#
# El contexto es la RAÍZ del repo (el punto final del comando): el Dockerfile
# vive en deploy/ pero copia ragb/ y requirements.txt de arriba. El archivo
# deploy/Dockerfile.dockerignore (convención de BuildKit) recorta el contexto.
#
# Ningún secreto entra acá: ni ARG, ni ENV, ni archivos .env. Todo lo sensible
# llega en runtime por variables de entorno (en Kubernetes, desde un Secret).

# Tag Y digest: el tag dice qué se quiso (3.12-slim) y el digest fija qué se
# obtuvo. Sin el digest, `python:3.12-slim` cambia bajo los pies con cada
# rebuild de la imagen oficial y dos builds del mismo commit dan bases
# distintas. Es el digest del índice multi-arquitectura, obtenido del registro
# el 2026-09-21 (`GET /v2/library/python/manifests/3.12-slim`, cabecera
# Docker-Content-Digest); al actualizar la base se actualizan los dos juntos.
FROM python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9

# - PYTHONUNBUFFERED: los logs salen al instante, que es lo que `kubectl logs`
#   y cualquier colector necesitan.
# - PYTHONDONTWRITEBYTECODE: con el sistema de archivos de solo lectura no se
#   podrían escribir .pyc de todas formas; mejor no intentarlo.
# - HF_HOME: el modelo de embeddings se descarga al arrancar (RAGB_WARMUP) y se
#   cachea acá. /tmp es el único lugar escribible del contenedor en Kubernetes
#   (emptyDir montado en el Deployment).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/tmp/hf

# Usuario sin privilegios con uid fijo: el Deployment exige runAsNonRoot y
# declara el mismo uid (10001), así que el kubelet puede verificarlo aunque la
# imagen no lleve /etc/passwd resoluble.
RUN groupadd --gid 10001 ragb \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin ragb

WORKDIR /app

# torch primero y desde el índice de CPU. La rueda de torch en PyPI para Linux
# pesa ~530 MB y arrastra las librerías CUDA (nvidia-cudnn, nccl, cusparselt,
# nvshmem…): varios GB que un pod sin GPU nunca usa y que amplían la
# superficie de CVEs de una imagen expuesta. Instalado antes, pip ya lo ve
# satisfecho cuando resuelve requirements.txt y no vuelve a bajarlo de PyPI.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# Después las dependencias, y recién al final el código: la capa cara solo se
# reconstruye cuando cambia requirements.txt.
#
# ⚠ requirements.txt trae también ragas/langchain (solo para `evals/`), que
# la imagen no necesita. Se instala igual porque el orquestador integra las
# dependencias; una imagen más chica sale de un requirements-runtime.txt
# separado (ver deploy/README.md, "Imagen más liviana"). Pendiente conocido:
# sin pines exactos ni `--require-hashes` el build no es reproducible byte a
# byte; el README dice cómo generarlos con pip-compile cuando se decida.
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY ragb/ ./ragb/

# Que el código pertenezca a root y el proceso corra como ragb es intencional:
# el proceso no puede modificar lo que ejecuta, aunque el sistema de archivos
# fuera escribible.
USER 10001:10001

EXPOSE 8080

# --host 0.0.0.0 es explícito: el default del módulo es 127.0.0.1 (cerrado),
# y escuchar en todas las interfaces es una decisión de quien despliega.
CMD ["python", "-m", "ragb.api", "--host", "0.0.0.0", "--port", "8080"]
