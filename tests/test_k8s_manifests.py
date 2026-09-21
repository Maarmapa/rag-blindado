"""Test de cableado de deploy/: las políticas de seguridad siguen escritas.

Un manifiesto no se compila: si un refactor borra `readOnlyRootFilesystem` o
cambia el tag por `latest`, `kubectl apply` acepta el resultado sin decir
nada y la política desaparece en silencio. Este archivo afirma, línea por
línea, lo que un despliegue de este servicio tiene que decir. Lo que se
prueba no es que Kubernetes funcione: es que nadie haya deshecho una decisión
sin que un test lo cuente.

Corre sin clúster, sin docker y sin red: solo pyyaml, que el job `guards` de
CI ya instala.
"""

from __future__ import annotations

import pathlib
import re

import pytest
import yaml

RAIZ = pathlib.Path(__file__).resolve().parent.parent
K8S = RAIZ / "deploy" / "k8s"
DOCKERFILE = RAIZ / "deploy" / "Dockerfile"

NAMESPACE = "rag-blindado"
CONFIGMAP = "rag-blindado-config"
SECRET = "rag-blindado-secrets"
PUERTO = 8080


def _cargar(nombre: str) -> list[dict]:
    with (K8S / nombre).open(encoding="utf-8") as fh:
        return [doc for doc in yaml.safe_load_all(fh) if doc]


@pytest.fixture(scope="module")
def objetos() -> list[dict]:
    """Todos los objetos de la kustomization, en el orden en que se aplican."""
    (kust,) = _cargar("kustomization.yaml")
    return [obj for recurso in kust["resources"] for obj in _cargar(recurso)]


def _uno(objetos: list[dict], kind: str) -> dict:
    encontrados = [o for o in objetos if o["kind"] == kind]
    assert len(encontrados) == 1, f"se esperaba exactamente un {kind}, hay {len(encontrados)}"
    return encontrados[0]


def _contenedores(deployment: dict) -> list[dict]:
    spec = deployment["spec"]["template"]["spec"]
    return spec.get("initContainers", []) + spec["containers"]


# --- Kustomization: qué entra y qué no --------------------------------------


def test_kustomization_lista_todo_menos_el_secret_de_ejemplo():
    """El ejemplo de Secret NO se aplica: si entrara, el pod arrancaría con
    valores vacíos creyendo que está configurado."""
    (kust,) = _cargar("kustomization.yaml")
    en_disco = {p.name for p in K8S.glob("*.yaml")} - {"kustomization.yaml", "secret.example.yaml"}
    assert set(kust["resources"]) == en_disco
    assert "secret.example.yaml" not in kust["resources"]
    assert kust["namespace"] == NAMESPACE


def test_todos_los_objetos_declaran_el_namespace(objetos):
    for obj in objetos:
        if obj["kind"] == "Namespace":
            assert obj["metadata"]["name"] == NAMESPACE
        else:
            assert obj["metadata"]["namespace"] == NAMESPACE, obj["kind"]


def test_un_objeto_de_cada_clase(objetos):
    kinds = sorted(o["kind"] for o in objetos)
    assert kinds == sorted(
        [
            "Namespace",
            "ConfigMap",
            "Deployment",
            "Service",
            "HorizontalPodAutoscaler",
            "PodDisruptionBudget",
            "NetworkPolicy",
        ]
    )


# --- Imagen -----------------------------------------------------------------


def test_ninguna_imagen_usa_latest_ni_va_sin_tag(objetos):
    """`latest` no dice qué corre y no permite volver atrás. Un tag explícito
    (o un digest) es la única forma de que `image:` signifique algo."""
    for c in _contenedores(_uno(objetos, "Deployment")):
        imagen = c["image"]
        assert "@sha256:" in imagen or re.search(r":[^/:]+$", imagen), f"sin tag: {imagen}"
        assert not imagen.endswith(":latest"), imagen
        # Un tag es mutable: sin digest, solo `Always` garantiza que todas las
        # réplicas de un rollout vean la misma imagen. Con digest da lo mismo.
        assert "@sha256:" in imagen or c.get("imagePullPolicy") == "Always", (
            f"{imagen}: tag mutable con imagePullPolicy={c.get('imagePullPolicy')}"
        )


def test_el_tag_de_la_imagen_coincide_con_el_readme_de_deploy():
    """El README dice qué construir y el Deployment qué correr; si divergen,
    alguien despliega una versión y documenta otra."""
    (deployment,) = _cargar("deployment.yaml")
    imagen = _contenedores(deployment)[0]["image"]
    readme = (RAIZ / "deploy" / "README.md").read_text(encoding="utf-8")
    assert imagen in readme


# --- Recursos y seguridad del pod -------------------------------------------


def test_todos_los_contenedores_tienen_requests_y_limits(objetos):
    for c in _contenedores(_uno(objetos, "Deployment")):
        for clase in ("requests", "limits"):
            for recurso in ("cpu", "memory"):
                assert c["resources"][clase][recurso], f"{c['name']}: falta {clase}.{recurso}"


def test_security_context_del_pod_y_de_cada_contenedor(objetos):
    deployment = _uno(objetos, "Deployment")
    pod = deployment["spec"]["template"]["spec"]

    assert pod["securityContext"]["runAsNonRoot"] is True
    assert pod["securityContext"]["runAsUser"] >= 10000
    assert pod["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault"
    assert pod["automountServiceAccountToken"] is False

    for c in _contenedores(deployment):
        sc = c["securityContext"]
        assert sc["runAsNonRoot"] is True, c["name"]
        assert sc["readOnlyRootFilesystem"] is True, c["name"]
        assert sc["allowPrivilegeEscalation"] is False, c["name"]
        assert sc["capabilities"]["drop"] == ["ALL"], c["name"]
        assert "add" not in sc["capabilities"], c["name"]
        assert not sc.get("privileged"), c["name"]


def test_readonly_root_tiene_un_tmp_escribible_con_tope(objetos):
    """Solo lectura sin /tmp escribible revienta en la primera consulta, cuando
    sentence-transformers intenta cachear el modelo. Y un emptyDir sin tope
    puede llenar el disco del nodo."""
    deployment = _uno(objetos, "Deployment")
    pod = deployment["spec"]["template"]["spec"]
    (tmp,) = [v for v in pod["volumes"] if v["name"] == "tmp"]
    assert tmp["emptyDir"]["sizeLimit"]
    for c in _contenedores(deployment):
        assert any(m["mountPath"] == "/tmp" and m["name"] == "tmp" for m in c["volumeMounts"]), c["name"]


def test_el_uid_del_deployment_es_el_del_dockerfile():
    (deployment,) = _cargar("deployment.yaml")
    uid = deployment["spec"]["template"]["spec"]["securityContext"]["runAsUser"]
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert re.search(rf"^USER {uid}:{uid}\s*$", dockerfile, re.M), "el Dockerfile no cambia a ese uid"


def test_namespace_exige_pod_security_restricted(objetos):
    """La política la aplica el API server, no solo el manifiesto: un pod
    futuro que olvide el securityContext no entra."""
    labels = _uno(objetos, "Namespace")["metadata"]["labels"]
    assert labels["pod-security.kubernetes.io/enforce"] == "restricted"


# --- Sondas y ciclo de vida -------------------------------------------------


def test_sondas_presentes_y_apuntando_a_las_rutas_correctas(objetos):
    """Liveness a /healthz (¿vive?) y readiness a /readyz (¿puede atender?).
    Cruzarlas convierte un Secret vacío en un crash loop."""
    (c,) = _contenedores(_uno(objetos, "Deployment"))
    assert c["livenessProbe"]["httpGet"]["path"] == "/healthz"
    assert c["readinessProbe"]["httpGet"]["path"] == "/readyz"
    assert c["startupProbe"]["httpGet"]["path"] == "/healthz"
    for sonda in ("livenessProbe", "readinessProbe", "startupProbe"):
        assert c[sonda]["httpGet"]["port"] == "http"
    (puerto,) = c["ports"]
    assert (puerto["name"], puerto["containerPort"]) == ("http", PUERTO)


def test_termination_grace_y_rolling_update_sin_bajar_de_las_replicas(objetos):
    deployment = _uno(objetos, "Deployment")
    assert deployment["spec"]["template"]["spec"]["terminationGracePeriodSeconds"] >= 10
    assert deployment["spec"]["replicas"] >= 2
    assert deployment["spec"]["strategy"]["rollingUpdate"]["maxUnavailable"] == 0


def test_el_apagado_cabe_en_el_grace_period(objetos):
    """preStop (propagación del Endpoint) + RAGB_SHUTDOWN_TIMEOUT (espera a los
    requests en vuelo) tienen que caber en terminationGracePeriodSeconds; si
    no, el SIGKILL llega antes de que el servidor termine de esperar y la
    espera que ragb/api.py implementa no sirve de nada."""
    deployment = _uno(objetos, "Deployment")
    pod = deployment["spec"]["template"]["spec"]
    (c,) = _contenedores(deployment)
    comando = c["lifecycle"]["preStop"]["exec"]["command"]
    assert comando[0] == "sleep"
    pre_stop = int(comando[1])
    datos = _uno(objetos, "ConfigMap")["data"]
    tope = int(datos["RAGB_SHUTDOWN_TIMEOUT"])
    assert pre_stop + tope < pod["terminationGracePeriodSeconds"], (pre_stop, tope)
    assert datos["RAGB_WARMUP"] == "1", "sin calentamiento, /readyz miente en cada réplica nueva"
    assert int(datos["RAGB_MAX_CONNECTIONS"]) > 0


def test_los_hilos_de_torch_van_atados_al_limite_de_cpu(objetos):
    """torch dimensiona su pool por los cores del nodo, no por el cgroup: sin
    esto, un límite de 2 CPU en un nodo grande es throttling permanente."""
    (c,) = _contenedores(_uno(objetos, "Deployment"))
    limite = str(c["resources"]["limits"]["cpu"])
    assert limite.isdigit(), "el límite de CPU tiene que ser entero para atarlo a OMP_NUM_THREADS"
    datos = _uno(objetos, "ConfigMap")["data"]
    assert datos["OMP_NUM_THREADS"] == limite
    assert datos["MKL_NUM_THREADS"] == limite


# --- Configuración y secretos -----------------------------------------------


def test_env_viene_del_configmap_y_del_secret_no_de_valores_en_linea(objetos):
    """`envFrom` a los dos objetos, y NINGÚN `env:` con valor en línea: un
    secreto pegado en el Deployment es un secreto en el repo público."""
    (c,) = _contenedores(_uno(objetos, "Deployment"))
    refs = {("configMapRef", e["configMapRef"]["name"]) for e in c["envFrom"] if "configMapRef" in e} | {
        ("secretRef", e["secretRef"]["name"]) for e in c["envFrom"] if "secretRef" in e
    }
    assert ("configMapRef", CONFIGMAP) in refs
    assert ("secretRef", SECRET) in refs
    assert "env" not in c

    assert _uno(objetos, "ConfigMap")["metadata"]["name"] == CONFIGMAP


def test_el_configmap_no_lleva_nada_secreto(objetos):
    datos = _uno(objetos, "ConfigMap")["data"]
    for clave in ("DATABASE_URL", "ANTHROPIC_API_KEY", "RAGB_API_TOKEN"):
        assert clave not in datos, f"{clave} es secreto y va en el Secret"
    assert datos["TRACE_INCLUDE_TEXT"] == "0"
    assert datos["HF_HOME"].startswith("/tmp/")


def test_secret_de_ejemplo_tiene_las_claves_y_solo_placeholders_vacios():
    (secreto,) = _cargar("secret.example.yaml")
    assert secreto["kind"] == "Secret"
    assert secreto["metadata"]["name"] == SECRET
    assert set(secreto["stringData"]) == {"DATABASE_URL", "ANTHROPIC_API_KEY", "RAGB_API_TOKEN"}
    assert all(v == "" for v in secreto["stringData"].values()), "el ejemplo no puede traer valores"
    assert "data" not in secreto


# --- Service, HPA, PDB, NetworkPolicy ---------------------------------------


def test_service_clusterip_apunta_al_puerto_nombrado_del_deployment(objetos):
    svc = _uno(objetos, "Service")
    deployment = _uno(objetos, "Deployment")
    assert svc["spec"]["type"] == "ClusterIP"
    assert svc["spec"]["selector"] == deployment["spec"]["selector"]["matchLabels"]
    (puerto,) = svc["spec"]["ports"]
    assert puerto["targetPort"] == "http"


def test_hpa_v2_con_piso_2_y_techo_finito(objetos):
    hpa = _uno(objetos, "HorizontalPodAutoscaler")
    deployment = _uno(objetos, "Deployment")
    assert hpa["apiVersion"] == "autoscaling/v2"
    assert hpa["spec"]["scaleTargetRef"] == {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "name": deployment["metadata"]["name"],
    }
    assert hpa["spec"]["minReplicas"] >= 2
    assert hpa["spec"]["minReplicas"] == deployment["spec"]["replicas"]
    assert hpa["spec"]["maxReplicas"] == 6
    (metrica,) = hpa["spec"]["metrics"]
    assert metrica["resource"]["name"] == "cpu"
    assert metrica["resource"]["target"]["averageUtilization"] == 70


def test_pdb_deja_al_menos_una_replica_y_selecciona_los_pods_del_deployment(objetos):
    pdb = _uno(objetos, "PodDisruptionBudget")
    deployment = _uno(objetos, "Deployment")
    assert pdb["spec"]["minAvailable"] == 1
    assert pdb["spec"]["selector"]["matchLabels"] == deployment["spec"]["selector"]["matchLabels"]


def test_networkpolicy_existe_y_cierra_ingreso_y_egreso(objetos):
    """Existe, selecciona a los pods de la API, y declara los DOS sentidos:
    sin `Egress` en policyTypes, la salida quedaría abierta por omisión."""
    np = _uno(objetos, "NetworkPolicy")
    deployment = _uno(objetos, "Deployment")
    assert np["spec"]["podSelector"]["matchLabels"] == deployment["spec"]["selector"]["matchLabels"]
    assert sorted(np["spec"]["policyTypes"]) == ["Egress", "Ingress"]

    (ingreso,) = np["spec"]["ingress"]
    assert ingreso["ports"] == [{"protocol": "TCP", "port": PUERTO}]
    assert all("namespaceSelector" in origen for origen in ingreso["from"]), "el ingreso se limita por namespace"

    puertos_egreso = {(p["protocol"], p["port"]) for regla in np["spec"]["egress"] for p in regla["ports"]}
    assert puertos_egreso == {("UDP", 53), ("TCP", 53), ("TCP", 5432), ("TCP", 443)}
    # Cada regla de egreso nombra puertos Y destinos: sin `ports` sería "todo
    # puerto"; sin `to`, "ese puerto hacia cualquier destino, internet incluido"
    # (así estaba el 443, y el encabezado prometía lo contrario).
    for regla in np["spec"]["egress"]:
        assert regla.get("ports"), regla
        assert regla.get("to"), f"regla de egreso sin destino: {regla}"


def test_el_egreso_https_excluye_lo_privado_y_la_metadata_del_nodo(objetos):
    """El 443 se abre a direcciones públicas (Anthropic y Hugging Face viven
    detrás de CDNs) pero nunca al API server, a servicios internos ni a la
    metadata de la nube (169.254.169.254), que reparte credenciales."""
    np = _uno(objetos, "NetworkPolicy")
    (https,) = [r for r in np["spec"]["egress"] if r["ports"] == [{"protocol": "TCP", "port": 443}]]
    (destino,) = https["to"]
    bloque = destino["ipBlock"]
    assert bloque["cidr"] == "0.0.0.0/0"
    assert {"10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16"} <= set(bloque["except"])

    (postgres,) = [r for r in np["spec"]["egress"] if r["ports"] == [{"protocol": "TCP", "port": 5432}]]
    for destino in postgres["to"]:
        # Un destino privado (ipBlock RFC 1918 o selector del clúster), nunca
        # 0.0.0.0/0: un Postgres público en el `to` es un canal de salida.
        if "ipBlock" in destino:
            assert destino["ipBlock"]["cidr"].startswith(("10.", "172.", "192.168.")), destino
        else:
            assert "namespaceSelector" in destino or "podSelector" in destino, destino


# --- Dockerfile -------------------------------------------------------------


def test_dockerfile_sin_root_sin_latest_y_sin_secretos():
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    lineas = [ln.strip() for ln in dockerfile.splitlines() if ln.strip() and not ln.startswith("#")]

    froms = [ln for ln in lineas if ln.startswith("FROM ")]
    assert froms and all(":" in ln and not ln.endswith(":latest") for ln in froms)
    # Tag Y digest: el tag dice qué se quiso, el digest fija qué se obtuvo.
    assert all(re.search(r"@sha256:[0-9a-f]{64}$", ln) for ln in froms), froms

    # torch se instala primero y desde el índice de CPU: la rueda de PyPI
    # arrastra varios GB de librerías CUDA que un pod sin GPU no usa.
    pips = [i for i, ln in enumerate(lineas) if ln.startswith("RUN pip install")]
    assert len(pips) == 2, lineas
    assert "--index-url https://download.pytorch.org/whl/cpu torch" in lineas[pips[0]]
    assert lineas[pips[1]].endswith("-r requirements.txt")

    users = [ln for ln in lineas if ln.startswith("USER ")]
    assert users and users[-1] not in ("USER root", "USER 0", "USER 0:0")

    assert lineas[-1].startswith('CMD ["python", "-m", "ragb.api"')
    assert "--host" in lineas[-1], "el bind a 0.0.0.0 es explícito en el CMD, no un default del módulo"

    # Ninguna forma de meter un secreto en la imagen.
    for ln in lineas:
        assert not re.match(r"(ARG|ENV)\s+\w*(TOKEN|KEY|PASSWORD|SECRET|DATABASE_URL)", ln, re.I), ln
        assert not re.search(r"sk-ant-|postgres(ql)?://\S+:\S+@", ln), ln
    assert not re.search(r"^COPY\s+\.env\b", dockerfile, re.M)
