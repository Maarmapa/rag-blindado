"""El dataset del juez SLM: determinista, sin torch, con conteos exactos.

Lo que se afirma es lo que un cambio descuidado rompería sin que nada avise:
que la cuarentena no entra al dataset (LLM01), que un caso de negativa no
produce positivos, que las paráfrasis heredan etiquetas y siguen a su original
en el split, y que dos corridas dan bytes idénticos.

Los conteos exactos dependen del corpus, del dataset dorado y de las
constantes de `slm.build_dataset`. Si cualquiera de los tres cambia, los
números de acá cambian: eso es una decisión, no un accidente, y se actualiza
a mano mirando el manifest nuevo.
"""

from __future__ import annotations

import json

import pytest

from ragb.guards import detect_injection
from slm import build_dataset as bd


@pytest.fixture(scope="module")
def rows():
    chunks = bd.load_chunks()
    cases = bd.load_cases()
    return bd.build_pairs(cases, chunks)


def test_conteos_exactos(rows):
    m = bd.summarize(rows)
    assert m["total"] == 312  # 8 casos x 3 preguntas x 13 fragmentos
    assert m["by_kind"] == {
        "positive": 36,
        "hard_negative": 27,
        "negative": 210,
        "refusal": 39,
    }
    assert m["by_split"]["train"] == {
        "rows": 234,
        "positives": 18,
        "negatives": 216,
        "cases": 6,
    }
    assert m["by_split"]["valid"] == {
        "rows": 78,
        "positives": 18,
        "negatives": 60,
        "cases": 2,
    }


def test_manifest_cuenta_la_cuarentena():
    """Un fragmento en cuarentena no entra al dataset, pero el manifest tiene
    que decir cuántos quedaron fuera: si un documento futuro se cuarentena
    entero, el dataset se achica y eso se ve, no se infiere."""
    chunks, quarantined = bd.load_chunks_with_quarantine()
    assert (len(chunks), quarantined) == (13, 1)
    manifest = bd.build(out_dir=None)
    assert manifest["quarantined"] == 1
    assert manifest["chunks"] == 13


def test_trece_fragmentos_saneados_y_ninguno_en_cuarentena():
    chunks = bd.load_chunks()
    assert len(chunks) == 13
    for c in chunks:
        assert detect_injection(c["text"]).flagged is False
        assert "IGNORA TODAS LAS INSTRUCCIONES" not in c["text"]
    # El dato legítimo que rodeaba a la inyección sí entra: la escisión por
    # párrafo conserva el plazo de pago y el horario (ver guards.excise_injection).
    nota = " ".join(c["text"] for c in chunks if c["source"] == "nota-proveedor.md")
    assert "30 días corridos" in nota and "9:00 a 18:00" in nota


def test_ningun_fragmento_del_dataset_esta_marcado(rows):
    for r in rows:
        assert detect_injection(r["chunk_text"]).flagged is False


def test_expects_refusal_solo_produce_negativos(rows):
    refusal_case_ids = {i for i, c in enumerate(bd.load_cases()) if c.get("expects_refusal")}
    assert refusal_case_ids == {6}
    subset = [r for r in rows if r["case_id"] in refusal_case_ids]
    assert len(subset) == 39
    assert all(r["label"] == 0 and r["kind"] == "refusal" for r in subset)


def test_cada_caso_no_refusal_tiene_al_menos_un_positivo(rows):
    por_caso: dict[int, int] = {}
    for r in rows:
        if not r["paraphrase"]:
            por_caso[r["case_id"]] = por_caso.get(r["case_id"], 0) + r["label"]
    assert por_caso == {0: 2, 1: 1, 2: 4, 3: 1, 4: 2, 5: 1, 6: 0, 7: 1}


def test_referencia_enumerada_marca_positivo_cada_item():
    """Caso 2: la referencia enumera cuatro condiciones que el corpus reparte
    en cuatro fragmentos. Con solo el promedio, tres quedaban como negativos
    difíciles y el juez aprendía a rechazar la respuesta. Ahora los cuatro
    son positivos, por cobertura de un ítem cada uno, y ningún fragmento de
    otro documento se cuela."""
    cases = bd.load_cases()
    chunks = bd.load_chunks()
    caso = cases[2]
    assert len(bd.reference_items(caso["reference"])) == 4
    positivos = [i for i, c in enumerate(chunks) if bd.is_positive(caso["question"], caso["reference"], c["text"])]
    assert positivos == [0, 1, 2, 3]
    assert all(chunks[i]["source"] == "estandar-agentes.md" for i in positivos)
    # Solo el promedio sigue dando un positivo: la regla 3b es lo que suma tres.
    por_promedio = [
        i
        for i, c in enumerate(chunks)
        if bd.relevance_score(caso["question"], caso["reference"], c["text"]) >= bd.POSITIVE_THRESHOLD
    ]
    assert por_promedio == [0]
    # Las referencias sin ";" no tienen ítems: la regla no las toca.
    assert bd.reference_items(cases[0]["reference"]) == []
    assert bd.item_score(cases[0]["reference"], chunks[10]["text"]) == 0.0


def test_negativo_dificil_es_del_mismo_documento_que_un_positivo(rows):
    for case_id in range(8):
        subset = [r for r in rows if r["case_id"] == case_id and not r["paraphrase"]]
        positive_sources = {r["source"] for r in subset if r["label"] == 1}
        for r in subset:
            if r["kind"] == "hard_negative":
                assert r["source"] in positive_sources
            if r["kind"] == "negative":
                assert r["source"] not in positive_sources


def test_parafrasis_heredan_etiqueta_y_split(rows):
    por_pregunta: dict[str, list[dict]] = {}
    for r in rows:
        por_pregunta.setdefault(r["question"], []).append(r)
    assert len(por_pregunta) == 24  # 8 casos x 3 variantes, todas distintas
    for case_id in range(8):
        variantes = [
            [r for r in rows if r["case_id"] == case_id and r["question"] == q]
            for q in {r["question"] for r in rows if r["case_id"] == case_id}
        ]
        firmas = {tuple((r["chunk_text"], r["label"], r["split"]) for r in v) for v in variantes}
        assert len(firmas) == 1  # misma etiqueta por fragmento, mismo split


def test_parafrasis_por_reglas():
    assert bd.paraphrases("¿Cuál es el plazo de pago?") == [
        "¿Cuál es el plazo de pago?",
        "Cuál es el plazo de pago",
        "Dime el plazo de pago",
    ]
    assert bd.paraphrases("¿En qué horario atiende?") == [
        "¿En qué horario atiende?",
        "En qué horario atiende",
        "Dime en qué horario atiende",
    ]
    # Una pregunta sin regla que calce solo pierde los signos.
    assert bd.paraphrases("¿Hay soporte?") == ["¿Hay soporte?", "Hay soporte"]


def test_split_es_estable_y_va_por_hashlib():
    assert bd.split_for("¿Qué condiciones debe cumplir un agente antes de habilitarse en producción?") == "valid"
    assert bd.split_for("¿Cuál es el plazo de respuesta para un incidente de severidad 1?") == "valid"
    assert bd.split_for("¿Cuál es el plazo de pago acordado con el proveedor?") == "train"
    # Espacios alrededor no cambian el destino; una letra sí puede.
    assert bd.split_for("  ¿Cuál es el plazo de pago acordado con el proveedor?  ") == "train"


def test_los_archivos_versionados_son_los_que_produce_el_codigo(tmp_path):
    """slm/data/*.jsonl y manifest.json están en el repo y train_grader los
    consume tal cual. Si corpus/, evals/dataset.yaml o las constantes cambian
    y nadie regenera, se entrena sobre datos viejos sin aviso."""
    fresco = tmp_path / "fresco"
    manifest = bd.build(out_dir=fresco)
    aviso = "slm/data está desactualizado: regenera con `python -m slm.build_dataset`"
    assert json.loads((bd.OUT_DIR / "manifest.json").read_text(encoding="utf-8")) == manifest, aviso
    for name in ("train.jsonl", "valid.jsonl", "manifest.json"):
        assert (bd.OUT_DIR / name).read_bytes() == (fresco / name).read_bytes(), aviso


def test_dos_corridas_producen_bytes_identicos(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    m1 = bd.build(out_dir=a)
    m2 = bd.build(out_dir=b)
    assert m1 == m2
    for name in ("train.jsonl", "valid.jsonl", "manifest.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    lineas = (a / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lineas) == 234
    fila = json.loads(lineas[0])
    assert set(fila) == {
        "case_id",
        "question",
        "paraphrase",
        "chunk_text",
        "source",
        "label",
        "kind",
        "split",
    }


def test_terminos_de_contenido_normalizan_numeros_y_plural():
    assert bd.content_terms("Veinticuatro meses") == {"24", "mes"}
    assert bd.content_terms("Una hora.") == {"1", "hora"}
    # "una" como artículo no es un número: antes "una acción" cubría el "1" de
    # "severidad 1" y todo fragmento con un artículo sumaba cobertura.
    assert bd.content_terms("una reunión cualquiera") == {"reunion", "cualquiera"}
    assert bd.content_terms("toma una acción no autorizada") == {"toma", "accion", "autorizada"}
    assert bd.content_terms("Las credenciales de acceso") == {"credencial", "acceso"}
    # "1 hora" del corpus y "Una hora" de la referencia cubren lo mismo.
    assert bd.coverage(bd.content_terms("Una hora"), bd.content_terms("plazo de 1 hora")) == 1.0
