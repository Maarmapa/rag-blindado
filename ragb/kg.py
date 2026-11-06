"""Grafo de conocimiento liviano sobre los fragmentos del corpus.

Qué resuelve: la búsqueda vectorial encuentra fragmentos *parecidos* a la
pregunta, no fragmentos *relacionados* con ella. Si el corpus dice en un
documento que "Severidad 1" se atiende por "Mesa Nivel 2" y en otro que "Mesa
Nivel 2" escala a "Guardia", una pregunta por Severidad 1 nunca trae el
segundo documento por similitud. Un grafo de entidades y co-ocurrencias sí lo
alcanza en un salto.

Es deliberadamente simple: entidades por heurística (nombres propios y una
lista de términos del dominio), aristas por co-ocurrencia en el mismo
fragmento, pesos por frecuencia, y cada arista recuerda de qué fuentes salió
para poder citar. No usa un modelo: es determinista, corre sin red y se prueba
con asserts exactos. Cuando haga falta extracción con LLM, se enchufa como
`extractor` sin cambiar el grafo.

Reglas de seguridad que hereda del resto del repo:
- Solo se construye con fragmentos que ya pasaron `sanitize_chunks`; un
  fragmento en cuarentena no aporta nodos ni aristas (LLM01).
- Los términos se normalizan para comparar, pero se guardan como aparecen
  para mostrarlos.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable

import networkx as nx

_STOP = {
    "el",
    "la",
    "los",
    "las",
    "un",
    "una",
    "unos",
    "unas",
    "de",
    "del",
    "en",
    "y",
    "o",
    "que",
    "si",
    "no",
    "para",
    "por",
    "con",
    "sin",
    "al",
    "a",
    "es",
    "son",
    "se",
    "su",
    "sus",
    "lo",
    "le",
    "les",
    "este",
    "esta",
    "esto",
    "ese",
    "esa",
    "eso",
    "como",
    "cuando",
    "donde",
    "pero",
    "más",
    "menos",
    "también",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
}

_CONECT = {"de", "del", "la", "el", "y"}
_TOKEN = re.compile(r"[\wáéíóúñÁÉÍÓÚÑ]+|[^\w\s]")


def _es_cap(tok: str) -> bool:
    return tok[0].isupper() and tok[0].isalpha()


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", s).strip().lower()


def _nombres_propios(text: str) -> list[str]:
    """Secuencias de palabras capitalizadas (con dígitos y conectores en el
    medio): 'Mesa Nivel 2', 'Guardia de Infraestructura'. Un recorrido por
    tokens, no una regex, para que un espacio simple entre dos mayúsculas no
    corte la entidad."""
    toks = _TOKEN.findall(text)
    out: list[str] = []
    i = 0
    while i < len(toks):
        if not _es_cap(toks[i]):
            i += 1
            continue
        j = i + 1
        while j < len(toks):
            t = toks[j]
            if _es_cap(t) or t.isdigit():
                j += 1
            elif t.lower() in _CONECT and j + 1 < len(toks) and _es_cap(toks[j + 1]):
                j += 2
            else:
                break
        words = toks[i:j]
        # un artículo al inicio de oración no es parte del nombre ("La Mesa Nivel 2")
        while words and words[0].lower() in _STOP:
            words.pop(0)
        if words and not (len(words) == 1 and words[0].lower() in _STOP):
            out.append(" ".join(words))
        i = j
    return out


def extract_entities(text: str, *, terms: Iterable[str] = ()) -> dict[str, str]:
    """Devuelve {clave_normalizada: forma_visible}. Combina nombres propios
    (secuencias de palabras capitalizadas) con una lista de términos del
    dominio, que pueden ir en minúscula ("severidad 1")."""
    found: dict[str, str] = {}
    for raw in _nombres_propios(text):
        key = _norm(raw)
        if key in _STOP or len(key) < 3:
            continue
        found.setdefault(key, raw)
    low = _norm(text)
    for t in terms:
        k = _norm(t)
        if k and re.search(rf"(?<!\w){re.escape(k)}(?!\w)", low):
            found.setdefault(k, t)
    return found


Extractor = Callable[[str], dict[str, str]]


def build_kg(
    chunks: list[dict], *, terms: Iterable[str] = (), extractor: Extractor | None = None
) -> nx.Graph:
    """Un nodo por entidad, una arista por co-ocurrencia en un fragmento.
    `chunks` deben venir ya saneados (ver `guards.sanitize_chunks`)."""
    terms = tuple(terms)
    ext = extractor or (lambda text: extract_entities(text, terms=terms))
    g = nx.Graph()
    for c in chunks:
        if c.get("quarantine_reason"):
            continue  # nunca aprender de un fragmento en cuarentena
        ents = ext(c.get("text", ""))
        src = c.get("source", "?")
        for key, label in ents.items():
            if not g.has_node(key):
                g.add_node(key, label=label, count=0, sources=set())
            g.nodes[key]["count"] += 1
            g.nodes[key]["sources"].add(src)
        keys = sorted(ents)
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                if g.has_edge(a, b):
                    g[a][b]["weight"] += 1
                    g[a][b]["sources"].add(src)
                else:
                    g.add_edge(a, b, weight=1, sources={src})
    return g


def neighbors(g: nx.Graph, term: str, *, k: int = 5) -> list[tuple[str, int]]:
    """Vecinos de un término por peso de co-ocurrencia, de mayor a menor."""
    key = _norm(term)
    if key not in g:
        return []
    ranked = sorted(
        ((n, g[key][n]["weight"]) for n in g.neighbors(key)),
        key=lambda x: (-x[1], x[0]),
    )
    return [(g.nodes[n]["label"], w) for n, w in ranked[:k]]


def expand_question(
    g: nx.Graph, question: str, *, terms: Iterable[str] = (), k: int = 5
) -> list[str]:
    """Términos relacionados con las entidades de la pregunta, para ensanchar
    la recuperación. No incluye las entidades de la pregunta misma."""
    ents = extract_entities(question, terms=terms)
    scores: Counter[str] = Counter()
    labels: dict[str, str] = {}
    for key in ents:
        if key not in g:
            continue
        for n in g.neighbors(key):
            if n in ents:
                continue
            scores[n] += g[key][n]["weight"]
            labels[n] = g.nodes[n]["label"]
    return [
        labels[n] for n, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
    ]


def subgraph_context(
    g: nx.Graph, entities: Iterable[str], *, max_edges: int = 10
) -> str:
    """Relaciones entre las entidades dadas y sus vecinos, en texto plano con
    fuentes, para adjuntar al contexto del modelo con citación."""
    keys = {_norm(e) for e in entities if _norm(e) in g}
    edges = []
    for a in keys:
        for b in g.neighbors(a):
            w = g[a][b]["weight"]
            srcs = ", ".join(sorted(g[a][b]["sources"]))
            edges.append((w, g.nodes[a]["label"], g.nodes[b]["label"], srcs))
    edges.sort(key=lambda e: (-e[0], e[1], e[2]))
    seen = set()
    lines = []
    for w, a, b, srcs in edges:
        pair = tuple(sorted((a, b)))
        if pair in seen:
            continue
        seen.add(pair)
        lines.append(f"{a} — {b} ({w} fragmento{'s' if w != 1 else ''}: {srcs})")
        if len(lines) >= max_edges:
            break
    return "\n".join(lines)


def stats(g: nx.Graph) -> dict:
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "sources": len({s for _, d in g.nodes(data=True) for s in d["sources"]}),
        "top": [
            g.nodes[n]["label"]
            for n, _ in sorted(g.degree, key=lambda x: (-x[1], x[0]))[:5]
        ],
    }
