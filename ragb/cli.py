"""CLI del pipeline.

    python -m ragb.cli --tenant acme ingest corpus/ --write
    python -m ragb.cli --tenant acme ask "¿Cuál es el plazo de severidad 1?"

`--tenant` y `--collection` van ANTES del subcomando: están declarados en el
parser de nivel superior, así que argparse los rechaza si aparecen después.
"""

from __future__ import annotations

import argparse
import json
import sys

from .guards import Permissions


def main() -> int:
    parser = argparse.ArgumentParser(prog="ragb")
    parser.add_argument("--tenant", default="demo")
    parser.add_argument("--collection", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="indexar archivos o un directorio")
    p_ingest.add_argument("path")
    p_ingest.add_argument(
        "--write",
        action="store_true",
        help="ejecuta la escritura; sin esta bandera corre en dry-run",
    )

    p_ask = sub.add_parser("ask", help="consultar el corpus indexado")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=None)
    p_ask.add_argument("--json", action="store_true")

    args = parser.parse_args()

    # Importado acá para que `--help` no cargue el stack de ML.
    from . import embeddings, pipeline, store

    if args.command == "ingest":
        perms = Permissions(
            tenant=args.tenant, can_write=True, dry_run=not args.write
        )
        store.init_schema(embeddings.dimension())
        report = pipeline.ingest_path(
            perms, args.path, collection=args.collection
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if report["dry_run"]:
            print("\n(dry-run: nada se escribió. Repite con --write.)", file=sys.stderr)
        return 0

    perms = Permissions(tenant=args.tenant, can_read=True)
    result = pipeline.query(
        perms, args.question, collection=args.collection, top_k=args.top_k
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    print(result["text"])
    if result["sources"]:
        print(f"\nFuentes: {', '.join(result['sources'])}")
    if result["quarantined"]:
        print("\nFragmentos en cuarentena:", file=sys.stderr)
        for q in result["quarantined"]:
            print(f"  · {q['source']} — {q['reason']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
