#!/usr/bin/env python3
"""
Generate an OpenAPI 3.1 spec from the FastAPI app.

Usage:
  python scripts/generate_openapi.py \
    --app-dir src/legal-document-simplifier \
    --app-path src.backend.main:app \
    --out docs/openapi.json
"""
import argparse, json, sys, yaml
from uvicorn.importer import import_from_string

parser = argparse.ArgumentParser()
parser.add_argument("--app-dir", required=True)
parser.add_argument("--app-path", default="src.backend.main:app")
parser.add_argument("--out", default="docs/openapi.json")
args = parser.parse_args()

sys.path.insert(0, args.app_dir)
app = import_from_string(args.app_path)
spec = app.openapi()

with open(args.out, "w") as f:
    if args.out.endswith(".yaml"):
        yaml.dump(spec, f, sort_keys=False)
    else:
        json.dump(spec, f, indent=2)

print(f"✅  OpenAPI written to {args.out}")
