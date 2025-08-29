#!/usr/bin/env python3
"""Generate OpenAPI specification for the API"""

import json
import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.backend.main import app


def generate_openapi_spec():
    """Generate and save OpenAPI specification"""
    openapi_spec = app.openapi()
    
    # Ensure docs directory exists
    docs_dir = Path(__file__).parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    # Save OpenAPI spec
    with open(docs_dir / "openapi.json", "w") as f:
        json.dump(openapi_spec, f, indent=2)
    
    print(f"OpenAPI specification saved to {docs_dir / 'openapi.json'}")
    print(f"Endpoints documented: {len(openapi_spec['paths'])}")
    
    # Print summary
    tags = set()
    for path_data in openapi_spec["paths"].values():
        for method_data in path_data.values():
            if isinstance(method_data, dict) and "tags" in method_data:
                tags.update(method_data["tags"])
    
    print(f"Tags: {sorted(tags)}")


if __name__ == "__main__":
    generate_openapi_spec()
