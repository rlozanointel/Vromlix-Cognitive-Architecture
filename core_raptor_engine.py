#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "tenacity>=9.0.0",
#     "httpx>=0.28.1",
#     "instructor>=1.7.0",
#     "google-genai>=1.68.0",
#     "pydantic>=2.12.5",
#     "sqlite-vec>=0.1.3",
#     "fastembed>=0.5.1",
#     "scikit-learn>=1.5.0",
#     "scikit-learn-intelex",
#     "umap-learn>=0.5.11",
#     "numpy>=1.24.0",
#     "markitdown>=0.0.1a4",
#     "feedparser>=6.0.12",
#     "lxml>=5.1.0",
# ]
# ///
# -*- coding: utf-8 -*-
# @description Consolidación de Memoria RAPTOR (v1.0) - SOTA Wrapper.

import sys

# Fix: Asegurar que el motor encuentre vromlix_utils en el root de la partición/repositorio
VROMLIX_ROOT = "/media/rogerman/14befb81-4210-4134-a9a0-0ee76166e483/VROMLIX_CORE"
if VROMLIX_ROOT not in sys.path:
    sys.path.append(VROMLIX_ROOT)

import argparse  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402

from vromlix_utils import VromlixRaptorEngine  # noqa: E402

# --- SILENCIO SOTA ---
logging.getLogger("sklearnex").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# --- SOTA Intel Optimization ---
try:
    from sklearnex import patch_sklearn  # type: ignore

    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        patch_sklearn()
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
except ImportError:
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VROMLIX RAPTOR Consolidation Engine")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Borra la jerarquía actual y realiza una consolidación global.",
    )
    args = parser.parse_args()

    # Inicialización del Engine Centralizado
    print("🦉 Launching RAPTOR SOTA Consolidator...")
    engine = VromlixRaptorEngine()
    try:
        engine.run_consolidation(force_full=args.full)
    except KeyboardInterrupt:
        print("\n🛑 Proceso interrumpido por el usuario.")
    except Exception as e:
        print(f"\n❌ Error Crítico: {e}")
