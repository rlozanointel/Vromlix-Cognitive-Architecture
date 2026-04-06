# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-genai>=1.68.0",
#     "instructor>=1.7.0",
#     "tenacity>=9.0.0",
#     "httpx>=0.28.1",
#     "numpy>=2.2.6",
#     "pydantic>=2.12.5",
#     "sqlite-vec>=0.1.9",
#     "duckduckgo-search>=8.1.1",
#     "feedparser>=6.0.12",
#     "lxml>=5.1.0",
#     "tqdm>=4.67.3",
#     "markitdown>=0.0.1a4",
#     "umap-learn>=0.5.11",
#     "scikit-learn>=1.5.0",
# ]
# ///
#!/usr/bin/env -S uv run
# -*- coding: utf-8 -*-
# @description SOTA auto-indexing engine that scans the Vromlix ecosystem, extracts metadata, and uses Gemini to generate file descriptions, updating the master JSON index and injecting missing metadata into notes.
"""
VROMLIX PHASE 2: Auto-Indexer & AI Librarian (v3.2 SOTA)
Now supports translation of Obsidian abstracts and automatic metadata injection in .md files.
"""

import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

SILENT_MODE = "--silent" in sys.argv


def dprint(msg):
    """Debug print function that respects silent mode."""
    if not SILENT_MODE:
        print(msg)


# Cross-injection to localize the Orchestrator
# sys.path.append(str(Path(__file__).parents[1]))

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

from vromlix_utils import vromlix  # noqa: E402

HAS_VROMLIX_UTILS = True

# --- CONFIGURATION ---
BASE_DIR = vromlix.paths.base
OLD_XML_PATH = vromlix.paths.codex_memory / "04_Knowledge_Index.xml"
NEW_JSON_PATH = vromlix.paths.codex_memory / "04_Knowledge_Index.json"


def extract_cognitive_core() -> dict:
    """Extract cognitive core philosophy from current Super-Index JSON (XML Independence)."""
    # 1. Try to rescue philosophy from current Super-Index JSON
    if NEW_JSON_PATH.exists():
        try:
            with open(NEW_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "cognitive_core" in data and data["cognitive_core"].get("protocols"):
                    return data["cognitive_core"]
        except Exception:
            pass

    # 2. If no JSON, try with old XML
    protocols: list[dict[str, str]] = []
    core_data = {"version": "12.0_SOTA_MODULAR", "protocols": protocols}
    if not OLD_XML_PATH.exists():
        return core_data
    try:
        tree = ET.parse(OLD_XML_PATH)
        cog_core = tree.getroot().find("CognitiveCore")
        if cog_core is not None:
            for rule in cog_core.findall(".//Rule"):
                logic = rule.find("Logic")
                algorithm = rule.find("Algorithm")
                content = (
                    logic.text
                    if logic is not None
                    else (algorithm.text if algorithm is not None else "")
                )
                if content:
                    protocols.append(
                        {
                            "id": str(rule.get("id", "Unknown")),
                            "priority": str(rule.get("priority", "NORMAL")),
                            "logic": content.strip(),
                        }
                    )
    except Exception:
        pass
    return core_data


def generate_ai_description(
    file_name: str, file_content: str, is_translation: bool = False
) -> str:
    if not HAS_VROMLIX_UTILS:
        return "Sin descripción."

    if is_translation:
        dprint(f"   🤖 Gemini traduciendo abstract de: {file_name}...")
        prompt = f"Traduce esta definición técnica al español en UNA SOLA ORACIÓN (máx 20 palabras). Manten los términos técnicos en inglés si es necesario:\n{file_content}"
    else:
        dprint(f"   🤖 Gemini auto-describiendo: {file_name}...")
        prompt = f"Eres el Bibliotecario de Vromlix Prime. Lee el archivo y escribe UNA SOLA ORACIÓN (máx 20 palabras) describiendo su función técnica. NOMBRE: {file_name}\nCONTENIDO:\n{file_content[:3000]}"

    try:
        client = genai.Client(api_key=vromlix.get_api_key())
        response = client.models.generate_content(
            model=vromlix.get_model("VOLUMEN"),
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        return (response.text or "").strip()
    except Exception:
        return "Auto-descripción fallida."


def parse_local_metadata(filepath: Path) -> tuple[str, bool]:
    """Retorna una tupla: (texto_extraido, requiere_traduccion_por_ia)"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read(2500)

            # 1. Buscar descripciones explícitas en español (Prioridad 1)
            m = (
                re.search(r"# @description (.*)", content)
                or re.search(r"\*\*Descripción:\*\*\s*(.*)", content)
                or re.search(r"<reasoning>(.*?)</reasoning>", content, re.DOTALL)
                or re.search(
                    r"descripcion:\s*[\"']?(.*?)[\"']?(?:\n|$)", content, re.IGNORECASE
                )
            )
            if m:
                return m.group(1).strip().replace("\n", " "), False

            # 2. Buscar el Abstract en inglés de las notas atómicas (Prioridad 2)
            m_abstract = re.search(r"> \[!abstract\].*?\n>\s*(.*?)(?:\n|$)", content)
            if m_abstract:
                english_abstract = m_abstract.group(1).strip()
                return english_abstract, True

    except Exception:
        pass
    return "", False


def get_files_to_scan():
    files_to_scan = []
    valid_extensions = {".py", ".xml", ".md", ".json", ".txt", ".csv"}

    # Exclusiones estrictas para no indexar repositorios anidados ni basura
    excluded_dirs = {
        "NexoContable",
        "00_sandbox",
        "venv",
        ".venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        ".git",
        ".github",
        "Technical_Lexicon",
    }
    excluded_files = {
        "04_Knowledge_Index.xml",
        "04_Knowledge_Index.json",
        "12_Code_Repository.md",
        "config_api_keys_secrets.py",
        "vromlix_snapshot.md",
        "config_api_keys_secrets.example.py",
        "pyproject.toml",
    }

    def is_excluded(path: Path) -> bool:
        """Verifica si el archivo o cualquiera de sus padres está en la lista de exclusión."""
        for part in path.parts:
            if part in excluded_dirs or part.startswith("."):
                # Permitir archivos ocultos si no están en excluded_dirs,
                # pero el rglob ya filtra por nombre. Aquí somos más estrictos.
                if part in excluded_dirs:
                    return True
        return False

    # Ahora escanea scripts, prompts, docs y config (Internos a VROMLIX_CORE)
    for folder_name in ["04_scripts", "03_prompts", "05_docs", "00_config"]:
        folder_path = BASE_DIR / folder_name
        if folder_path.exists():
            for f in folder_path.rglob("*"):
                if f.is_file() and not f.name.startswith("."):
                    if not is_excluded(f.relative_to(BASE_DIR)):
                        if f.suffix.lower() in valid_extensions:
                            if f.name not in excluded_files and not f.name.startswith(
                                "index_VROMLIX_CORE_"
                            ):
                                files_to_scan.append((f, folder_name))

    # Escanear los Repositorios Externos Centralizados desde Utils
    for repo_path in vromlix.paths.repos_externos:
        if repo_path.exists():
            for f in repo_path.rglob("*"):  # Usar rglob para consistencia
                if f.is_file() and not f.name.startswith("."):
                    if not is_excluded(f.relative_to(repo_path.parent)):
                        if (
                            f.suffix.lower() in valid_extensions
                            and f.name not in excluded_files
                        ):
                            files_to_scan.append((f, f"REPO_{repo_path.name.upper()}"))

    # Escanear archivos sueltos en la raíz de VROMLIX_CORE
    for f in BASE_DIR.iterdir():
        if f.is_file() and f.name.endswith(".py") and not f.name.startswith("."):
            if f.name not in excluded_files:
                files_to_scan.append((f, "core_system"))

    return files_to_scan


def run_indexer():
    dprint("\n🧠 Auto-Indexador SOTA en ejecución...")
    existing_index: dict[str, Any] = {"files": {}}
    if NEW_JSON_PATH.exists():
        try:
            with open(NEW_JSON_PATH, "r", encoding="utf-8") as f:
                existing_index = json.load(f)
        except Exception:
            pass

    new_index = {"cognitive_core": extract_cognitive_core(), "files": {}}
    updated_count = 0

    files_list = get_files_to_scan()
    total_files = len(files_list)
    dprint(f"📊 Total de archivos a procesar: {total_files}")

    for filepath, type_folder in files_list:
        filename = filepath.name
        mtime = os.path.getmtime(filepath)
        old_record = existing_index.get("files", {}).get(filename)

        # Si el mtime es el mismo, evaluamos si la descripción previa es un placeholder de error
        is_failure_desc = old_record and old_record.get("description") in [
            "Auto-descripción fallida.",
            "Sin descripción.",
            "Archivo binario o codificación no soportada.",
        ]

        if old_record and old_record.get("mtime") == mtime and not is_failure_desc:
            new_index["files"][filename] = old_record
            continue
        else:
            desc_raw, needs_translation = parse_local_metadata(filepath)

            if needs_translation:
                desc = generate_ai_description(filename, desc_raw, is_translation=True)
                updated_count += 1
            elif desc_raw:
                desc = desc_raw
            else:
                # Fallback a leer todo el archivo y auto-describir
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        full_content = f.read()
                    desc = generate_ai_description(
                        filename, full_content, is_translation=False
                    )
                    updated_count += 1
                except UnicodeDecodeError:
                    desc = "Archivo binario o codificación no soportada."
                except Exception as e:
                    desc = f"Error leyendo archivo: {e}"

            # Cálculo de ruta relativa estricta (Soporta repositorios externos con ../)
            try:
                rel_path = str(filepath.relative_to(BASE_DIR)).replace("\\", "/")
            except ValueError:
                rel_path = "../" + str(filepath.relative_to(BASE_DIR.parent)).replace(
                    "\\", "/"
                )

            new_index["files"][filename] = {
                "path": rel_path,
                "type": type_folder,
                "description": desc,
                "mtime": mtime,
                "last_indexed": datetime.now().isoformat(),
            }

    with open(NEW_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(new_index, f, indent=4, ensure_ascii=False)

    dprint(f"✅ Índice actualizado. {updated_count} archivos procesados por IA.")


if __name__ == "__main__":
    run_indexer()
