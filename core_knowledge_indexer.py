#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "tenacity>=9.0.0",
#     "httpx>=0.28.1",
#     "instructor>=1.7.0",
#     "google-genai>=1.68.0",
#     "pydantic>=2.12.5",
#     "feedparser>=6.0.12",
#     "lxml>=5.1.0",
#     "sqlite-vec>=0.1.3",
#     "tqdm>=4.66.0",
#     "markitdown>=0.0.1a4",
#     "umap-learn>=0.5.11",
#     "scikit-learn>=1.5.0",
#     "numpy>=1.24.0",
#     "llama-cpp-python>=0.2.56",
# ]
# ///

# -*- coding: utf-8 -*-
# @description SOTA Differential Ingestion Engine (v5.3)
# to feed the vector database with native batching.

import csv
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
from pathlib import Path

import sqlite_vec
from tenacity import retry, stop_after_attempt, wait_exponential

# Multimodal SOTA Extraction Factory
try:
    from markitdown import MarkItDown

    HAS_MARKITDOWN = True
except ImportError:
    HAS_MARKITDOWN = False

# Transversal injection to locate the Orchestrator
sys.path.append(str(Path(__file__).parents[1]))
from vromlix_utils import vromlix

# --- CENTRALIZED CONFIGURATION ---
# Strict mathematical limit: 35 chunks * ~500 tokens = ~17,500 tokens per Batch
BATCH_SIZE = 35


class VromlixKnowledgeIndexer:
    def __init__(self, source_dir: str, db_name: str) -> None:
        self.target_dir = source_dir
        self.db_path = str(vromlix.paths.databases / db_name)
        if not self.db_path:
            raise ValueError("❌ Target database not specified.")

        Path(Path(self.db_path).parent).mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.db_path)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)

        self.cursor = self.db.cursor()

        # SOTA Multimodal Engine
        self.md_engine = None
        if HAS_MARKITDOWN:
            logging.info(" [SOTA] Initializing MarkItDown (Universal Engine)...")
            # Silenciar warnings de MarkItDown
            import warnings

            warnings.filterwarnings("ignore", message="Could not get FontBBox")
            self.md_engine = MarkItDown()

        self.model_id = vromlix.get_model("EMBEDDINGS")

        self._init_db()

    def _init_db(self) -> None:
        self.cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS vromlix_vectors USING vec0(
                id INTEGER PRIMARY KEY,
                embedding float[768]
            );
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS vromlix_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT,
                chunk_type TEXT,
                content TEXT,
                network_type TEXT DEFAULT 'W',
                confidence_score REAL DEFAULT 1.0,
                tree_level INTEGER DEFAULT 0,
                parent_id INTEGER,
                cluster_id INTEGER
            );
        """)

        # Safe migration for existing databases (Prevents data deletion)
        try:
            self.cursor.execute(
                "ALTER TABLE vromlix_metadata ADD COLUMN network_type TEXT DEFAULT 'W'"
            )
            self.cursor.execute(
                "ALTER TABLE vromlix_metadata ADD COLUMN confidence_score REAL DEFAULT 1.0"
            )
            self.cursor.execute(
                "ALTER TABLE vromlix_metadata ADD COLUMN tree_level INTEGER DEFAULT 0"
            )
            self.cursor.execute("ALTER TABLE vromlix_metadata ADD COLUMN parent_id INTEGER")
            self.cursor.execute("ALTER TABLE vromlix_metadata ADD COLUMN cluster_id INTEGER")
        except sqlite3.OperationalError:
            pass  # Columns already exist

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS vromlix_file_hashes (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.db.commit()

    def _calculate_md5(self, filepath: str) -> str:
        hash_md5 = hashlib.md5()
        try:
            with Path(filepath).open("rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""

    def _parse_xml_or_md(self, filepath: str) -> list[str]:
        chunks = []
        try:
            with Path(filepath).open(encoding="utf-8") as f:
                content = f.read()
            paragraphs = content.split("\n\n")
            current_chunk = ""
            for p in paragraphs:
                if len(current_chunk) + len(p) < 2500:
                    current_chunk += p + "\n\n"
                else:
                    chunks.append(f"File [{Path(filepath).name}]:\n{current_chunk.strip()}")
                    current_chunk = p + "\n\n"
            if current_chunk.strip():
                chunks.append(f"File [{Path(filepath).name}]:\n{current_chunk.strip()}")
        except Exception as e:
            print(f"❌ Error parsing XML/MD {filepath}: {e}")
        return chunks

    def _parse_json(self, filepath: str) -> list[str]:
        chunks = []
        try:
            with Path(filepath).open(encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                chunks.extend(
                    [
                        f"JSON Object [{Path(filepath).name}]:\n"
                        f"{json.dumps(item, ensure_ascii=False)}"
                        for item in data
                    ]
                )
            else:
                chunks.append(
                    f"JSON Data [{Path(filepath).name}]:\n{json.dumps(data, ensure_ascii=False)}"
                )
        except Exception as e:
            print(f"❌ Error parsing JSON {filepath}: {e}")
        return chunks

    def _parse_txt_deep_memory(self, filepath: str) -> list[str]:
        chunks = []
        try:
            with Path(filepath).open(encoding="utf-8") as f:
                content = f.read()
            if (
                "================================================================================"
                in content
            ):
                blocks = re.split(r"={20,}", content)
                for block in blocks:
                    if block.strip():
                        truncated_block = block.strip()
                        chunks.append(f"Deep Memory Record:\n{truncated_block[:3000]}")
            else:
                return self._parse_xml_or_md(filepath)
        except Exception as e:
            print(f"❌ Error parsing TXT {filepath}: {e}")
        return chunks

    def _parse_csv(self, filepath: str) -> list[str]:
        chunks = []
        try:
            with Path(filepath).open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_text = ", ".join([f"{k}: {v}" for k, v in row.items()])
                    chunks.append(f"CSV Record [{Path(filepath).name}]: {row_text}")
        except Exception as e:
            print(f"❌ Error parsing CSV {filepath}: {e}")
        return chunks

    def route_file(self, filepath: str) -> list[str]:
        """Routes the file to the corresponding parser or the Multimodal MarkItDown engine."""
        ext = filepath.lower().split(".")[-1]

        # 1. Optimized native parsers
        if ext in ["xml", "md", "py"]:
            return self._parse_xml_or_md(filepath)
        elif ext == "json":
            return self._parse_json(filepath)
        elif ext == "txt":
            return self._parse_txt_deep_memory(filepath)
        elif ext == "csv":
            return self._parse_csv(filepath)

        # 2. SOTA Multimodal Engine (Universal Ingestion)
        elif ext in ["pdf", "docx", "pptx", "xlsx", "xls", "odt"]:
            return self._parse_via_markitdown(filepath)

        return []

    def _parse_via_markitdown(self, filepath: str) -> list[str]:
        """Extracts Markdown content from ANY complex file with SOTA Resilience."""
        if not self.md_engine:
            return []
        try:
            # MarkItDown converts anything to native structured Markdown
            result = self.md_engine.convert(filepath)
            content = result.text_content

            # Reusing the XML/MD chunking logic
            return self._chunk_content_sota(content, filepath)
        except Exception as e:
            logging.error(f"❌ MarkItDown error for {filepath}: {e}")
            return []

    def _chunk_content_sota(self, content: str, filepath: str) -> list[str]:
        """Universal chunking for content extracted by MarkItDown."""
        chunks = []
        paragraphs = content.split("\n\n")
        current_chunk = ""
        for p in paragraphs:
            if len(current_chunk) + len(p) < 2500:
                current_chunk += p + "\n\n"
            else:
                chunks.append(
                    f"Multimodal Knowledge [{Path(filepath).name}]:\n{current_chunk.strip()}"
                )
                current_chunk = p + "\n\n"
        if current_chunk.strip():
            chunks.append(f"Multimodal Knowledge [{Path(filepath).name}]:\n{current_chunk.strip()}")
        return chunks

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _embed_and_store(self, task, cursor):
        """
        Vectorización Universal: Delega la inteligencia a vromlix_utils.
        Elimina la dependencia directa de la SDK de Google.
        """
        try:
            text = task["text"]
            filepath = task["filepath"]

            # --- PHASE 1: EMBEDDINGS GENERATION (Centralized via Jina) ---
            vector = vromlix.get_embeddings(text, role="EMBEDDINGS")

            if not vector:
                return False

            # --- PHASE 2: SOTA NETWORK CLASSIFICATION ---
            net_type = "W"
            fp_lower = filepath.lower()
            if fp_lower.endswith(".csv") or any(x in fp_lower for x in ["chat", "session"]):
                net_type = "B"
            elif any(x in fp_lower for x in ["raptor", "summary"]):
                net_type = "S"
            elif any(x in fp_lower for x in ["perfil_psicologico", "opinion"]):
                net_type = "O"

            # --- PHASE 3: DUAL PERSISTENCE ---
            cursor.execute(
                (
                    "INSERT INTO vromlix_metadata "
                    "(source_file, chunk_type, content, network_type) VALUES (?, ?, ?, ?)"
                ),
                (filepath, "auto_routed", text, net_type),
            )
            row_id = cursor.lastrowid

            cursor.execute(
                "INSERT INTO vromlix_vectors (id, embedding) VALUES (?, ?)",
                (row_id, json.dumps(vector)),
            )
            return True
        except Exception as e:
            logging.error(f"Error en embedding {filepath}: {e}")
            raise

    def process_single_file(self, filepath: str) -> bool:
        """Incremental ingestion of a single file with hash management and prior purging."""
        # SOTA FIX: Normalizar a ruta absoluta canónica para consistencia en el ledger
        filepath = str(Path(filepath).resolve())
        if not Path(filepath).exists():
            print(f"❌ File does not exist: {filepath}")
            return False

        cursor = self.db.cursor()
        current_hash = self._calculate_md5(filepath)
        filename = Path(filepath).name

        # 1. Check if the file is already indexed with the same hash
        cursor.execute("SELECT file_hash FROM vromlix_file_hashes WHERE file_path = ?", (filepath,))
        row = cursor.fetchone()
        if row and row[0] == current_hash:
            print(f"✨ {filename} is already updated in the index. Skipping...")
            return True

        print(f"📥 Indexing individual file: {filename}...")

        # 2. Surgical purge of previous records for this file
        cursor.execute("SELECT id FROM vromlix_metadata WHERE source_file = ?", (filepath,))
        ids_to_delete = [str(r[0]) for r in cursor.fetchall()]
        if ids_to_delete:
            placeholders = ",".join("?" * len(ids_to_delete))
            cursor.execute(
                f"DELETE FROM vromlix_vectors WHERE id IN ({placeholders})",
                ids_to_delete,
            )
            cursor.execute(
                f"DELETE FROM vromlix_metadata WHERE id IN ({placeholders})",
                ids_to_delete,
            )

        # 3. Chunking and Vectorization
        chunks = self.route_file(filepath)
        if not chunks:
            print(f"⚠️ No chunks extracted from {filename}.")
            return False

        total_inserted = 0
        for text in chunks:
            task = {"filepath": filepath, "text": text}
            if self._embed_and_store(task, cursor):
                total_inserted += 1

        # 4. Update hash ledger
        cursor.execute(
            "INSERT OR REPLACE INTO vromlix_file_hashes "
            "(file_path, file_hash, last_updated) VALUES (?, ?, CURRENT_TIMESTAMP)",
            (filepath, current_hash),
        )
        self.db.commit()
        print(f"✅ {filename} successfully indexed ({total_inserted} chunks).")
        return True

    def process_directories(
        self,
        target_extensions: tuple[str, ...] = (
            ".xml",
            ".md",
            ".csv",
            ".txt",
            ".py",
            ".json",
            ".pdf",
            ".docx",
            ".pptx",
            ".xlsx",
            ".xls",
            ".odt",
        ),
    ) -> None:
        cursor = self.db.cursor()

        # 0. DEFINITIVE SOTA EXCLUSIONS
        # Ahora el obrero solo ignora carpetas de sistema.
        # Las carpetas de proyectos y storage ahora son permitidas
        # porque el Capataz las enviará explícitamente.

        # 1. ADVANCED PURGE PHASE (Smart Garbage Collection)
        cursor.execute("SELECT file_path FROM vromlix_file_hashes")
        tracked_files = [row[0] for row in cursor.fetchall()]

        EXCLUDED_DIRS = {
            "venv",
            ".venv",
            "env",
            ".mypy_cache",
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".uv",
            "node_modules",
            ".codex_backups",
            ".secrets",
            "codex_memory",
            "json",
        }

        for tracked_file in tracked_files:
            path_parts = set(tracked_file.replace("\\", "/").split("/"))
            is_now_excluded = bool(path_parts.intersection(EXCLUDED_DIRS))

            filename = Path(tracked_file).name
            is_ignored_file = filename == "vromlix_snapshot.md" or filename.startswith(
                "index_VROMLIX_CORE_"
            )

            if not Path(tracked_file).exists() or is_now_excluded or is_ignored_file:
                cursor.execute(
                    "SELECT id FROM vromlix_metadata WHERE source_file = ?",
                    (tracked_file,),
                )
                ids_to_delete = [str(r[0]) for r in cursor.fetchall()]

                if ids_to_delete:
                    placeholders = ",".join("?" * len(ids_to_delete))
                    cursor.execute(
                        f"DELETE FROM vromlix_vectors WHERE id IN ({placeholders})",
                        ids_to_delete,
                    )
                    cursor.execute(
                        f"DELETE FROM vromlix_metadata WHERE id IN ({placeholders})",
                        ids_to_delete,
                    )
                cursor.execute(
                    "DELETE FROM vromlix_file_hashes WHERE file_path = ?",
                    (tracked_file,),
                )
        self.db.commit()

        # Mostrar resumen de purga
        purged_count = len(
            [
                f
                for f in tracked_files
                if not Path(f).exists()
                or set(f.replace("\\", "/").split("/")).intersection(EXCLUDED_DIRS)
                or Path(f).name == "vromlix_snapshot.md"
                or Path(f).name.startswith("index_VROMLIX_CORE_")
            ]
        )
        if purged_count > 0:
            print(f"   🗑️ {purged_count} registros obsoletos eliminados")

        # 2. CHECKING PHASE WITH SOTA EXCLUSION
        files_to_process = []
        dir_file_counts = {}

        # El obrero solo escanea el directorio que se le ordenó
        for directory in [self.target_dir]:
            if not Path(directory).exists():
                print(f"⚠️ Directorio origen no encontrado: {directory}")
                continue
            dir_file_counts[directory] = 0

            for root, dirs, files in os.walk(directory):
                if directory == "." and root == ".":
                    valid_dirs = [
                        d
                        for d in dirs
                        if d not in EXCLUDED_DIRS
                        and not d.startswith(".")
                        and d not in ["02_projects", "99_deep_storage", "05_docs"]
                    ]
                else:
                    valid_dirs = [
                        d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")
                    ]
                dirs.clear()
                dirs.extend(valid_dirs)

                for file in files:
                    if (
                        file.startswith("backup_")
                        or file.endswith(".meta.json")
                        or file.endswith(".data.json")
                        or file == "vromlix_snapshot.md"
                        or file.startswith("index_VROMLIX_CORE_")
                        or file == "config_api_keys_secrets.example.py"
                        or file == "pyproject.toml"
                    ):
                        continue

                    if file.endswith(target_extensions):
                        filepath = (Path(root) / file).resolve()
                        current_hash = self._calculate_md5(str(filepath))
                        cursor.execute(
                            "SELECT file_hash FROM vromlix_file_hashes WHERE file_path = ?",
                            (str(filepath),),
                        )
                        row = cursor.fetchone()

                        if row is None:
                            files_to_process.append(
                                {
                                    "path": str(filepath),
                                    "hash": current_hash,
                                    "status": "new",
                                }
                            )
                        elif row[0] != current_hash:
                            files_to_process.append(
                                {
                                    "path": str(filepath),
                                    "hash": current_hash,
                                    "status": "modified",
                                }
                            )
                        dir_file_counts[directory] += 1

        if not files_to_process:
            print(f"   ✅ Up to date ({sum(dir_file_counts.values())} files).")
            return

        print(f"   🔎 Scan completed: {len(files_to_process)} entries to evaluate.")

        # 3. SURGICAL REPLACEMENT PHASE (TRUE DIFFERENTIAL AT CHUNK LEVEL)
        all_tasks = []
        files_without_new_chunks = set()
        total_chunks_deleted = 0

        for file_data in files_to_process:
            filepath = file_data["path"]
            new_chunks = self.route_file(filepath)
            new_chunks_set = {c.strip() for c in new_chunks if c.strip()}

            cursor.execute(
                "SELECT id, content FROM vromlix_metadata WHERE source_file = ?",
                (filepath,),
            )
            existing_records = cursor.fetchall()

            existing_contents = {row[1].strip(): str(row[0]) for row in existing_records}

            ids_to_delete = [
                existing_contents[content]
                for content in existing_contents
                if content not in new_chunks_set
            ]

            if ids_to_delete:
                placeholders = ",".join("?" * len(ids_to_delete))
                cursor.execute(
                    f"DELETE FROM vromlix_vectors WHERE id IN ({placeholders})",
                    ids_to_delete,
                )
                cursor.execute(
                    f"DELETE FROM vromlix_metadata WHERE id IN ({placeholders})",
                    ids_to_delete,
                )
                total_chunks_deleted += len(ids_to_delete)

            chunks_to_process = [text for text in new_chunks_set if text not in existing_contents]

            if not chunks_to_process:
                files_without_new_chunks.add((filepath, file_data["hash"]))

            all_tasks.extend(
                [
                    {
                        "filepath": filepath,
                        "file_hash": file_data["hash"],
                        "text": text,
                    }
                    for text in chunks_to_process
                ]
            )
        self.db.commit()

        # 4. SOTA DRIP VECTORIZATION (1-by-1)
        total_tasks = len(all_tasks)
        processed_files = set()

        if total_tasks > 0:
            print(f"\n   Processing {total_tasks} chunks...")
            for i, task in enumerate(all_tasks):
                try:
                    if self._embed_and_store(task, cursor):
                        processed_files.add((task["filepath"], task["file_hash"]))

                    if (i + 1) % 50 == 0 or (i + 1) == total_tasks:
                        progress_pct = (i + 1) / total_tasks * 100
                        sys.stdout.write(
                            f"\r   \u26cf Vectorizando chunks [{i + 1}/{total_tasks}] "
                            f"({progress_pct:.1f}%)..."
                        )
                        sys.stdout.flush()
                except Exception as e:
                    msg = str(e)
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                        print(
                            f"\n   [QUOTA] Regional limit reached on "
                            f"{Path(task['filepath']).name}. Retrying..."
                        )
                    else:
                        print(f"\n   Error in {Path(task['filepath']).name}: {msg[:100]}...")

                if (i + 1) % 50 == 0:
                    self.db.commit()
            print(f"\n✅ Procesamiento completado: {len(processed_files)} chunks nuevos integrados")
        # Silenciamos detalles para el dashboard
        # if total_chunks_deleted > 0:
        #     print(f"   🧹 Purged {total_chunks_deleted} obsolete data points.") UPDATE HASH LEDGER
        # We merge the files that went through the API and those that only had deletions
        all_processed_files = processed_files.union(files_without_new_chunks)

        for filepath, f_hash in all_processed_files:
            cursor.execute(
                """
                INSERT OR REPLACE INTO vromlix_file_hashes (file_path, file_hash, last_updated)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
                (filepath, f_hash),
            )

        # --- DEFLATION PROTOCOL (VACUUM) ---
        print("\n   🧹 Executing VACUUM to deflate and optimize the SQLite database...")
        self.db.commit()  # Seal any pending writes
        self.db.isolation_level = None  # Release Python's secret transaction (Autocommit)
        cursor.execute("VACUUM;")
        self.db.isolation_level = "DEFERRED"  # Restore transactional security

        print(f"\n🎉 SOTA INGESTION COMPLETED. Database secured at {self.db_path}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VROMLIX Universal Indexer Worker")
    parser.add_argument("--source", required=True, help="Ruta absoluta de la carpeta a escanear")
    parser.add_argument(
        "--db",
        required=True,
        help="Nombre del archivo SQLite de salida (ej. projects.sqlite)",
    )
    args = parser.parse_args()

    indexer = VromlixKnowledgeIndexer(source_dir=args.source, db_name=args.db)
    indexer.process_directories()
