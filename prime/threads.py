"""
prime.threads.py — VROMLIX Prime: Background Threads and Async Agents
DeepMemoryRetriever, RealTimeVectorizer, SubconsciousUpdater, DocumentForgeAgent.
Split from core_vromlix_prime.py (God Class refactor).
"""

import json
import logging
import os
import sqlite3
import sys
import threading
from datetime import datetime
from pathlib import Path

from vromlix_utils import vromlix

try:
    import sqlite_vec

    HAS_SQLITE_VEC = True
except ImportError:
    sqlite_vec = None
    HAS_SQLITE_VEC = False


class DeepMemoryRetriever:
    """Augmented Retrieval Engine (RAG) connected to sqlite-vec."""

    def __init__(self):
        self.db_path = str(vromlix.paths.databases / "vromlix_memory.sqlite")
        embed_config = vromlix.get_secret("EMBEDDINGS")
        self.embedding_model = (
            embed_config.get("model_id") or embed_config.get("primary")
            if embed_config
            else "gemini-embedding-2-preview"
        )

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        if not HAS_SQLITE_VEC or not Path(self.db_path).exists():
            return ""
        query_vector = vromlix.get_embeddings(query)
        if not query_vector:
            return ""

        try:
            db = sqlite3.connect(self.db_path)
            db.enable_load_extension(True)
            sqlite_vec.load(db)
            db.enable_load_extension(False)
            cursor = db.cursor()

            cursor.execute(
                """
                WITH seed_nodes AS (
                    SELECT m.id, m.source_file, m.network_type, m.confidence_score, v.distance
                    FROM vromlix_vectors v
                    JOIN vromlix_metadata m ON v.id = m.id
                    WHERE v.embedding MATCH ? AND k = ?
                )
                SELECT
                    (SELECT group_concat(content, '\n... [ADJACENT NODE] ...\n')
                     FROM (
                         SELECT content FROM vromlix_metadata
                         WHERE source_file = s.source_file
                         AND id BETWEEN s.id - 1 AND s.id + 1
                         ORDER BY id ASC
                     )
                    ) AS expanded_content,
                    s.network_type, s.confidence_score, s.distance
                FROM seed_nodes s ORDER BY s.distance ASC
            """,
                (json.dumps(query_vector), top_k),
            )

            results = cursor.fetchall()
            db.close()

            if not results:
                return ""

            context_blocks = [
                "=== DEEP MEMORY CONTEXT (HINDSIGHT W-B-O-S) ===",
                "INSTRUCTION: [W]=World Facts. [B]=Experiences. [O]=Opinions. [S]=Summaries.",
            ]
            for i, (content, net_type, conf_score, _distance) in enumerate(results):
                net_label = {
                    "W": "WORLD FACT",
                    "B": "BIOGRAPHICAL EXP",
                    "O": f"OPINION (Conf: {conf_score})",
                    "S": "SUMMARY",
                }.get(net_type, "UNKNOWN")
                context_blocks.extend([f"--- Fragment {i + 1} [{net_label}] ---", content.strip()])

            logging.info(f"📚 RAG: {len(results)} fragments retrieved.")
            return "\n".join(context_blocks)
        except Exception as e:
            logging.error(f"Error in RAG: {e}")
            return ""


class RealTimeVectorizer(threading.Thread):
    """Vectorizes interaction in the background and saves it to sqlite-vec."""

    def __init__(self, interaction_text: str, db_path: str, embedding_model: str):
        super().__init__()
        self.interaction_text = interaction_text
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.daemon = True

    def run(self):
        if not HAS_SQLITE_VEC or not Path(self.db_path).exists():
            return
        try:
            vector = vromlix.get_embeddings(self.interaction_text)
            db = sqlite3.connect(self.db_path)
            db.enable_load_extension(True)
            sqlite_vec.load(db)
            db.enable_load_extension(False)
            cursor = db.cursor()
            cursor.execute(
                "INSERT INTO vromlix_metadata"
                " (source_file, chunk_type, content, network_type) VALUES (?, ?, ?, ?)",
                ("LIVE_SESSION", "real_time_memory", self.interaction_text, "B"),
            )
            cursor.execute(
                "INSERT INTO vromlix_vectors (id, embedding) VALUES (?, ?)",
                (cursor.lastrowid, json.dumps(vector)),
            )
            db.commit()
            db.close()
        except Exception as e:
            logging.error(f"Vectorizer failed: {e}")


class SubconsciousUpdater(threading.Thread):
    """Analyzes chat looking for new data for 01_Dynamic_Profile.xml."""

    def __init__(self, interaction_text: str, profile_path: Path, profiler_prompt: str):
        super().__init__()
        self.interaction_text = interaction_text
        self.profile_path = profile_path
        self.profiler_prompt = profiler_prompt
        self.daemon = True

    def run(self):
        prompt = self.profiler_prompt.format(
            interaction_text=self.interaction_text,
            timestamp=datetime.now().strftime("%Y-%m-%d"),
        )
        try:
            res = vromlix.query_universal_llm(system_prompt="", user_prompt=prompt, role="VOLUMEN")
            result = res.text.strip()
            if result != "NONE" and "<user_fact" in result:
                db_path = vromlix.paths.databases / "vromlix_master_brain.sqlite"
                if db_path.exists():
                    import sqlite3

                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT content FROM user_profile WHERE section = 'biographical_history'"
                    )
                    row = cursor.fetchone()
                    if row:
                        content = row[0]
                        if "</historical_archive>" in content:
                            new_content = content.replace(
                                "</historical_archive>",
                                f"  {result}\n</historical_archive>",
                            )
                            cursor.execute(
                                "UPDATE user_profile SET content = ?, updated_at = ? WHERE section = 'biographical_history'",
                                (new_content, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                            )
                            conn.commit()
                    conn.close()
        except Exception as e:
            logging.error(f"SubconsciousUpdater failed: {e}")


class DocumentForgeAgent:
    """SOTA Operative Agent (The Forge): Generates complete documents in the Sandbox."""

    def __init__(self, forge_prompt: str):
        self.sandbox_dir = vromlix.paths.sandbox
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.forge_prompt = forge_prompt

    def execute_missions(self, missions_json: str):
        try:
            missions = json.loads(missions_json)
        except json.JSONDecodeError as e:
            logging.error(f"   ❌ Error parsing JSON missions: {e}")
            return

        total = len(missions)
        for index, mission in enumerate(missions, 1):
            target, source, instruction = (
                mission.get("target", "unknown.txt"),
                mission.get("source", "NONE"),
                mission.get("instruction", ""),
            )
            source_content = "No source provided."

            if source != "NONE":
                source_path = self._find_file(source)
                if source_path:
                    try:
                        with source_path.open(encoding="utf-8") as f:
                            source_content = f.read()
                    except Exception as e:
                        source_content = f"Error: {e}"

            prompt = self.forge_prompt.format(
                target=target, instruction=instruction, source_content=source_content
            )

            try:
                res = vromlix.query_universal_llm(
                    system_prompt="", user_prompt=prompt, role="VOLUMEN"
                )
                content = res.text.strip()
                if content.startswith("```") and content.endswith("```"):
                    lines = content.split("\n")
                    if len(lines) > 2:
                        content = "\n".join(lines[1:-1])

                with (self.sandbox_dir / target).open("w", encoding="utf-8") as f:
                    f.write(content)
                sys.stdout.write("\033[K")
                logging.info(f"   ⚡ [{index}/{total}] Mission completed -> SANDBOX/{target}")
            except Exception as e:
                sys.stdout.write("\033[K")
                logging.error(f"   ❌ [{index}/{total}] Error forging '{target}': {e}")

    def _find_file(self, filename: str) -> Path | None:
        for root, _, files in os.walk(vromlix.paths.base):
            if "venv" in root or ".git" in root or "databases" in root:
                continue
            if filename in files:
                return Path(root) / filename
        return None
