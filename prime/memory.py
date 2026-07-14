"""
prime.memory.py — VROMLIX Prime: Short-Term Memory Layer
TokenMonitor, VromlixContextLoader, SessionTracker.
Split from core_vromlix_prime.py (God Class refactor).
"""

import functools
import logging
import re
import threading
from hashlib import md5
from pathlib import Path

from vromlix_utils import vromlix


def _cached_read_file(filepath_str: str, _mtime: float) -> str:
    """Module-level cached file reader, keyed by (path, mtime). Avoids B019."""
    try:
        with Path(filepath_str).open(encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading {filepath_str}: {e}")
        return ""


class TokenMonitor:
    """Tracks current session token consumption by expert/API. Thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self.expert_usage: dict[str, dict[str, int]] = {}

    def add_usage(self, expert_id: str, usage_metadata):
        if not usage_metadata:
            return
        with self._lock:
            if expert_id not in self.expert_usage:
                self.expert_usage[expert_id] = {"in": 0, "out": 0}
            self.expert_usage[expert_id]["in"] += (
                getattr(usage_metadata, "prompt_token_count", 0) or 0
            )
            self.expert_usage[expert_id]["out"] += (
                getattr(usage_metadata, "candidates_token_count", 0) or 0
            )

    def get_summary(self) -> str:
        with self._lock:
            if not self.expert_usage:
                return "🪙 Tokens: 0"
            lines = ["📊 Token Breakdown (Session):"]
            total_in, total_out = 0, 0
            for exp, data in self.expert_usage.items():
                lines.append(f"   ├─ [{exp}]: {data['in']} In | {data['out']} Out")
                total_in += data["in"]
                total_out += data["out"]
            lines.append(f"   └─ TOTAL: {total_in} In | {total_out} Out")
            return "\n".join(lines)


@functools.lru_cache(maxsize=64)
class VromlixContextLoader:
    """Loads and merges immutable configuration files from SQLite with fallback."""

    def __init__(self):
        self._file_cache: dict[str, tuple[str, float]] = {}
        self._master_prompt_cache: str | None = None
        self._prompt_hash: str = ""
        self.cache_ttl_seconds = 300  # 5 minutes
        self.base_path: Path = vromlix.paths.base
        self.db_path: Path = vromlix.paths.databases / "vromlix_master_brain.sqlite"
        self.repo_file: Path = self._find_file("Project_Atlas.md")

    def _find_file(self, filename: str) -> Path:
        for path in [
            self.base_path,
            vromlix.paths.config_json,
            vromlix.paths.docs,
        ]:
            if (path / filename).exists():
                return path / filename
        return self.base_path / filename

    def load_system_prompts(self) -> dict:
        prompts: dict[str, str] = {}
        if not self.db_path.exists():
            logging.info(
                "ℹ️ Running in DEMO mode (Master Brain database not found). Loading mock system prompts."
            )
            return {
                "moe_router": "Demo Router: Analyze user query and route it.",
                "ockham_fusion": "Demo Fusion: Summarize inputs in Spanish.",
                "ockham_auditor": "Demo Auditor: Enforce guidelines and trackers.",
                "subconscious_profiler": "Demo Profiler: Extract user facts.",
                "document_forge": "Demo Forge: Write file contents.",
                "osint_synthesis": "Demo OSINT: Synthesize search results.",
            }
        try:
            import sqlite3

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name, instructions FROM protocols WHERE name LIKE 'orchestrator_%'"
                )
                for name, instructions in cursor.fetchall():
                    p_id = name.replace("orchestrator_", "")
                    prompts[p_id] = instructions
        except Exception as e:
            logging.error(f"Error loading system prompts from SQLite: {e}")
        return prompts

    def load_moe_routing(self) -> str:
        if not self.db_path.exists():
            # For public repo demo, attempt to load local file if exists
            local_moe = self.base_path / "03_prompts" / "moe_routing.json"
            if local_moe.exists():
                return self._read_file(local_moe)
            return "[]"
        try:
            import sqlite3

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT content FROM user_profile WHERE section = 'moe_routing'")
                row = cursor.fetchone()
                if row:
                    return row[0]
        except Exception as e:
            logging.error(f"Error loading MoE routing from SQLite: {e}")
        return "[]"

    def _read_file_cached(self, filepath: Path) -> str:
        mtime = filepath.stat().st_mtime if filepath.exists() else 0.0
        return _cached_read_file(str(filepath), mtime)

    def _read_file(self, filepath: Path) -> str:
        if not filepath.exists():
            logging.error(f"CRITICAL: Core file not found -> {filepath.name}")
            return f"<!-- ERROR: {filepath.name} MISSING -->"
        try:
            with filepath.open(encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logging.error(f"Error reading {filepath.name}: {e}")
            return ""

    def _calculate_prompt_hash(self) -> str:
        if self.db_path.exists():
            return md5(str(self.db_path.stat().st_mtime).encode()).hexdigest()
        return "demo_hash"

    def _compress_prompt(self, prompt: str) -> str:
        compressed = re.sub(r"\n\s*\n", "\n", prompt)
        compressed = re.sub(r"\s+", " ", compressed).strip()
        if len(prompt) != len(compressed):
            reduction = (len(prompt) - len(compressed)) / len(prompt) * 100
            logging.info(f"🗜️ Prompt compressed: {reduction:.1f}% size reduction")
        return compressed

    def build_master_system_prompt(self) -> str:
        current_hash = self._calculate_prompt_hash()
        if self._master_prompt_cache is None or self._prompt_hash != current_hash:
            logging.info("🧠 Assembling Master System Prompt (Kernel + Profile + MoE)...")

            logic_text = ""
            profile_text = ""

            if not self.db_path.exists():
                logic_text = "Demo Operating Logic: Be helpful and precise."
                profile_text = "Demo Profile: User is Roger, an AI systems developer."
            else:
                try:
                    import sqlite3

                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT content FROM user_profile WHERE section = 'system_operating_logic'"
                        )
                        row_logic = cursor.fetchone()
                        logic_text = row_logic[0] if row_logic else ""

                        cursor.execute(
                            "SELECT content FROM user_profile WHERE section = 'dynamic_profile'"
                        )
                        row_profile = cursor.fetchone()
                        profile_text = row_profile[0] if row_profile else ""
                except Exception as e:
                    logging.error(f"Error reading profile from master brain SQLite: {e}")

            master_prompt = f"""
You are VROMLIX PRIME, Polymatic Operating System Orchestrator.
You operate strictly under the architectural definitions provided below.
Your cognitive state is externalized in these documents. Do not hallucinate features.

=== 1. SYSTEM OPERATING LOGIC (KERNEL) ===
{logic_text}

=== 2. DYNAMIC PROFILE (THE SOUL) ===
{profile_text}

=== ORCHESTRATOR DIRECTIVES ===
1. Analyze the user's input and the recent conversation history.
2. Adopt the persona, mechanics, and constraints of the assigned Expert(s).
3. ALWAYS append the VROMLIX_STATE_TRACKER at the end of your response.

=== FILE PATCHING PROTOCOL ===
You are a CONSULTATIVE Senior Architect. DO NOT generate code patches proactively.
1. First, analyze the user's request and provide your findings, analysis, or theoretical solution.
2. End your response by ASKING the user: "Do you want me to generate the code patch to apply these
   changes in [filename]?"
3. ONLY if the user explicitly replies with a "Yes" or gives a direct command to patch, you MUST use
   the following exact format to apply surgical patches. DO NOT rewrite the entire file:
📄 File: [filename.ext]
<<<< SEARCH
[Exact lines to find and replace. Must match the original file perfectly]
====
[New lines to insert]
>>>> REPLACE
"""
            self._master_prompt_cache = self._compress_prompt(master_prompt)
            self._prompt_hash = current_hash
        return self._master_prompt_cache if self._master_prompt_cache is not None else ""


class SessionTracker:
    """Manages current session's short-term memory using SQLite."""

    def __init__(self):
        import importlib
        import sys

        # Append path to 04_scripts/web relative to prime/memory.py location
        web_path = str(Path(__file__).parents[2] / "web")
        if web_path not in sys.path:
            sys.path.append(web_path)

        try:
            chat_mod = importlib.import_module("chat_session_manager")
            self.manager = chat_mod.ChatSessionManager()
        except ImportError:
            # Fallback if the path is not correctly resolved in some environments
            logging.error("Failed to dynamically import chat_session_manager.")
            raise

        self.session_id = None

    def start_session(self, model: str = "default", context: str = "") -> str:
        self.session_id = self.manager.create_session(model, context)
        return self.session_id

    def log_interaction(self, role: str, content: str, tokens: int | None = None) -> None:
        if self.session_id:
            self.manager.add_message(self.session_id, role, content, tokens)

    def get_recent_context(self, max_turns: int = 5) -> str:
        if not self.session_id:
            return ""
        try:
            messages = self.manager.get_session_messages(self.session_id)
            recent_messages = (
                messages[-max_turns * 2 :] if len(messages) > max_turns * 2 else messages
            )
            return "\n\n".join(
                [
                    f"{'👤' if msg['role'] == 'user' else '🤖'} {msg['content']}"
                    for msg in recent_messages
                ]
            )
        except Exception as e:
            logging.error(f"Error getting recent context: {e}")
            return ""

    def end_session(self) -> str:
        if self.session_id:
            self.manager.close_session(self.session_id)
            self.session_id = None
        return ""

    def append_state_tracker(
        self, focus: str, locked: str, stack: str, friction: str, loop: str
    ) -> str:
        tracker = (
            f"::: VROMLIX_STATE_TRACKER :::"
            f"\n[FOCUS]::{focus}\n[LOCKED]::{locked}\n[STACK]::{stack}"
            f"\n[FRICTION]::{friction}\n[LOOP]::{loop}\n::: END_TRACKER :::"
        )
        try:
            if self.session_id:
                self.manager.add_message(
                    self.session_id,
                    "system",
                    tracker.strip(),
                    metadata={"type": "state_tracker"},
                )
        except Exception as e:
            logging.error(f"Failed to write tracker: {e}")
        return tracker.strip()
