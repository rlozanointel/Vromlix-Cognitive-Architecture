#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "pydantic>=2.12.5", "google-genai>=1.68.0", "instructor>=1.7.0",
#   "tenacity>=9.0.0", "httpx>=0.28.1", "numpy>=2.2.6", "sqlite-vec>=0.1.9",
#   "feedparser>=6.0.12", "duckduckgo-search>=8.1.1", "markitdown>=0.0.1a4",
#   "lxml>=5.1.0", "tqdm>=4.67.3", "google-api-core", "urllib3", "jsonref",
#   "openai>=1.14.0", "llama-cpp-python>=0.2.56"
# ]
# ///
"""
VROMLIX UTILS - Centralized Orchestrator v3.0.4
Purpose: Environment detection, API Hot-Swapping, RAPTOR Consolidation, and I/O Management.
"""

import importlib.util
import json
import logging
import os
import re
import sqlite3
import sys
import threading
import time
import urllib.parse
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

# --- VROMLIX DEPENDENCY SHIELD (SOTA Resilience) ---
try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError:
    # Fallback to dummy decorators if tenacity is missing
    def retry(*args, **kwargs):
        return lambda f: f

    def stop_after_attempt(*args, **kwargs):
        return None

    def wait_exponential(*args, **kwargs):
        return None


warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")
logging.getLogger("duckduckgo_search").setLevel(logging.ERROR)
logging.getLogger("google.ai.generativelanguage").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_MINLOGLEVEL"] = "2"


class VromlixUsage:
    """Normalizes token counting between Google and OpenAI/GitHub."""

    def __init__(self, prompt_tokens, candidate_tokens):
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = candidate_tokens


class VromlixResponse:
    """Unified response object to maintain compatibility with Prime."""

    def __init__(self, text: str, thoughts: str = "", usage=None):
        self.text = text
        self.thoughts = thoughts
        self.usage_metadata = usage


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [VROMLIX PRIME] - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


class AFCSilencer(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not ("AFC is enabled" in msg or "response: https://" in msg)


for logger_name in [
    "google",
    "google.auth",
    "google.api_core",
    "google.genai",
    "httpx",
    "httpcore",
    "absl",
    "urllib3",
]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)
    logger.addFilter(AFCSilencer())

root_logger = logging.getLogger()
root_logger.addFilter(AFCSilencer())
for handler in root_logger.handlers:
    handler.addFilter(AFCSilencer())


class OSINTGrounder:
    """SOTA OSINT Engine. Uses Google News RSS Multiplexing to evade IP blocks."""

    @staticmethod
    def clean_value(val: Any) -> Any:
        if isinstance(val, str):
            clean = re.sub(r"<[^>]+>", "", val)
            return re.sub(r"\s+", " ", clean).strip()
        return val

    @classmethod
    def fetch_news_rss(cls, query: str, max_results: int = 30) -> list[dict[str, Any]]:
        encoded_query = urllib.parse.quote_plus(f'"{query}"')
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        results = []
        try:
            import httpx

            response = httpx.get(url, headers=headers, timeout=10.0)
            if response.status_code == 200:
                import feedparser

                feed = feedparser.parse(response.content)
                results.extend(
                    [
                        {
                            "title": getattr(entry, "title", ""),
                            "link": getattr(entry, "link", ""),
                            "published": getattr(entry, "published", ""),
                            "source": getattr(entry, "source", {}).get("title", ""),
                        }
                        for entry in feed.entries[:max_results]
                    ]
                )
        except Exception as e:
            logging.warning(f"[OSINT] Warning in RSS feed for '{query}': {e}")
        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=False,
    )
    def fetch_rss_summary(self, url: str) -> str:
        try:
            import httpx

            with httpx.Client(timeout=15.0) as client:
                resp = client.get(url)
                resp.raise_for_status()
                import feedparser

                feed = feedparser.parse(resp.text)
                items = [f"{e.title}: {e.summary[:200]}" for e in feed.entries[:3]]
                return "\n".join(items) if items else "No entries found."
        except Exception as e:
            logging.warning(f"[OSINT] RSS Fail: {e}")
            return f"RSS Error: {e}"

    @classmethod
    def execute_deep_research(cls, queries: list[str], prompt_template: str = "") -> str:
        if not queries:
            return ""
        master_doc: dict[str, Any] = {"research_data": {}}
        logging.info(
            f"🌐 [OSINT] Phase 1 (MAP): Extracting Google News RSS for {len(queries)} queries..."
        )

        for query in queries:
            master_doc["research_data"][query] = {
                "news_results": cls.fetch_news_rss(query, max_results=40)
            }
            time.sleep(1.0)

        logging.info("🧠 [OSINT] Phase 2 (REDUCE): Synthesizing data with Flash Lite...")
        raw_json = json.dumps(master_doc, ensure_ascii=False)
        json_slice = str(raw_json)[:80000]
        prompt_sintesis = prompt_template.format(raw_json=json_slice)

        try:
            # FIX SOTA: Delegar al Bridge Universal para usar la cascada de 4 niveles
            response = vromlix.query_universal_llm(
                system_prompt="You are an OSINT Analyst.",
                user_prompt=prompt_sintesis,
                role="VOLUMEN",
            )
            logging.info("✅ [OSINT] Executive RSS Report generated successfully via Bridge.")
            return response.text.strip()
        except Exception as e:
            logging.error(f"❌ Error in OSINT synthesis: {e}")
            return "ERROR: News synthesis failed."


class ActiveKeyManager:
    """SOTA Round-Robin Load Balancer for Google APIs with SQLite persistence."""

    def __init__(
        self,
        keys: list[str],
        cooldown_seconds: float = 61.0,
        db_path: str | None = None,
    ):
        import random

        self.keys = keys
        self.cooldown = cooldown_seconds
        self.db_path = db_path
        self.last_used = dict.fromkeys(keys, 0.0)
        self.suspended_until = dict.fromkeys(keys, 0.0)
        self.recent_failures: list[float] = []
        self.current_idx = random.randint(0, max(0, len(keys) - 1)) if keys else 0
        self._db_lock = threading.Lock()
        self._db_conn: sqlite3.Connection | None = None

        logging.getLogger("instructor").setLevel(logging.ERROR)
        logging.getLogger("google.genai").setLevel(logging.ERROR)

        if self.db_path:
            self._init_db()
            self._load_suspensions()

    def _get_db(self) -> sqlite3.Connection:
        assert self.db_path is not None, "_get_db called without a db_path"
        if self._db_conn is None:
            # We already asserted db_path is not None above
            db_p = str(self.db_path)
            self._db_conn = sqlite3.connect(db_p, check_same_thread=False)
            self._db_conn.execute("PRAGMA journal_mode=WAL")
            self._db_conn.execute("PRAGMA busy_timeout=5000")
        return self._db_conn

    def _init_db(self):
        assert self.db_path is not None, "_init_db called without a db_path"
        Path(Path(self.db_path).parent).mkdir(parents=True, exist_ok=True)
        conn = self._get_db()
        with self._db_lock:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS key_suspensions "
                "(api_key TEXT PRIMARY KEY, suspended_until REAL)"
            )
            conn.commit()

    def _load_suspensions(self):
        try:
            with self._db_lock:
                conn = self._get_db()
                cursor = conn.cursor()
                cursor.execute("SELECT api_key, suspended_until FROM key_suspensions")
                for key_val, suspended_until_val in cursor.fetchall():
                    s_key = str(key_val)
                    s_until = float(suspended_until_val)
                    if s_key in self.suspended_until and s_until > time.time():
                        self.suspended_until[s_key] = s_until
        except Exception as e:
            logging.error(f"Error loading DB suspensions: {e}")

    def get_fresh_key(self) -> str | None:
        if not self.keys:
            return None
        now = time.time()

        def _wait_needed(key: str) -> float:
            return max(
                0.0,
                self.cooldown - (now - self.last_used[key]),
                self.suspended_until[key] - now,
            )

        ready = [k for k in self.keys if _wait_needed(k) == 0.0]
        if ready:
            key = ready[self.current_idx % len(ready)]
            self.last_used[key] = now
            self.current_idx = (self.current_idx + 1) % len(self.keys)
            return key

        best_key = min(self.keys, key=_wait_needed)
        wait_time = min(10.0, _wait_needed(best_key))
        time.sleep(wait_time)

        self.last_used[best_key] = time.time()
        self.current_idx = (self.keys.index(best_key) + 1) % len(self.keys)
        return best_key

    def report_failure(self, key: str, error_msg: str = ""):
        now = time.time()
        error_lower = error_msg.lower()

        if "quota" in error_lower or "exhausted" in error_lower or "daily" in error_lower:
            duration = 86400.0
            logging.warning("🔴 [KeyManager] Daily quota exhausted. Key suspended for 24h.")
        else:
            duration = 300.0
            logging.warning("🟡 [KeyManager] Temporary saturation (503/RPM). Key suspended for 5m.")

        if key in self.suspended_until:
            suspend_time = now + duration
            self.suspended_until[key] = suspend_time
            self.recent_failures.append(now)

            if self.db_path:
                with self._db_lock:
                    try:
                        conn = self._get_db()
                        conn.execute(
                            "INSERT OR REPLACE INTO key_suspensions "
                            "(api_key, suspended_until) VALUES (?, ?)",
                            (key, suspend_time),
                        )
                        conn.commit()
                    except Exception as e:
                        logging.error(f"Error saving suspension to DB: {e}")

            self.recent_failures = [f for f in self.recent_failures if now - f < 60]
            if len(self.recent_failures) > 5:
                logging.warning(
                    "🛑 [KeyManager] Multiple failures detected. Tactical 15s pause "
                    "to prevent IP ban."
                )
                time.sleep(15)

    def get_status(self) -> dict[str, Any]:
        """Returns health status of the key manager."""
        now = time.time()
        active_keys = [k for k, t in self.suspended_until.items() if t <= now]
        suspended_keys = [k for k, t in self.suspended_until.items() if t > now]
        return {
            "total_keys": len(self.keys),
            "active_keys": len(active_keys),
            "suspended_keys": len(suspended_keys),
            "recent_failures": len(self.recent_failures),
            "health": "OK" if active_keys else "CRITICAL",
        }


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Per-provider circuit breaker with zero blocking."""

    failure_threshold: int = 3
    recovery_timeout: float = 30.0
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def can_execute(self) -> bool:
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    return True
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self.failure_count = 0
            self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN


class VromlixOrchestrator:
    _ROLE_ALIASES: ClassVar[dict[str, str]] = {
        "PRIME": "PRECISION",
        "CONSULTA": "VOLUMEN",
        "REASONING": "PRECISION",
        "INDEXER": "VOLUMEN",
    }

    def __init__(self):
        self.is_colab = "google.colab" in sys.modules
        self.is_firebase = (
            "FIREBASE_CONFIG" in os.environ
            or "K_SERVICE" in os.environ
            or "FUNCTION_NAME" in os.environ
        )
        self.is_local = (
            not self.is_colab and not self.is_firebase and sys.platform.startswith("linux")
        )

        if self.is_colab:
            self.base_path = Path("/content/drive/MyDrive/VROMLIX_CORE")
        elif self.is_local:
            self.base_path = Path(__file__).resolve().parent
        else:
            self.base_path = Path("/tmp/VROMLIX_CORE")

        self.paths = self._init_paths()
        self.config_path = self._get_config_path()
        self.config = self._load_config()

        # Load environment variables from .env files
        self._load_env()

        # Load keys from env (GEMINI_API_KEYS) or fallback to config
        keys_list = []
        if "GEMINI_API_KEYS" in os.environ:
            keys_list = [k.strip() for k in os.environ["GEMINI_API_KEYS"].split(",") if k.strip()]
        if not keys_list and self.config:
            keys_list = getattr(self.config, "LISTA_DE_APIS", [])

        api_db_path = str(self.paths.databases / "vromlix_api_manager.sqlite")
        self.key_manager = ActiveKeyManager(keys_list, cooldown_seconds=61.0, db_path=api_db_path)

        # NUEVO: Rotador SOTA para tus 6 cuentas de Groq
        groq_keys_list = []
        if "GROQ_API_KEYS" in os.environ:
            groq_keys_list = [
                k.strip() for k in os.environ["GROQ_API_KEYS"].split(",") if k.strip()
            ]
        if not groq_keys_list and self.config:
            groq_keys_list = getattr(self.config, "LISTA_GROQ", [])

        self.groq_key_manager = None
        if groq_keys_list:
            groq_db_path = str(self.paths.databases / "groq_api_manager.sqlite")
            self.groq_key_manager = ActiveKeyManager(
                groq_keys_list, cooldown_seconds=61.0, db_path=groq_db_path
            )

        self._model_cache = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

        # SOTA LOCAL EMBEDDER (Dinámico desde Config)
        self._local_embedder = None
        try:
            from llama_cpp import Llama

            embed_config = getattr(self.config, "MODEL_ROUTING_REGISTRY", {}).get("EMBEDDINGS", {})
            model_name = (
                embed_config.get("primary")
                if embed_config.get("provider") == "local_llama_cpp"
                else embed_config.get("fallback")
            )
            if not model_name:
                model_name = "v5-nano-retrieval-Q8_0.gguf"

            model_path = self.paths.local_llms / model_name
            if not model_path.exists():
                model_path = self.base_path / "models" / model_name

            if model_path.exists():
                logging.debug("🚀 [SOTA] Initializing Jina v5 Nano via llama.cpp...")
                self._local_embedder = Llama(
                    model_path=str(model_path),
                    embedding=True,
                    n_ctx=8192,
                    verbose=False,
                )
        except ImportError:
            logging.debug(
                "⚙️ llama-cpp-python no disponible. Si se usa Jina local, el pipeline fallará."
            )

    def _init_paths(self):
        class Paths:
            def __init__(self, base_path):
                self.base = Path(base_path).resolve()
                self.sandbox = self.base / "00_sandbox"
                self.config = self.base / "01_config"
                self.config_xml = self.base / "01_config/xml"
                self.config_json = self.base / "01_config/json"
                self.scripts = self.base / "04_scripts"
                self.docs = self.base / "05_docs"

                # SSoT: Centralized dynamic DB Path
                db_env = os.environ.get("VROMLIX_DATA")
                if db_env:
                    self.databases = Path(db_env).resolve()
                else:
                    self.databases = self.base.parent / "VROMLIX_DATA"

                self.projects = self.base / "07_projects"
                self.tests = self.base / "08_tests"
                self.shell_scripts = self.base / "09_shell_scripts"
                self.deep_storage = self.base / "99_deep_storage"
                self.apeiron = self.base / "07_projects/01_Apeiron"
                self.icatmor = self.base / "07_projects/02_ICATMOR"

                # Local LLMs definition
                llm_env = os.environ.get("VROMLIX_LOCAL_LLMS")
                if llm_env:
                    self.local_llms = Path(llm_env).resolve()
                else:
                    self.local_llms = self.base.parent / "Local_LLMs"
                    if not self.local_llms.exists():
                        self.local_llms = self.base / "models"

                self.repos_externos = [
                    self.base.parent / "cv",
                    self.base.parent / "blueprints",
                    self.base.parent / "rlozano.intel",
                    self.base.parent / "vromlix-cognitive-architecture",
                ]

        paths = Paths(self.base_path)
        paths.sandbox.mkdir(parents=True, exist_ok=True)
        return paths

    def _load_env(self):
        env_paths = [
            self.base_path / ".env",
            self.base_path / ".secrets" / ".env",
            Path.home() / ".vromlix.env",
            Path.cwd() / ".env",
        ]
        loaded_vars = {}
        for p in env_paths:
            if p.exists():
                try:
                    with open(p, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            if "=" in line:
                                k, v = line.split("=", 1)
                                k = k.strip()
                                v = v.strip()
                                # Remove quotes
                                if (v.startswith('"') and v.endswith('"')) or (
                                    v.startswith("'") and v.endswith("'")
                                ):
                                    v = v[1:-1]
                                os.environ.setdefault(k, v)
                                loaded_vars[k] = v
                except Exception as e:
                    logging.warning(f"Error loading env from {p}: {e}")
        return loaded_vars

    def _get_config_path(self):
        if self.is_colab:
            import importlib

            colab_drive = importlib.import_module("google.colab").drive
            if not Path("/content/drive").exists():
                colab_drive.mount("/content/drive")
            return "/content/drive/MyDrive/Colab Notebooks/config_api_keys_secrets.py"
        elif self.is_local:
            return str(self.paths.base / ".secrets" / "config_api_keys_secrets.py")
        return "./.secrets/config_api_keys_secrets.py"

    def _load_config(self):
        if not Path(self.config_path).exists():
            logging.warning(f"WARNING: No config found at {self.config_path}. Using defaults.")
            return None
        spec = importlib.util.spec_from_file_location("config_module", self.config_path)
        if spec is None or spec.loader is None:
            return None
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        return config

    def get_model(self, role: str) -> str:
        normalized_role = self._ROLE_ALIASES.get(role.upper(), role.upper())
        cache_key = f"model_{normalized_role}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        if not self.config and normalized_role in ["VOLUMEN", "PRECISION", "MASIVO"]:
            model = "gemini-3.1-flash-lite-preview"
            self._model_cache[cache_key] = model
            return model

        try:
            registry = getattr(self.config, "MODEL_ROUTING_REGISTRY", {})
            if normalized_role in registry:
                model_config = registry[normalized_role]
                _m: str | None = model_config.get("primary", model_config.get("model_id"))
                if _m:
                    self._model_cache[cache_key] = _m
                    return _m
        except Exception as e:
            logging.warning(f"Error reading MODEL_ROUTING_REGISTRY: {e}")

        attr_name = f"MODELO_{normalized_role}"
        _m2: str | None = getattr(self.config, attr_name, None)
        if not _m2:
            _m2 = getattr(self.config, f"MODELO_{role.upper()}", None)

        if not _m2:
            raise ValueError(f"HARD STOP: Model for role '{role}' is not defined.")

        self._model_cache[cache_key] = _m2
        return _m2

    def get_model_capabilities(self, role):
        try:
            registry = getattr(self.config, "MODEL_ROUTING_REGISTRY", {})
            normalized_role = self._ROLE_ALIASES.get(role.upper(), role.upper())
            if normalized_role in registry:
                config = registry[normalized_role]
                return {
                    "model_id": config.get("primary", config.get("model_id")),
                    "provider": config.get("provider", "google"),
                    "fallback": config.get("fallback"),
                    "fallback_provider": config.get("fallback_provider", "google"),
                    "fallback_2": config.get("fallback_2"),
                    "fallback_2_provider": config.get("fallback_2_provider", "github"),
                    "fallback_3": config.get("fallback_3"),
                    "fallback_3_provider": config.get("fallback_3_provider", "local_llama_cpp"),
                    "role": config.get("role"),
                    "cost_tier": config.get("cost_tier"),
                    "capability": config.get("capability"),
                    "rpd_per_key": config.get("rpd_per_key"),
                    "rpm_per_key": config.get("rpm_per_key"),
                    "tpm_limit": config.get("tpm_limit"),
                }
        except Exception as e:
            logging.warning(f"Error getting model capabilities: {e}")
        return None

    def get_api_key(self, provider: str = "gemini"):
        if provider.lower() == "groq" and self.groq_key_manager:
            return self.groq_key_manager.get_fresh_key()
        return self.key_manager.get_fresh_key()

    def report_exhaustion(self, key: str, provider: str = "gemini", error_msg: str = ""):
        if provider.lower() == "groq" and self.groq_key_manager:
            self.groq_key_manager.report_failure(key, error_msg=error_msg)
        else:
            self.key_manager.report_failure(key, error_msg=error_msg)

    def get_secret(self, key_name: str):
        try:
            if key_name == "GEMINI_API_KEY" and hasattr(self, "key_manager"):
                fresh_key = self.key_manager.get_fresh_key()
                if fresh_key:
                    return fresh_key

            # Prioritize environment variables loaded from .env
            if key_name in os.environ:
                return os.environ[key_name]
            if key_name == "GITHUB_TOKEN" and "GITHUB_PAT" in os.environ:
                return os.environ["GITHUB_PAT"]
            if key_name == "GITHUB_PAT" and "GITHUB_TOKEN" in os.environ:
                return os.environ["GITHUB_TOKEN"]

            registry = getattr(self.config, "MODEL_ROUTING_REGISTRY", {})
            if key_name in registry:
                return registry[key_name]
            if hasattr(self.config, key_name):
                return getattr(self.config, key_name)
            return getattr(self.config, "SECRETS", {}).get(key_name)
        except Exception:
            return None

    def get_safety_settings(self) -> list[Any]:
        try:
            from google.genai import types

            return [
                types.SafetySetting(
                    category=getattr(types.HarmCategory, c),
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                )
                for c in [
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                ]
            ]
        except (ImportError, AttributeError):
            return []

    def query_local_ollm(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        """SOTA Bare-Metal Execution (No Ollama). Locks RAM to prevent swap thrashing."""
        if not hasattr(self, "_local_llm") or getattr(self, "_local_llm_name", "") != model_name:
            try:
                from llama_cpp import Llama

                model_path = self.paths.local_llms / model_name
                if not model_path.exists():
                    # Fallback SOTA: Intentar en la carpeta models interna si la externa falla
                    model_path = self.base_path / "models" / model_name

                if not model_path.exists():
                    raise FileNotFoundError(f"Local model not found: {model_path}")

                logging.info(f"🚀 [EDGE COMPUTE] Loading {model_name} directly into RAM...")
                self._local_llm = Llama(
                    model_path=str(model_path),
                    n_ctx=4096,
                    n_threads=4,  # Optimizado para P-Cores de i5
                    use_mlock=True,  # CRÍTICO: Evita que Ubuntu use el disco duro (Swap)
                    verbose=False,
                )
                self._local_llm_name = model_name
            except ImportError as e:
                raise RuntimeError("llama-cpp-python is required for bare-metal execution.") from e

        # Forzar modo determinista si es Qwen
        if "qwen" in model_name.lower():
            system_prompt += (
                "\nCRITICAL: Output ONLY valid JSON or direct answers. "
                "NO chain-of-thought. NO <think> tags."
            )

        response = self._local_llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=1024,
        )
        return response["choices"][0]["message"]["content"].strip()

    def get_embeddings(self, text: str | list[str], role: str = "EMBEDDINGS") -> Any:
        info = self.get_model_capabilities(role)
        providers = []
        if info and info.get("model_id"):
            providers.append({"id": info.get("provider"), "model": info.get("model_id")})
        if info and info.get("fallback"):
            providers.append({"id": info.get("fallback_provider"), "model": info.get("fallback")})

        is_batch = isinstance(text, list)

        for prov in providers:
            if prov["id"] == "local_llama_cpp":
                if hasattr(self, "_local_embedder") and self._local_embedder:
                    import os
                    import sys

                    fd = sys.stderr.fileno()
                    old_stderr = os.dup(fd)
                    devnull = os.open(os.devnull, os.O_WRONLY)
                    os.dup2(devnull, fd)
                    try:
                        if is_batch:
                            embeddings = []
                            for t in text:
                                response = self._local_embedder.create_embedding(t)
                                embeddings.append(response["data"][0]["embedding"])
                            return embeddings
                        else:
                            response = self._local_embedder.create_embedding(text)
                            return response["data"][0]["embedding"]
                    except Exception as e:
                        os.dup2(old_stderr, fd)
                        logging.error(f"[EMBEDDINGS] Error en Jina Local: {e}")
                        continue
                    finally:
                        os.dup2(old_stderr, fd)
                        os.close(devnull)
                        os.close(old_stderr)

            elif prov["id"] == "google":
                try:
                    from google import genai
                    from google.genai import types

                    api_key = self.get_api_key(provider="gemini")
                    client = genai.Client(api_key=api_key)
                    res = client.models.embed_content(
                        model=prov["model"],
                        contents=text,
                        config=types.EmbedContentConfig(output_dimensionality=768),
                    )
                    if is_batch:
                        return [e.values for e in res.embeddings]
                    else:
                        return res.embeddings[0].values
                except Exception as e:
                    logging.error(f"[EMBEDDINGS] Error en Gemini: {e}")
                    continue

        raise RuntimeError("⚠️ HARD STOP: Todos los proveedores de Embeddings fallaron.")

    def query_universal_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        role: str = "VOLUMEN",
        response_model: Any = None,
        tools: list | None = None,
        thinking: bool = False,
    ) -> Any:
        """Universal Bridge SOTA v4.0: 4-Level Waterfall (Groq -> Gemini -> GitHub -> Local)."""
        info = self.get_model_capabilities(role)
        if not info:
            return VromlixResponse("ERROR: Role not configured.")

        # Construcción dinámica de la cascada de resiliencia
        providers = []
        if info.get("model_id"):
            providers.append({"id": info.get("provider", "google"), "model": info.get("model_id")})
        if info.get("fallback"):
            providers.append(
                {"id": info.get("fallback_provider", "google"), "model": info.get("fallback")}
            )
        if info.get("fallback_2"):
            providers.append(
                {"id": info.get("fallback_2_provider", "github"), "model": info.get("fallback_2")}
            )
        if info.get("fallback_3"):
            providers.append(
                {
                    "id": info.get("fallback_3_provider", "local_llama_cpp"),
                    "model": info.get("fallback_3"),
                }
            )

        last_error = None
        for prov in providers:
            if not prov["model"]:
                continue

            cb = self._circuit_breakers.setdefault(prov["id"], CircuitBreaker())
            if not cb.can_execute():
                logging.warning(f"⚡ Circuit OPEN for {prov['id']}. Skipping to next fallback.")
                continue

            try:
                match prov["id"]:
                    case "groq":
                        import instructor
                        from openai import OpenAI

                        api_key = self.get_api_key(provider="groq")
                        if not api_key:
                            continue
                        client_groq = OpenAI(
                            base_url="https://api.groq.com/openai/v1", api_key=api_key
                        )

                        if response_model:
                            instr_groq = instructor.from_openai(
                                client_groq, mode=instructor.Mode.JSON
                            )
                            res = instr_groq.chat.completions.create(
                                model=prov["model"],
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt},
                                ],
                                response_model=response_model,
                            )
                            cb.record_success()
                            return res

                        resp_groq = client_groq.chat.completions.create(
                            model=prov["model"],
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=0.1,
                        )
                        text = resp_groq.choices[0].message.content or ""
                        _usage = resp_groq.usage
                        usage = VromlixUsage(
                            _usage.prompt_tokens if _usage else 0,
                            _usage.completion_tokens if _usage else 0,
                        )
                        cb.record_success()
                        return VromlixResponse(text, "", usage)

                    case "google":
                        import instructor
                        from google import genai
                        from google.genai import types

                        api_key = self.get_api_key(provider="gemini")
                        if not api_key:
                            continue
                        client = genai.Client(
                            api_key=api_key,
                            http_options=types.HttpOptions(api_version="v1alpha"),
                        )

                        if response_model:
                            instr = instructor.from_genai(
                                client, mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS
                            )
                            res = instr.chat.completions.create(
                                model=prov["model"],
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt},
                                ],
                                response_model=response_model,
                            )
                            cb.record_success()
                            return res

                        resp = client.models.generate_content(
                            model=prov["model"],
                            contents=user_prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=system_prompt,
                                temperature=0.2,
                                tools=tools,
                                thinking_config=(
                                    types.ThinkingConfig(include_thoughts=True)
                                    if thinking
                                    else None
                                ),
                            ),
                        )
                        text, thoughts = "", ""
                        candidate = resp.candidates[0] if resp.candidates else None
                        if candidate and candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                part_text: str = part.text or ""
                                if getattr(part, "thought", False):
                                    thoughts += part_text
                                elif part_text:
                                    text += part_text
                        cb.record_success()
                        return VromlixResponse(
                            text or (resp.text or ""), thoughts, resp.usage_metadata
                        )

                    case "github":
                        import instructor
                        from openai import OpenAI

                        token = self.get_secret("GITHUB_TOKEN")
                        if not token:
                            continue
                        client_gh = OpenAI(
                            base_url="https://models.inference.ai.azure.com", api_key=token
                        )
                        final_user = (
                            f"[SYSTEM: Think step by step]\n{user_prompt}"
                            if thinking
                            else user_prompt
                        )

                        if response_model:
                            instr_gh = instructor.from_openai(client_gh)
                            res = instr_gh.chat.completions.create(
                                model=prov["model"],
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": final_user},
                                ],
                                response_model=response_model,
                            )
                            cb.record_success()
                            return res

                        resp_gh = client_gh.chat.completions.create(
                            model=prov["model"],
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": final_user},
                            ],
                            temperature=0.1,
                        )
                        raw_text: str = resp_gh.choices[0].message.content or ""
                        thoughts_match = re.search(r"<thought>(.*?)</thought>", raw_text, re.DOTALL)
                        clean_text = re.sub(
                            r"<thought>.*?</thought>", "", raw_text, flags=re.DOTALL
                        ).strip()
                        final_thoughts = (
                            thoughts_match.group(1).strip()
                            if thoughts_match
                            else (f"Rescue via {prov['id'].upper()}..." if thinking else "")
                        )
                        _usage = resp_gh.usage
                        usage = VromlixUsage(
                            _usage.prompt_tokens if _usage else 0,
                            _usage.completion_tokens if _usage else 0,
                        )
                        cb.record_success()
                        return VromlixResponse(clean_text, final_thoughts, usage)

                    case "local_llama_cpp":
                        # Fallback 3: Supervivencia Zero-Internet
                        logging.warning(
                            f"🛡️ Activando Fallback Local (Zero-Internet): {prov['model']}"
                        )
                        text = self.query_local_ollm(
                            model_name=prov["model"],
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                        )

                        # Si se requiere JSON, intentamos parsearlo (SOTA Blindado)
                        if response_model:
                            try:
                                import json

                                # 1. Limpiar posibles bloques markdown (```json ... ```)
                                clean_text = re.sub(
                                    r"```(?:json)?\n?(.*?)\n?```", r"\1", text, flags=re.DOTALL
                                )

                                # 2. Extraer el objeto JSON puro
                                json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
                                if json_match:
                                    parsed = json.loads(json_match.group(0))
                                    return response_model(**parsed)
                                else:
                                    raise ValueError("No JSON object found in local model output.")
                            except Exception as e:
                                logging.error(f"Local JSON parse failed: {e}")
                                raise ValueError(
                                    "Local model failed to produce valid JSON schema."
                                ) from e

                        cb.record_success()
                        return VromlixResponse(text, "", None)

            except Exception as e:
                last_error = str(e)
                cb.record_failure()

                # Manejo SOTA de Error 429 (Exponential Backoff para Groq y Gemini)
                if any(
                    err in last_error.lower()
                    for err in ["429", "quota", "503", "overloaded", "rate_limit"]
                ):
                    wait_time = 60.0
                    if prov["id"] == "groq" and hasattr(e, "response") and e.response is not None:
                        wait_time = float(e.response.headers.get("retry-after", 60.0))

                    logging.warning(
                        f"⏳ {prov['id'].upper()} Rate Limit (429). "
                        f"Pausando {wait_time}s antes de rotar..."
                    )
                    time.sleep(wait_time)

                    if prov["id"] in ["google", "groq"]:
                        self.report_exhaustion(api_key, provider=prov["id"], error_msg=last_error)
                    continue

                logging.error(f"❌ Error en {prov['id']}: {last_error}")
                continue  # Intenta el siguiente fallback

        return VromlixResponse(f"CRITICAL ERROR: All 4 providers failed. Last error: {last_error}")

    def update_knowledge_base(
        self, source_folder_name: str, db_filename: str, exclude_list: list[str] | None = None
    ):
        """
        Orchestrator Command: Orders the Universal Indexer to vectorize a specific folder.
        Example: update_knowledge_base(
            "02_projects", "projects_index.sqlite", exclude_list=["temp"]
        )
        """
        import subprocess

        indexer_script = self.paths.scripts / "core" / "core_knowledge_indexer.py"

        # Si le pasan la raíz vacía, escanea VROMLIX_CORE
        if source_folder_name == "" or source_folder_name == ".":
            source_path = self.paths.base
        else:
            source_path = self.paths.base / source_folder_name

        logging.info(f"📢 [Orchestrator] Ordering Indexer: {source_path.name} -> {db_filename}")

        cmd = [
            "uv",
            "run",
            str(indexer_script),
            "--source",
            str(source_path),
            "--db",
            db_filename,
        ]

        if exclude_list:
            cmd.extend(["--exclude", *exclude_list])

        try:
            subprocess.run(cmd, check=True)
            logging.debug(f"✅ [Orchestrator] Indexing completed for {db_filename}")
        except subprocess.CalledProcessError as e:
            logging.error(f"❌ [Orchestrator] Indexer failed for {db_filename}: {e}")


vromlix = VromlixOrchestrator()


class IOManager:
    @staticmethod
    def select_file(provided_path: str | None = None, title: str = "Select a file") -> str | None:
        if provided_path:
            try:
                if Path(provided_path).exists():
                    return provided_path
            except (PermissionError, OSError):
                return None
        if vromlix.is_colab:
            import importlib

            colab_files = importlib.import_module("google.colab").files
            print(f"📂 {title}:")
            uploaded = colab_files.upload()
            return next(iter(uploaded.keys())) if uploaded else None
        elif vromlix.is_local:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(title=title)
            return file_path if file_path else None
        else:
            file_path = input(f"📂 {title}: ").strip()
            return file_path if Path(file_path).exists() else None

    @staticmethod
    def select_files(title: str = "Select files") -> list[str]:
        if vromlix.is_colab:
            import importlib

            colab_files = importlib.import_module("google.colab").files
            uploaded = colab_files.upload()
            return list(uploaded.keys()) if uploaded else []
        elif vromlix.is_local:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            file_paths = filedialog.askopenfilenames(title=title)
            return list(file_paths) if file_paths else []
        else:
            paths = input(f"📂 {title} (comma separated): ")
            return [p.strip() for p in paths.split(",") if Path(p.strip().exists())]

    @staticmethod
    def export_file(file_path: str):
        if vromlix.is_colab:
            import importlib

            colab_files = importlib.import_module("google.colab").files
            colab_files.download(file_path)
        else:
            logging.info(f"✅ File saved: {Path(file_path).name}")

    @staticmethod
    def select_directory(title: str = "Select a directory") -> str | None:
        if vromlix.is_colab:
            dir_path = input(f"📂 {title} in Colab: ").strip()
            return dir_path if Path(dir_path).exists() else None
        elif vromlix.is_local:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            dir_path = filedialog.askdirectory(title=title)
            return dir_path if dir_path else None
        else:
            dir_path = input(f"📂 {title}: ").strip()
            return dir_path if Path(dir_path).exists() else None


class ModelSelector:
    """Intelligent Model Selection Layer for VROMLIX Orchestrator."""

    def __init__(self, orchestrator: "VromlixOrchestrator"):
        self.vromlix = orchestrator

    def get_model_info(self, role: str) -> dict[str, Any] | None:
        """Delegates to orchestrator capabilities."""
        return self.vromlix.get_model_capabilities(role)

    def get_model_with_fallback(self, role: str) -> str:
        """Delegates to orchestrator model retrieval."""
        return self.vromlix.get_model(role)

    def get_optimal_model(self, task_type: str, complexity: str = "medium") -> str:
        """SOTA Logic: Maps human task descriptions to technical roles."""
        role_mapping = {
            "code_generation": "PRECISION",
            "reasoning": "PRECISION",
            "auditing": "PRECISION",
            "json_extraction": "VOLUMEN",
            "rag": "VOLUMEN",
            "triage": "VOLUMEN",
            "embeddings": "EMBEDDINGS",
            "image_generation": "IMAGEN_FAST",
            "audio_processing": "AUDIO_NATIVO",
        }
        role = role_mapping.get(task_type, "VOLUMEN")

        # Contextual upgrade
        if complexity == "high" and role == "VOLUMEN":
            role = "PRECISION"

        return self.get_model_with_fallback(role)
