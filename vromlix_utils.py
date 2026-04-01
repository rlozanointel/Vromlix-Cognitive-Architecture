#!/usr/bin/env -S uv run
# -*- coding: utf-8 -*-
# @description Este módulo orquesta el entorno, gestiona rutas globales y filtra logs para optimizar la ejecución de servicios en Vromlix.
"""
VROMLIX UTILS - Centralized Orchestrator v3.0
Version: 3.0.0 (Marzo 2026) - Active Round-Robin SOTA
Purpose: Environment detection, API Hot-Swapping, Agnostic Routing, Global Paths, and I/O Management.
"""

import os
import sys
import urllib.parse

# Silenciar logs de C++ (gRPC/ABSL) a nivel de Sistema Operativo ANTES de cargar Google SDK
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

import importlib.util
import json
import logging
import re
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")
logging.getLogger("duckduckgo_search").setLevel(logging.ERROR)

# ==============================================================================
# CONFIGURACIÓN GLOBAL DE LOGGING Y SILENCIADOR AFC
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [VROMLIX PRIME] - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


class AFCSilencer(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "AFC is enabled" in msg:
            return False
        if "response: https://" in msg:
            return False  # Silencia los INFO de DuckDuckGo
        return True


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

# Asegurar que los loggers ruidosos silencien esto antes de propagar
for noisy_logger in ["absl", "grpc", "google.api_core", "google.genai"]:
    logger_instance = logging.getLogger(noisy_logger)
    logger_instance.addFilter(AFCSilencer())
    for handler_instance in logger_instance.handlers:
        handler_instance.addFilter(AFCSilencer())


class OSINTGrounder:
    """
    Motor de Inteligencia OSINT SOTA (Zero-Cost Architecture).
    Utiliza Google News RSS Multiplexing para evadir bloqueos IP y extraer noticias en tiempo real.
    """

    @staticmethod
    def clean_value(val: Any) -> Any:
        if isinstance(val, str):
            clean = re.sub(r"<[^>]+>", "", val)
            clean = re.sub(r"\s+", " ", clean)
            return clean.strip()
        return val

    @classmethod
    def fetch_news_rss(cls, query: str, max_results: int = 30) -> list[dict[str, Any]]:
        """Extrae noticias estructuradas vía XML nativo de Google News."""
        import feedparser
        import httpx

        # Forzamos comillas para exactitud: "Nombre de la Startup"
        encoded_query = urllib.parse.quote_plus(f'"{query}"')
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }
        results = []
        try:
            response = httpx.get(url, headers=headers, timeout=10.0)
            if response.status_code == 200:
                feed = feedparser.parse(response.content)
                for entry in feed.entries[:max_results]:
                    results.append(
                        {
                            "title": getattr(entry, "title", ""),
                            "link": getattr(entry, "link", ""),
                            "published": getattr(entry, "published", ""),
                            "source": getattr(entry, "source", {}).get("title", ""),
                        }
                    )
        except Exception as e:
            logging.warning(f"[OSINT] Advertencia en feed RSS para '{query}': {e}")
        return results

    @classmethod
    def execute_deep_research(
        cls, queries: list[str], prompt_template: str = ""
    ) -> str:
        if not queries:
            return ""

        master_doc: dict[str, Any] = {"research_data": {}}
        logging.info(
            f"🌐 [OSINT] Fase 1 (MAP): Extrayendo Google News RSS para {len(queries)} queries..."
        )

        for query in queries:
            query_data = {}
            query_data["news_results"] = cls.fetch_news_rss(query, max_results=40)
            master_doc["research_data"][query] = query_data
            time.sleep(1.0)  # Pausa táctica

        logging.info("🧠 [OSINT] Fase 2 (REDUCE): Sintetizando datos con Flash Lite...")
        raw_json = json.dumps(master_doc, ensure_ascii=False)

        # Inyección de Dependencia: Usamos el prompt que viene del XML
        prompt_sintesis = prompt_template.format(raw_json=raw_json[:80000])

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=vromlix.get_api_key())
            response = client.models.generate_content(
                model=vromlix.get_model("VOLUMEN"),
                contents=prompt_sintesis,
                config=types.GenerateContentConfig(temperature=0.1),
            )
            logging.info("✅ [OSINT] Reporte Ejecutivo RSS generado exitosamente.")
            return response.text.strip()
        except Exception as e:
            logging.error(f"❌ Error en síntesis OSINT: {e}")
            return "ERROR: Fallo en la síntesis de noticias."


class ActiveKeyManager:
    """
    SOTA Round-Robin Load Balancer para APIs de Google.
    Garantiza que una llave no se reutilice antes de un tiempo de enfriamiento (Cooldown).
    """

    def __init__(self, keys: list[str], cooldown_seconds: float = 61.0):
        import random

        self.keys = keys
        self.cooldown = cooldown_seconds
        # Rastrea la última vez (timestamp) que se usó cada llave
        self.last_used = {key: 0.0 for key in keys}
        # Empezar en un índice aleatorio para distribuir el desgaste de RPD (20/día)
        self.current_idx = random.randint(0, max(0, len(keys) - 1)) if keys else 0

    def get_fresh_key(self) -> str | None:
        if not self.keys:
            return None

        total_keys = len(self.keys)
        attempts = 0

        while attempts < total_keys:
            candidate_key = self.keys[self.current_idx]
            time_since_last_use = time.time() - self.last_used[candidate_key]

            # Si la llave ya se enfrió, la entregamos inmediatamente
            if time_since_last_use >= self.cooldown:
                self.last_used[candidate_key] = time.time()
                self.current_idx = (self.current_idx + 1) % total_keys
                return candidate_key

            # Si no está fría, pasamos a la siguiente llave
            self.current_idx = (self.current_idx + 1) % total_keys
            attempts += 1

        # Si llegamos aquí, significa que TODAS las 110 llaves se usaron en los últimos 61 segundos.
        # En lugar de fallar, calculamos cuánto falta para que la llave más antigua se enfríe y esperamos.
        oldest_key = min(self.last_used, key=lambda k: self.last_used[k])
        time_to_wait = self.cooldown - (time.time() - self.last_used[oldest_key])

        if time_to_wait > 0:
            # print(f"   [VROMLIX UTILS] Pausa térmica de seguridad: {time_to_wait:.1f}s...")
            time.sleep(time_to_wait)

        # Tras la pausa, la llave más vieja ya está lista
        self.last_used[oldest_key] = time.time()

        # Sincronizamos el índice para continuar desde la llave que acabamos de entregar
        self.current_idx = (self.keys.index(oldest_key) + 1) % total_keys
        return oldest_key


class VromlixOrchestrator:
    def __init__(self):
        # 1. Tri-State Detection
        self.is_colab = "google.colab" in sys.modules
        self.is_firebase = (
            "FIREBASE_CONFIG" in os.environ
            or "K_SERVICE" in os.environ
            or "FUNCTION_NAME" in os.environ
        )
        self.is_local = (
            not self.is_colab
            and not self.is_firebase
            and sys.platform.startswith("linux")
        )

        # 2. Base Paths
        if self.is_colab:
            self.base_path = Path("/content/drive/MyDrive/VROMLIX_CORE")
        elif self.is_local:
            # Anclaje Absoluto SOTA: Intenta la ruta física, si falla usa detección dinámica (útil en CI)
            prod_base = Path(
                "/media/rogerman/14befb81-4210-4134-a9a0-0ee76166e483/VROMLIX_CORE"
            )
            if prod_base.exists():
                self.base_path = prod_base
            else:
                # Fallback para CI/GitHub Actions: asumimos estructura de repos hermanos
                self.base_path = Path(__file__).resolve().parents[1] / "VROMLIX_CORE"
        else:
            self.base_path = Path("/tmp/VROMLIX_CORE")

        # 3. REGISTRO CENTRAL DE RUTAS
        self.paths = self._init_paths()

        # 4. Config Loading
        self.config_path = self._get_config_path()
        self.config = self._load_config()

        # 5. API Rotation SOTA (NUEVO)
        keys_list = getattr(self.config, "LISTA_DE_APIS", [])
        self.key_manager = ActiveKeyManager(keys_list, cooldown_seconds=61.0)

    def _init_paths(self):
        class Paths:
            base = self.base_path
            sandbox = self.base_path / "00_sandbox"
            codex_memory = self.base_path / "01_codex_memory"
            active_memory = self.base_path / "02_active_memory"
            prompts = self.base_path / "03_prompts"
            scripts = self.base_path / "04_scripts"
            docs = self.base_path / "05_docs"
            vector_db = self.base_path / "06_vector_db"
            raw_knowledge = self.base_path / "99_deep_storage"
            deep_memory = self.base_path / "98_deep_memory_corpus"
            local_llms = Path(
                "/media/rogerman/14befb81-4210-4134-a9a0-0ee76166e483/Local_LLMs"
            )
            if not local_llms.exists():
                local_llms = self.base_path.parent / "Local_LLMs"

            # --- REPOSITORIOS EXTERNOS CENTRALIZADOS (Solo los activos) ---
            repos_externos = [
                self.base_path.parent / "cv",
                self.base_path.parent / "blueprints",
                self.base_path.parent / "rlozano.intel",
                self.base_path.parent / "vromlix-cognitive-architecture",
            ]

        # Asegurar que SANDBOX siempre exista
        Paths.sandbox.mkdir(parents=True, exist_ok=True)
        return Paths()

    def _get_config_path(self):
        if self.is_colab:
            from google.colab import drive

            if not os.path.exists("/content/drive"):
                drive.mount("/content/drive")
            return "/content/drive/MyDrive/Colab Notebooks/config_api_keys_secrets.py"
        elif self.is_local:
            # --- MODIFICADO: Ahora apunta a la carpeta oculta .secrets ---
            return str(self.paths.base / ".secrets" / "config_api_keys_secrets.py")
        return "./.secrets/config_api_keys_secrets.py"

    def _load_config(self):
        if not os.path.exists(self.config_path):
            logging.warning(
                f"AVISO: No se encontró config en {self.config_path}. Usando defaults."
            )
            return None
        spec = importlib.util.spec_from_file_location("config_module", self.config_path)
        if spec is None:
            return None

        if spec.loader is None:
            logging.warning(
                f"AVISO: No se pudo crear el spec de carga para {self.config_path}."
            )
            return None

        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        return config

    def get_model(self, role):
        # Fallback si no hay config pero se pide un rol estándar para evitar crashes
        if not self.config and role.upper() in [
            "VOLUMEN",
            "CONSULTA",
            "PRIME",
            "REASONING",
        ]:
            return "gemini-2.0-flash-exp"

        attr_name = f"MODELO_{role.upper()}"
        model = getattr(self.config, attr_name, None)
        if not model:
            raise ValueError(
                f"HARD STOP: El modelo para el rol '{role}' no está definido."
            )
        return model

    def get_api_key(self):
        """Retorna una llave garantizada como fresca (fuera del cooldown)."""
        return self.key_manager.get_fresh_key()

    def get_secret(self, key_name: str):
        try:
            registry = getattr(self.config, "MODEL_ROUTING_REGISTRY", {})
            if key_name in registry:
                return registry[key_name]

            # --- NUEVA LÍNEA: Buscar como variable global directa ---
            if hasattr(self.config, key_name):
                return getattr(self.config, key_name)

            return getattr(self.config, "SECRETS", {}).get(key_name)
        except Exception:
            return None

    def get_safety_settings(self) -> list[Any]:
        """Devuelve la configuración global de seguridad SOTA para la API v3."""
        try:
            from google.genai import types

            return [
                types.SafetySetting(category=c, threshold="BLOCK_NONE")
                for c in [
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                ]
            ]
        except ImportError:
            return []

    def query_local_ollm(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        """Centraliza las peticiones a la API local de Ollama para cualquier script."""
        import httpx

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            # Timeout de 120s por si el modelo está "frío" y tarda en cargar a RAM
            response = httpx.post(
                "http://localhost:11434/api/chat", json=payload, timeout=1200
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            logging.error(f"Fallo en API local Ollama: {e}")
            return f"ERROR LOCAL: {e}"


# Instancia única global
vromlix = VromlixOrchestrator()


class IOManager:
    @staticmethod
    def select_file(
        provided_path: str | None = None, title: str = "Selecciona un archivo"
    ) -> str | None:
        if provided_path and os.path.exists(provided_path):
            return provided_path
        if vromlix.is_colab:
            from google.colab import files

            print(f"📂 {title}:")
            uploaded = files.upload()
            return list(uploaded.keys())[0] if uploaded else None
        elif vromlix.is_local:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(title=title)
            return file_path if file_path else None
        else:
            file_path = input(f"📂 {title}: ").strip()
            return file_path if os.path.exists(file_path) else None

    @staticmethod
    def select_files(title: str = "Selecciona archivos") -> list[str]:
        if vromlix.is_colab:
            from google.colab import files

            uploaded = files.upload()
            return list(uploaded.keys()) if uploaded else []
        elif vromlix.is_local:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            file_paths = filedialog.askopenfilenames(title=title)
            return list(file_paths) if file_paths else []
        else:
            paths = input(f"📂 {title} (separadas por coma): ")
            return [p.strip() for p in paths.split(",") if os.path.exists(p.strip())]

    @staticmethod
    def export_file(file_path: str):
        if vromlix.is_colab:
            from google.colab import files

            files.download(file_path)
        else:
            print(f"✅ Archivo guardado en: {os.path.abspath(file_path)}")

    @staticmethod
    def select_directory(title: str = "Selecciona un directorio") -> str | None:
        if vromlix.is_colab:
            dir_path = input(f"📂 {title} en Colab: ").strip()
            return dir_path if os.path.exists(dir_path) else None
        elif vromlix.is_local:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            dir_path = filedialog.askdirectory(title=title)
            return dir_path if dir_path else None
        else:
            dir_path = input(f"📂 {title}: ").strip()
            return dir_path if os.path.exists(dir_path) else None
