# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-genai",
#     "sqlite-vec",
#     "ddgs",
#     "feedparser>=6.0.12",
#     "requests>=2.32.5",
#     "pydantic>=2.12.5"
# ]
# ///
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @description Vromlix Prime V2.0: Infraestructura, Memoria a Corto Plazo y Monitor de Tokens.

import os
import sys
import readline  # noqa: F401
import shutil
import logging
import json
import time
import sqlite3
import threading
import re

try:
    import sqlite_vec

    HAS_SQLITE_VEC = True
except ImportError:
    HAS_SQLITE_VEC = False
    logging.warning(
        "⚠️ sqlite-vec no está instalado. La memoria profunda (RAG) estará desactivada."
    )
import concurrent.futures
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

from google import genai
from google.genai import types

# --- 1. VROMLIX CENTRAL BRAIN ---
from vromlix_utils import vromlix, OSINTGrounder


class TokenMonitor:
    """
    Rastrea el consumo de tokens de la sesión actual por experto/API.
    """

    def __init__(self):
        self.expert_usage: dict[str, dict[str, int]] = {}

    def add_usage(self, expert_id: str, usage_metadata: Any) -> None:
        """Suma los tokens asociados a un experto específico."""
        if not usage_metadata:
            return
        if expert_id not in self.expert_usage:
            self.expert_usage[expert_id] = {"in": 0, "out": 0}

        self.expert_usage[expert_id]["in"] += getattr(
            usage_metadata, "prompt_token_count", 0
        )
        self.expert_usage[expert_id]["out"] += getattr(
            usage_metadata, "candidates_token_count", 0
        )

    def get_summary(self) -> str:
        """Retorna un desglose limpio por experto."""
        if not self.expert_usage:
            return "🪙 Tokens: 0"

        lines = ["📊 Desglose de Tokens (Sesión):"]
        total_in, total_out = 0, 0
        for exp, data in self.expert_usage.items():
            lines.append(f"   ├─ [{exp}]: {data['in']} In | {data['out']} Out")
            total_in += data["in"]
            total_out += data["out"]
        lines.append(f"   └─ TOTAL: {total_in} In | {total_out} Out")
        return "\n".join(lines)


class VromlixContextLoader:
    """
    Carga y fusiona los archivos de configuración inmutables (XML/JSON).
    Mantiene el Repositorio de Código en RAM para inyección bajo demanda (Lazy Loading).
    """

    def __init__(self):
        self.base_path: Path = vromlix.paths.base
        self.codex_path: Path = vromlix.paths.codex_memory

        self.logic_file: Path = self._find_file("00_System_Operating_Logic.xml")
        self.profile_file: Path = self._find_file("01_Dynamic_Profile.xml")
        self.moe_file: Path = self._find_file("02_MoE_Routing.json")
        self.repo_file: Path = self._find_file("12_Code_Repository.md")
        self.prompts_file: Path = self._find_file("05_Orchestrator_Prompts.xml")

    def load_system_prompts(self) -> dict:
        """Carga los prompts del orquestador desde el archivo XML."""
        prompts: dict[str, str] = {}
        if not self.prompts_file.exists():
            logging.error(f"CRITICAL: {self.prompts_file.name} no encontrado.")
            return prompts
        try:
            tree = ET.parse(self.prompts_file)
            root = tree.getroot()
            for prompt_elem in root.findall("prompt"):
                p_id = prompt_elem.get("id")
                p_text = prompt_elem.text
                if p_id and p_text:
                    prompts[p_id] = p_text.strip()
        except Exception as e:
            logging.error(f"Error parseando prompts XML: {e}")
        return prompts

    def _find_file(self, filename: str) -> Path:
        """Busca el archivo en la raíz y en codex_memory."""
        if (self.base_path / filename).exists():
            return self.base_path / filename
        if (self.codex_path / filename).exists():
            return self.codex_path / filename
        return self.base_path / filename

    def _read_file(self, filepath: Path) -> str:
        """Lee un archivo de texto de forma segura."""
        if not filepath.exists():
            logging.error(f"CRITICAL: Archivo core no encontrado -> {filepath.name}")
            return f"<!-- ERROR: {filepath.name} MISSING -->"

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logging.error(f"Error leyendo {filepath.name}: {e}")
            return ""

    def build_master_system_prompt(self) -> str:
        """
        Ensambla la arquitectura cognitiva inyectando los archivos directamente
        en las instrucciones del sistema.
        """
        logging.info("🧠 Ensamblando Master System Prompt (Kernel + Profile + MoE)...")

        logic_content = self._read_file(self.logic_file)
        profile_content = self._read_file(self.profile_file)
        moe_content = self._read_file(self.moe_file)

        master_prompt = f"""
You are VROMLIX PRIME, the Polymatic Operating System Orchestrator.
You operate strictly under the architectural definitions provided below.
Your cognitive state is externalized in these documents. Do not hallucinate features.

=== 1. SYSTEM OPERATING LOGIC (KERNEL) ===
{logic_content}

=== 2. DYNAMIC PROFILE (THE SOUL) ===
{profile_content}

=== 3. MIXTURE OF EXPERTS (MoE) ROUTING ===
{moe_content}

=== ORCHESTRATOR DIRECTIVES ===
1. Analyze the user's input and the recent conversation history.
2. Adopt the persona, mechanics, and constraints of the assigned Expert(s).
3. ALWAYS append the VROMLIX_STATE_TRACKER at the end of your response.

=== FILE PATCHING PROTOCOL ===
You are a CONSULTATIVE Senior Architect. DO NOT generate code patches proactively.
1. First, analyze the user's request and provide your findings, analysis, or theoretical solution.
2. End your response by ASKING the user: "¿Deseas que genere el parche de código para aplicar estos cambios en [nombre_del_archivo]?"
3. ONLY if the user explicitly replies with a "Yes" or gives a direct command to patch, you MUST use the following exact format to apply surgical patches. DO NOT rewrite the entire file:
📄 File: [filename.ext]
<<<< SEARCH
[Exact lines to find and replace. Must match the original file perfectly]
====
[New lines to insert]
>>>> REPLACE
"""
        return master_prompt.strip()


class SessionTracker:
    """
    Gestiona la memoria a corto plazo (Short-Term Memory) de la sesión actual.
    Escribe los logs en Markdown y lee los últimos turnos para evitar amnesia.
    """

    def __init__(self):
        self.logs_dir: Path = vromlix.paths.active_memory / "sessions"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file: Path = self.logs_dir / f"session_{self.session_id}.md"
        self._init_log()

    def _init_log(self) -> None:
        """Inicializa el archivo Markdown de la sesión."""
        try:
            with open(self.session_file, "w", encoding="utf-8") as f:
                f.write("# 🧠 VROMLIX PRIME: SESSION LOG\n")
                f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Session ID:** {self.session_id}\n\n---\n\n")
            logging.info(f"📝 Sesión iniciada: {self.session_file.name}")
        except Exception as e:
            logging.error(f"Fallo al inicializar el log de sesión: {e}")

    def log_interaction(self, role: str, content: str) -> None:
        """Registra un turno de la conversación (Usuario o Vromlix)."""
        try:
            with open(self.session_file, "a", encoding="utf-8") as f:
                f.write(f"### {role.upper()}\n{content}\n\n")
        except Exception as e:
            logging.error(f"Fallo al escribir en el log: {e}")

    def get_recent_context(self, max_turns: int = 5) -> str:
        """
        Lee el archivo de sesión y extrae los últimos 'max_turns' (pares de Q/A)
        para inyectarlos como Memoria a Corto Plazo.
        """
        if not self.session_file.exists():
            return ""

        try:
            with open(self.session_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Dividir el documento por los encabezados de rol
            blocks = re.split(r"(?=### USER|### VROMLIX)", content)

            # Filtrar solo los bloques de interacción (ignorando el header del archivo)
            interactions = [
                b.strip()
                for b in blocks
                if b.startswith("### USER") or b.startswith("### VROMLIX")
            ]

            if not interactions:
                return ""

            # Tomar los últimos N bloques (max_turns * 2 porque es pregunta y respuesta)
            recent_blocks = interactions[-(max_turns * 2) :]

            context_str = "=== RECENT CONVERSATION HISTORY (SHORT-TERM MEMORY) ===\n"
            context_str += "\n\n".join(recent_blocks)
            return context_str

        except Exception as e:
            logging.error(f"Fallo al leer el contexto reciente: {e}")
            return ""

    def append_state_tracker(
        self, focus: str, locked: str, stack: str, friction: str, loop: str
    ) -> str:
        """
        Genera y registra el bloque de estado determinista requerido por la arquitectura.
        """
        tracker = f"""
::: VROMLIX_STATE_TRACKER :::
[FOCUS]::{focus}
[LOCKED]::{locked}
[STACK]::{stack}
[FRICTION]::{friction}
[LOOP]::{loop}
::: END_TRACKER :::
"""
        try:
            with open(self.session_file, "a", encoding="utf-8") as f:
                f.write(tracker.strip() + "\n\n---\n\n")
        except Exception as e:
            logging.error(f"Fallo al escribir el tracker: {e}")

        return tracker.strip()


# --- ESQUEMAS PYDANTIC (DAG) ---
class ExecutionStep(BaseModel):
    step_id: str = Field(description="Unique ID for this step, e.g., 'step_1'")
    expert_id: str = Field(description="ID of the selected expert")
    required_files: list[str] = Field(
        default_factory=list, description="Exact filenames required by this expert"
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="List of step_ids that must finish before this step starts",
    )


class SimulatedPath(BaseModel):
    path_logic: str = Field(description="Description of this potential routing path")
    success_probability: float = Field(
        description="Estimated probability of success (0.0 to 1.0)"
    )


class RoutingResult(BaseModel):
    mcts_simulations: list[SimulatedPath] = Field(
        description="Simulate at least 2 different routing paths before deciding"
    )
    internal_analysis: str = Field(
        description="Logic on why the winning DAG structure was chosen over the alternatives"
    )
    execution_plan: list[ExecutionStep] = Field(
        description="List of execution steps forming the winning Directed Acyclic Graph"
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="Array of search queries, empty if none needed",
    )


# --- 2. COGNITIVE PIPELINE & MoE ROUTING (V2.0) ---
class MoERouter:
    """
    Enrutador Semántico Avanzado (Mixture of Experts).
    Analiza el input y selecciona MÚLTIPLES expertos, determinando si se requiere
    búsqueda web (Grounding) o acceso al repositorio de código.
    Implementa Pasarela de Enrutamiento Semántico (Zero-LLM) para consultas directas.
    """

    def __init__(
        self, moe_json_content: str, monitor: TokenMonitor, router_prompt: str
    ):
        self.model_id = vromlix.get_model("VOLUMEN")  # Modelo rápido para triaje
        self.embed_model = (
            vromlix.get_secret("EMBEDDINGS")["model_id"]
            if vromlix.get_secret("EMBEDDINGS")
            else "gemini-embedding-2-preview"
        )
        self.monitor = monitor
        self.router_prompt = router_prompt
        self.cache_path = vromlix.paths.active_memory / "expert_embeddings_cache.json"
        try:
            self.moe_data = json.loads(moe_json_content)
        except json.JSONDecodeError:
            logging.error("CRITICAL: Fallo al parsear 02_MoE_Routing.json")
            self.moe_data = []

        # Cargar o generar embeddings de los expertos en background (Lazy)
        self.expert_vectors = self._load_expert_vectors()

    def _cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        import math

        dot_product = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(b * b for b in v2))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot_product / (mag1 * mag2)

    def _load_expert_vectors(self) -> dict[str, list[float]]:
        """Carga los vectores cacheados o los genera si el MoE cambió."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                    # Validar si la cantidad de expertos coincide
                    if len(cache) == len(self.moe_data):
                        return cache
            except Exception:
                pass

        # Si no hay caché válido, generamos los embeddings
        logging.info(
            "🧠 [Semantic Router] Generando embeddings de expertos (Cache Miss)..."
        )
        vectors = {}
        try:
            client = genai.Client(api_key=vromlix.get_api_key())
            for exp in self.moe_data:
                # Crear un documento semántico denso para el experto
                doc = f"Role: {exp['expert_id']}. Mechanics: {', '.join(exp.get('mechanics', []))}. Instructions: {', '.join(exp.get('instructions', []))}."
                response = client.models.embed_content(
                    model=self.embed_model,
                    contents=doc,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768
                    ),
                )
                vectors[exp["expert_id"]] = response.embeddings[0].values

            # Guardar caché
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(vectors, f)
        except Exception as e:
            logging.warning(f"⚠️ Fallo al generar vectores de expertos: {e}")
        return vectors

    def determine_routing(self, user_query: str, recent_context: str) -> dict:
        """
        Devuelve un diccionario con los expertos seleccionados y las banderas de contexto.
        Utiliza Pasarela Semántica (Zero-LLM) primero, y Pydantic como Fallback.
        """
        if not self.moe_data:
            return {
                "execution_plan": [
                    {
                        "step_id": "step_1",
                        "expert_id": "ORCHESTRATE_SYSTEM_CORE",
                        "instructions": ["Fallback mode."],
                        "dynamic_files": [],
                        "depends_on": [],
                    }
                ],
                "search_queries": [],
            }

        # --- 1. PASARELA SEMÁNTICA (ZERO-LLM ROUTING) ---
        # Si no hay contexto reciente complejo, intentamos enrutamiento matemático directo
        if self.expert_vectors and len(recent_context) < 500:
            try:
                client = genai.Client(api_key=vromlix.get_api_key())
                response = client.models.embed_content(
                    model=self.embed_model,
                    contents=user_query,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_QUERY", output_dimensionality=768
                    ),
                )
                query_vector = response.embeddings[0].values

                best_expert = None
                max_sim = -1.0

                for exp_id, exp_vector in self.expert_vectors.items():
                    sim = self._cosine_similarity(query_vector, exp_vector)
                    if sim > max_sim:
                        max_sim = sim
                        best_expert = exp_id

                # Umbral de confianza estricto (0.75 suele ser alto en embeddings de Gemini)
                if max_sim > 0.75 and best_expert:
                    logging.info(
                        f"⚡ [Semantic Router] Cortocircuito activado: {best_expert} (Similitud: {max_sim:.2f})"
                    )
                    matched_profile: dict[str, Any] = next(
                        (
                            exp
                            for exp in self.moe_data
                            if exp["expert_id"] == best_expert
                        ),
                        {},
                    )
                    step_dict = dict(matched_profile)
                    step_dict["step_id"] = "step_1"
                    step_dict["dynamic_files"] = []
                    step_dict["depends_on"] = []
                    return {"execution_plan": [step_dict], "search_queries": []}
            except Exception as e:
                logging.warning(
                    f"⚠️ Fallo en Pasarela Semántica, usando LLM Fallback: {e}"
                )

        # --- 2. ENRUTAMIENTO LLM (PYDANTIC FALLBACK) ---
        routing_map = [
            {"id": exp["expert_id"], "cluster": exp["parent_cluster"]}
            for exp in self.moe_data
        ]

        current_date = datetime.now().strftime("%Y-%m-%d")
        current_year = datetime.now().strftime("%Y")

        prompt = self.router_prompt.format(
            recent_context=recent_context,
            user_query=user_query,
            routing_map=json.dumps(routing_map, indent=2),
            current_date=current_date,
            current_year=current_year,
        )

        try:
            client = genai.Client(api_key=vromlix.get_api_key())
            response = client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=RoutingResult,  # <--- LA MAGIA DE PYDANTIC AQUI
                ),
            )
            self.monitor.add_usage("MoERouter", response.usage_metadata)

            # La API de Google GenAI (>=1.0) ya valida y parsea el esquema en .parsed
            if not response.parsed:
                raise ValueError("MoE Router: El modelo no devolvió una respuesta estructurada válida.")

            result: RoutingResult = response.parsed

            # Imprimir el razonamiento interno del Router (CoT) en la consola
            logging.info(f"🧠 [Router CoT]: {result.internal_analysis}")

            # Extraer y mapear el Plan de Ejecución (DAG)
            execution_plan = []
            for step in result.execution_plan[:3]:  # Límite de 3 pasos
                profile: dict[str, Any] = next(
                    (
                        exp
                        for exp in self.moe_data
                        if exp["expert_id"] == step.expert_id
                    ),
                    {},
                )
                if not profile:
                    profile = next(
                        (
                            exp
                            for exp in self.moe_data
                            if exp["expert_id"] == "ORCHESTRATE_SYSTEM_CORE"
                        ),
                        {},
                    )

                step_dict = dict(profile or {})
                step_dict["step_id"] = step.step_id
                step_dict["dynamic_files"] = step.required_files
                step_dict["depends_on"] = step.depends_on
                execution_plan.append(step_dict)

            logging.info(
                f"🔀 MoE Router: DAG Planificado -> {[s['step_id'] + '(' + s['expert_id'] + ')' for s in execution_plan]}"
            )

            if result.search_queries:
                logging.info(
                    f"🌐 MoE Router: Búsqueda Web solicitada -> {result.search_queries}"
                )

            return {
                "execution_plan": execution_plan,
                "search_queries": result.search_queries,
            }

        except Exception as e:
            logging.warning(f"Fallo en MoE Routing ({e}). Usando Fallback.")
            fallback_profile: dict[str, Any] = next(
                (
                    exp
                    for exp in self.moe_data
                    if exp["expert_id"] == "ORCHESTRATE_SYSTEM_CORE"
                ),
                {},
            )
            fallback_profile["step_id"] = "step_1"
            fallback_profile["dynamic_files"] = []
            fallback_profile["depends_on"] = []
            return {
                "execution_plan": [fallback_profile],
                "search_queries": [],
            }


# --- HERRAMIENTAS LOCALES (MCP / FUNCTION CALLING) ---
def leer_lineas_de_archivo(
    filepath: str, linea_inicio: int = 1, linea_fin: int = 100
) -> str:
    """
    Lee un rango específico de líneas de un archivo local para no saturar la memoria.
    Útil para inspeccionar repositorios de código o documentos grandes por partes.

    Args:
        filepath: Ruta del archivo (ej. 'vromlix_utils.py' o 'docs/readme.md').
        linea_inicio: Número de línea donde empezar a leer (1-indexado). Default 1.
        linea_fin: Número de línea donde terminar. Default 100.
    """
    try:
        from pathlib import Path

        # Resolver ruta relativa a la base de Vromlix por seguridad
        target_path = Path(vromlix.paths.base) / filepath

        if not target_path.exists() or not target_path.is_file():
            # Buscar en codex_memory como fallback
            target_path = Path(vromlix.paths.codex_memory) / filepath
            if not target_path.exists():
                return f"[ERROR TOOL]: El archivo {filepath} no existe."

        with open(target_path, "r", encoding="utf-8") as f:
            lineas = f.readlines()

        total_lineas = len(lineas)
        inicio_idx = max(0, linea_inicio - 1)
        fin_idx = min(total_lineas, linea_fin)

        if inicio_idx >= total_lineas:
            return f"[ERROR TOOL]: linea_inicio ({linea_inicio}) es mayor que el total de líneas ({total_lineas})."

        fragmento = "".join(lineas[inicio_idx:fin_idx])
        return f"--- ARCHIVO: {filepath} (Líneas {linea_inicio} a {fin_idx} de {total_lineas}) ---\n{fragmento}"

    except Exception as e:
        return f"[ERROR TOOL]: Fallo al leer {filepath} -> {str(e)}"


class AgenticExecutor:
    """
    Motor de Ejecución Multi-Agente.
    Ejecuta múltiples expertos en paralelo (Threading) y maneja la inyección
    dinámica de herramientas (Google Search) y contexto (Code Repo).
    """

    def __init__(
        self,
        master_prompt: str,
        tracker: SessionTracker,
        monitor: TokenMonitor,
        repo_file: Path,
    ):
        self.master_prompt = master_prompt
        self.tracker = tracker
        self.monitor = monitor
        self.repo_file = repo_file
        self.model_id = vromlix.get_model("PRECISION")
        self.safety_settings = vromlix.get_safety_settings()

    def _execute_single_expert(
        self,
        expert_profile: dict,
        user_query: str,
        recent_context: str,
        web_context: str,
        retrieved_rag: str,
    ) -> dict:
        """Ejecuta un solo experto. Diseñado para correr en un hilo paralelo."""
        expert_id = expert_profile.get("expert_id", "UNKNOWN")
        dynamic_files = expert_profile.get("dynamic_files", [])
        if dynamic_files:
            logging.info(f"📂 Inyectando a [{expert_id}]: {dynamic_files}")

        # 1. Construir el System Instruction Dinámico
        expert_context = f"""
        === ACTIVE EXPERT OVERRIDE ===
        You are currently operating as: {expert_id}
        Mechanics: {", ".join(expert_profile.get("mechanics", []))}
        Constraints: {", ".join(expert_profile.get("constraints", []))}
        Instructions: {", ".join(expert_profile.get("instructions", []))}
        Output Signature: {expert_profile.get("output_signature", "")}
        """

        if web_context:
            current_date = datetime.now().strftime("%Y-%m-%d")
            expert_context += f"""
            \nCRITICAL DIRECTIVE - TEMPORAL ANCHOR & ANTI-HALLUCINATION GUARDRAIL:
            1. The absolute current date is {current_date}.
            2. You have been provided with a LIVE WEB GROUNDING CONTEXT. You MUST prioritize this fresh information.
            3. Discard outdated internal knowledge if newer developments exist in the web context.
            4. Cross-reference web results with the provided RAG memory.
            5. STRICT PROHIBITION: DO NOT extrapolate, guess, or invent version numbers. If a specific fact is not EXPLICITLY confirmed by the web context, DO NOT mention it.
            """

        final_system_instruction = self.master_prompt + "\n" + expert_context
        # 2. Ensamblar el Prompt del Usuario con Contextos Dinámicos
        full_user_prompt = f"{recent_context}\n\n"

        if web_context:
            full_user_prompt += f"=== LIVE WEB GROUNDING CONTEXT (DEEP RESEARCH) ===\n{web_context}\n==================================================\n\n"

        if retrieved_rag:
            full_user_prompt += f"=== DEEP MEMORY CONTEXT ===\n{retrieved_rag}\n===========================\n\n"

        # --- DIAGNÓSTICO DE TOKENS (RAYOS X) ---
        len_master = len(final_system_instruction)
        len_web = len(web_context) if web_context else 0
        len_rag = len(retrieved_rag) if retrieved_rag else 0
        logging.info(
            f"🔍 [Rayos X - {expert_id}] Chars -> Master: {len_master} | Web: {len_web} | RAG: {len_rag}"
        )

        # --- REFERENCIA DINÁMICA DE ARCHIVOS (MCP SOTA) ---
        if dynamic_files:
            full_user_prompt += "=== ARCHIVOS RELEVANTES IDENTIFICADOS ===\n"
            full_user_prompt += "El sistema ha detectado que los siguientes archivos están relacionados con tu tarea.\n"
            full_user_prompt += "NO intentes adivinar su contenido. DEBES usar explícitamente tu herramienta 'leer_lineas_de_archivo' para inspeccionarlos si lo requieres.\n"
            for fname in dynamic_files:
                full_user_prompt += f"- {fname}\n"
            full_user_prompt += "=========================================\n\n"

        full_user_prompt += f"USER QUERY:\n{user_query}"

        # 3. Configurar Herramientas (Function Calling / MCP Local)
        tools = [leer_lineas_de_archivo]

        # --- 3.5 EJECUCIÓN LOCAL OFFLINE (VÍA API OLLAMA EN UTILS) ---
        if "LOCAL_" in expert_id:
            logging.info(
                f"🔋 [Local Engine] Solicitando inferencia a Ollama para {expert_id}..."
            )

            # 1. Obtenemos el nombre dinámico desde config_api_keys_secrets.py
            modelo_local = vromlix.get_model("LOCAL")

            # 2. Llamada limpia al orquestador central (vromlix_utils)
            respuesta_local = vromlix.query_local_ollm(
                model_name=modelo_local,
                system_prompt=final_system_instruction,
                user_prompt=full_user_prompt,
            )

            return {"expert_id": expert_id, "response": respuesta_local}

        # 4. Ejecución con Rotación Total y Thinking (Nube)
        max_attempts = (
            len(vromlix.key_manager.keys) if hasattr(vromlix, "key_manager") else 110
        )
        for attempt in range(max_attempts):
            api_key = vromlix.get_api_key()
            if not api_key:
                return {"expert_id": expert_id, "response": "ERROR: No API Keys."}

            # Aplicamos el mismo escudo anti-congelamiento a los Expertos del Enjambre
            client = genai.Client(
                api_key=api_key, http_options=types.HttpOptions(api_version="v1alpha")
            )
            try:
                response = client.models.generate_content(
                    model=self.model_id,
                    contents=full_user_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        system_instruction=final_system_instruction,
                        safety_settings=self.safety_settings,
                        tools=tools if tools else None,
                        thinking_config=types.ThinkingConfig(include_thoughts=True),
                    ),
                )
                self.monitor.add_usage(expert_id, response.usage_metadata)

                # Extraer pensamientos y respuesta
                pensamientos = ""
                respuesta_final = ""
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if getattr(part, "thought", False):
                            pensamientos += (
                                f"\n[🧠 RAZONAMIENTO DEL EXPERTO]:\n{part.text}\n"
                            )
                        elif part.text:
                            respuesta_final += part.text
                else:
                    respuesta_final = response.text

                # --- FASE 3: BUCLE DE REFLEXIÓN (AUTO-CORRECCIÓN CÍCLICA) ---
                expected_sig = expert_profile.get("output_signature", "")
                if expected_sig:
                    # Extraer el marcador principal de la firma esperada
                    sig_marker = ""
                    if "```python" in expected_sig:
                        sig_marker = "```python"
                    elif "```bash" in expected_sig:
                        sig_marker = "```bash"
                    elif "###" in expected_sig:
                        sig_marker = expected_sig.split("\n")[0].strip()

                    # Si el marcador no está en la respuesta, forzamos un reintento (Ciclo)
                    if sig_marker and sig_marker not in respuesta_final:
                        if attempt < max_attempts - 1:
                            logging.warning(
                                f"🔄 [Reflexión] {expert_id} omitió la firma '{sig_marker}'. Forzando auto-corrección interna..."
                            )
                            # Inyectamos el regaño al final del prompt y el bucle 'for' vuelve a girar
                            full_user_prompt += f"\n\n[SYSTEM FEEDBACK - ATTEMPT {attempt + 1} FAILED]: You hallucinated or ignored the constraints. You MUST include the exact signature '{sig_marker}' in your response. Correct your output and try again."
                            continue
                        else:
                            logging.error(
                                f"❌ [Reflexión] {expert_id} falló la auto-corrección tras múltiples intentos."
                            )
                # ------------------------------------------------------------

                output_completo = (
                    f"{pensamientos}\n{respuesta_final}"
                    if pensamientos
                    else respuesta_final
                )
                return {"expert_id": expert_id, "response": output_completo.strip()}
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "503" in error_str or "quota" in error_str:
                    logging.warning(
                        f"⚠️ API Limit ({expert_id}). Descartando llave... ({attempt + 1}/{max_attempts})"
                    )
                    time.sleep(
                        0.5
                    )  # Micro-pausa para evitar penalización por DDoS en la IP
                    continue
                else:
                    logging.error(f"❌ Error en experto {expert_id}: {e}")
                    return {"expert_id": expert_id, "response": f"ERROR INTERNO: {e}"}

        return {"expert_id": expert_id, "response": "ERROR: Timeout por cuota."}

    def process_swarm(
        self,
        user_query: str,
        routing_data: dict,
        recent_context: str,
        retrieved_rag: str = "",
        web_context: str = "",
    ) -> dict[str, str]:
        """
        Orquesta la ejecución basada en un Grafo Acíclico Dirigido (DAG).
        Si no hay dependencias, ejecuta en paralelo (Colmena).
        Si hay dependencias, encola y hereda el contexto.
        """
        self.tracker.log_interaction("User", user_query)

        execution_plan = routing_data.get("execution_plan", [])
        if not execution_plan:
            return {}

        logging.info(
            f"🧠 Ejecutando Motor DAG ({len(execution_plan)} pasos en la cola)..."
        )

        pending_steps = {step["step_id"]: step for step in execution_plan}
        completed_responses: dict[str, str] = {}  # step_id -> response
        in_progress: dict[concurrent.futures.Future, str] = {}  # Future -> step_id

        # Motor de resolución de dependencias
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(execution_plan)
        ) as executor:
            while pending_steps or in_progress:
                # 1. Identificar pasos listos para ejecutarse (cuyas dependencias ya terminaron)
                ready_steps = []
                for step_id, step in list(pending_steps.items()):
                    if all(
                        dep in completed_responses for dep in step.get("depends_on", [])
                    ):
                        ready_steps.append(step_id)

                # 2. Despachar pasos listos a los hilos
                for step_id in ready_steps:
                    step = pending_steps.pop(step_id)

                    # Preparar la inyección del contexto heredado si hubo dependencias
                    inherited_context = ""
                    for dep in step.get("depends_on", []):
                        inherited_context += f"--- OUTPUT HEREDADO DEL PASO: {dep} ---\n{completed_responses[dep]}\n\n"

                    augmented_query = user_query
                    if inherited_context:
                        augmented_query = f"=== CONTEXTO DE EXPERTOS PREVIOS (DEPENDENCIAS) ===\n{inherited_context}\nUSER QUERY ORIGINAL:\n{user_query}"

                    # Enviar a ejecución
                    future = executor.submit(
                        self._execute_single_expert,
                        step,
                        augmented_query,
                        recent_context,
                        web_context,
                        retrieved_rag,
                    )
                    in_progress[future] = step_id

                # 3. Esperar a que al menos UN hilo termine antes de continuar el bucle
                if in_progress:
                    done, _ = concurrent.futures.wait(
                        in_progress.keys(),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )

                    for future in done:
                        step_id = in_progress.pop(future)
                        try:
                            result = future.result()
                            # Guardar la respuesta etiquetada para que Ockham sepa quién dijo qué
                            completed_responses[step_id] = (
                                f"[{result['expert_id']}]:\n{result['response']}"
                            )
                        except Exception as e:
                            logging.error(f"Error en paso DAG {step_id}: {e}")
                            completed_responses[step_id] = (
                                f"[{step.get('expert_id', 'UNKNOWN')} ERROR]: {e}"
                            )

        return completed_responses


# --- 3. OCKHAM SYNTHESIS & SANDBOX FIREWALL (V2.0) ---


class OckhamSynthesizer:
    """
    Sintetizador Maestro.
    Fusiona las respuestas de múltiples expertos en una sola narrativa coherente
    y aplica una auditoría estricta (Red Team) antes de la salida final.
    """

    def __init__(
        self,
        master_prompt: str,
        monitor: TokenMonitor,
        fusion_prompt: str,
        auditor_prompt: str,
    ):
        self.master_prompt = master_prompt
        self.monitor = monitor
        self.fusion_prompt = fusion_prompt
        self.auditor_prompt = auditor_prompt
        self.model_id = vromlix.get_model("PRECISION")
        self.safety_settings = vromlix.get_safety_settings()

    def _call_llm(self, prompt: str, temp: float) -> str:
        max_attempts = (
            len(vromlix.key_manager.keys) if hasattr(vromlix, "key_manager") else 110
        )
        for attempt in range(max_attempts):
            api_key = vromlix.get_api_key()
            if not api_key:
                return "ERROR: No API Keys."

            # Configurar cliente para fallar rápido y evitar que Google congele el hilo
            client = genai.Client(
                api_key=api_key, http_options=types.HttpOptions(api_version="v1alpha")
            )
            try:
                response = client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temp, safety_settings=self.safety_settings
                    ),
                )
                self.monitor.add_usage("Ockham_Synthesizer", response.usage_metadata)
                return response.text
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "503" in error_str or "quota" in error_str:
                    logging.warning(
                        f"⚠️ API Limit (Ockham). Rotando llave... ({attempt + 1}/{max_attempts})"
                    )
                    time.sleep(1.0)
                    continue
                logging.error(f"Error en Synthesizer: {e}")
                return f"ERROR INTERNO: {e}"
        return "ERROR: Timeout en Synthesizer."

    def synthesize(
        self, user_query: str, swarm_responses: dict[str, str], routing_data: dict
    ) -> str:
        """
        Ejecuta la fusión (si hay múltiples expertos) y la auditoría final.
        """
        # 1. FASE DE FUSIÓN (Solo si hay > 1 experto)
        if len(swarm_responses) > 1:
            logging.info(
                f"🔬 [Ockham] Fusionando {len(swarm_responses)} perspectivas..."
            )

            raw_inputs = ""
            for exp_id, resp in swarm_responses.items():
                raw_inputs += f"\n--- PERSPECTIVA DE {exp_id} ---\n{resp}\n"

            current_date = datetime.now().strftime("%Y-%m-%d")
            fusion_prompt = self.fusion_prompt.format(
                user_query=user_query, raw_inputs=raw_inputs
            )
            fusion_prompt = (
                f"[SYSTEM ANCHOR: The absolute current date is {current_date}. STRICT ANTI-HALLUCINATION: You are the final auditor. You must ruthlessly eliminate any mention of unverified model versions, speculative features, or hallucinated APIs from the experts' raw inputs. Only preserve explicitly verifiable facts.]\n\n"
                + fusion_prompt
            )
            draft_response = self._call_llm(fusion_prompt, temp=0.2)
        else:
            # Si solo hay 1 experto, el borrador es su respuesta directa
            draft_response = list(swarm_responses.values())[0]

        if "ERROR" in draft_response:
            return draft_response

        # 2. FASE DE AUDITORÍA (Red Team & Tracker Enforcement)
        logging.info("   -> [Auditor] Evaluando integridad y forzando State Tracker...")

        # Extraer todas las restricciones de los expertos involucrados
        experts = routing_data.get("execution_plan", [])
        all_constraints = []
        for exp in experts:
            all_constraints.extend(exp.get("constraints", []))
        constraints_str = "\n".join([f"- {c}" for c in set(all_constraints)])

        auditor_prompt = self.auditor_prompt.format(
            constraints_str=constraints_str, draft_response=draft_response
        )

        final_response = self._call_llm(auditor_prompt, temp=0.0)
        return final_response


class SandboxFirewall:
    """
    Intercepta código Python y comandos de sistema operativo generados por el LLM.
    Aplica el protocolo Human-in-the-Loop (HitL) para guardar archivos de forma segura.
    """

    def __init__(self):
        self.sandbox_dir: Path = vromlix.paths.sandbox
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

    def _hitl_prompt(self, message: str) -> bool:
        """Muestra un prompt de seguridad en la terminal."""
        print("\n" + "🛡️" * 20)
        print(" 🛑 FIREWALL: INTERVENCIÓN REQUERIDA")
        print("🛡️" * 20)
        print(f" 🔹 {message}")
        print("-" * 40)

        while True:
            choice = input("❓ ¿Autorizas esta acción? [Y/n]: ").strip().lower()
            if choice in ["y", "yes", ""]:
                return True
            elif choice in ["n", "no"]:
                return False

    def execute_if_present(self, llm_response: str) -> str:
        """
        Escanea la respuesta en busca de comandos de sistema operativo.
        (La detección interactiva de Python ha sido delegada a La Forja).
        """
        logs = []

        # 1. Detección de Código Python (DESACTIVADA)
        # Dejamos que el LLM lo envíe a La Forja a través de VROMLIX_MISSIONS

        # 2. Detección de OS Action (JSON Legacy)
        match = re.search(
            r'```(?:json)?\s*(\{.*?"vromlix_os_action".*?\})\s*```',
            llm_response,
            re.DOTALL,
        )
        if not match:
            match = re.search(
                r'(\{.*?"vromlix_os_action".*?\})', llm_response, re.DOTALL
            )

        if match:
            try:
                plan = json.loads(match.group(1)).get("vromlix_os_action", {})
                action = str(plan.get("action", ""))
                target_str = str(plan.get("target_path", ""))
                source_str = (
                    str(plan.get("source_path", "")) if plan.get("source_path") else ""
                )
                content = str(plan.get("content", ""))

                if self._hitl_prompt(
                    f"El agente solicita ejecutar una acción de sistema: {action.upper()} en {target_str}"
                ):
                    try:
                        # Resolver rutas de forma segura dentro del SANDBOX permitiendo subcarpetas
                        target_path = (self.sandbox_dir / target_str).resolve()
                        if not target_path.is_relative_to(self.sandbox_dir):
                            raise ValueError(
                                "Path Traversal detectado. Acceso denegado."
                            )

                        source_path = None
                        if source_str:
                            source_path = (self.sandbox_dir / source_str).resolve()
                            if not source_path.is_relative_to(self.sandbox_dir):
                                raise ValueError(
                                    "Path Traversal detectado en origen. Acceso denegado."
                                )

                        # Crear subdirectorios si no existen
                        target_path.parent.mkdir(parents=True, exist_ok=True)

                        if action == "create_file":
                            with open(target_path, "w", encoding="utf-8") as f:
                                f.write(content)
                            msg = f"File created successfully at {target_path}"
                            print(f"   ✅ {msg}")
                            logs.append(msg)
                        elif action == "delete_file":
                            if target_path.exists():
                                target_path.unlink()
                                msg = f"File deleted successfully: {target_path}"
                                print(f"   ✅ {msg}")
                                logs.append(msg)
                            else:
                                msg = f"File not found for deletion: {target_path}"
                                print(f"   ⚠️ {msg}")
                                logs.append(msg)
                        elif action == "move_file":
                            if source_path and source_path.exists():
                                shutil.move(str(source_path), str(target_path))
                                msg = f"File moved from {source_path.name} to {target_path.name}"
                                print(f"   ✅ {msg}")
                                logs.append(msg)
                            else:
                                msg = "Source file not found for move operation."
                                print(f"   ⚠️ {msg}")
                                logs.append(msg)
                        else:
                            msg = f"Unknown OS action requested: {action}"
                            print(f"   ⚠️ {msg}")
                            logs.append(msg)
                    except Exception as e:
                        err_msg = f"Error ejecutando OS Action: {e}"
                        print(f"   ❌ {err_msg}")
                        logs.append(err_msg)
                else:
                    logs.append("Acción OS cancelada por el usuario.")
            except Exception as e:
                logs.append(f"Error parseando OS Action: {e}")

        # 3. Detección de Parches Quirúrgicos (Diffs) tolerante a Markdown
        patch_blocks = re.finditer(
            r"File:\s*([a-zA-Z0-9_.\-/]+).*?<<<< SEARCH\s*\n(.*?)\n====\s*\n(.*?)\n>>>> REPLACE",
            llm_response,
            re.DOTALL | re.IGNORECASE,
        )
        for match in patch_blocks:
            target_str = match.group(1).strip()
            search_text = match.group(2)
            replace_text = match.group(3)

            target_name = Path(target_str).name
            sandbox_path = self.sandbox_dir / target_name

            # Buscar el archivo original con validación estricta de Path Traversal
            source_path = sandbox_path
            if not source_path.exists():
                alt_path = Path(target_str)
                if not alt_path.is_absolute():
                    alt_path = (vromlix.paths.base / target_str).resolve()

                # Seguridad: Solo permitir parchear archivos dentro de VROMLIX_CORE
                if alt_path.exists() and alt_path.is_relative_to(vromlix.paths.base):
                    source_path = alt_path

            if self._hitl_prompt(
                f"El agente propone un PARCHE (Diff) para {target_name}. ¿Aplicar y guardar en SANDBOX?"
            ):
                if source_path.exists():
                    with open(source_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Intento 1: Reemplazo exacto
                    if search_text in content:
                        new_content = content.replace(search_text, replace_text)
                        patch_successful = True
                    # Intento 2: Reemplazo normalizando espacios en los extremos (Resiliencia)
                    elif search_text.strip() in content:
                        new_content = content.replace(
                            search_text.strip(), replace_text.strip()
                        )
                        patch_successful = True
                    else:
                        patch_successful = False

                    if patch_successful:
                        # SIEMPRE guardar en el Sandbox por seguridad
                        with open(sandbox_path, "w", encoding="utf-8") as f:
                            f.write(new_content)
                        msg = f"Parche aplicado. Archivo guardado en SANDBOX/{target_name}"
                        print(f"   ✅ {msg}")
                        logs.append(msg)
                    else:
                        msg = f"Fallo al aplicar parche: El bloque SEARCH no coincide con el contenido de {source_path.name} (ni siquiera normalizando espacios)."
                        print(f"   ❌ {msg}")
                        logs.append(msg)
                else:
                    msg = f"Fallo al aplicar parche: No se encontró el archivo original {target_name}."
                    print(f"   ❌ {msg}")
                    logs.append(msg)
            else:
                logs.append(f"Parche para {target_name} cancelado por el usuario.")

        return " | ".join(logs) if logs else "No OS/Code actions detected."


# --- 4. RAG, BAYESIAN LEARNING & TERMINAL UI (V2.0) ---


class DeepMemoryRetriever:
    """
    Motor de Recuperación Aumentada (RAG) conectado a sqlite-vec.
    """

    def __init__(self):
        self.db_path = str(vromlix.paths.vector_db / "vromlix_memory.sqlite")
        embed_config = vromlix.get_secret("EMBEDDINGS")
        self.embedding_model = (
            embed_config["model_id"] if embed_config else "gemini-embedding-2-preview"
        )

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        if not HAS_SQLITE_VEC or not os.path.exists(self.db_path):
            return ""

        query_vector = None
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                api_key = vromlix.get_api_key()
                client = genai.Client(api_key=api_key)
                response = client.models.embed_content(
                    model=self.embedding_model,
                    contents=query,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_QUERY", output_dimensionality=768
                    ),
                )
                query_vector = response.embeddings[0].values
                break
            except Exception as e:
                if "429" in str(e) or "503" in str(e):
                    time.sleep(1.5)
                    continue
                logging.error(f"Error generando embedding RAG: {e}")
                return ""

        if not query_vector:
            return ""

        try:
            db = sqlite3.connect(self.db_path)
            db.enable_load_extension(True)
            sqlite_vec.load(db)
            db.enable_load_extension(False)
            cursor = db.cursor()

            # FASE 3: GraphRAG (1-Hop Neighborhood Traversal)
            # Extraemos el nodo semilla y sus nodos adyacentes (contexto global)
            cursor.execute(
                """
                WITH seed_nodes AS (
                    SELECT m.id, m.source_file, m.network_type, m.confidence_score, v.distance
                    FROM vromlix_vectors v
                    JOIN vromlix_metadata m ON v.id = m.id
                    WHERE v.embedding MATCH ? AND k = ?
                )
                SELECT 
                    (SELECT group_concat(content, '\n... [NODO ADYACENTE] ...\n') 
                     FROM (
                         SELECT content FROM vromlix_metadata 
                         WHERE source_file = s.source_file 
                         AND id BETWEEN s.id - 1 AND s.id + 1 
                         ORDER BY id ASC
                     )
                    ) AS expanded_content,
                    s.network_type, 
                    s.confidence_score, 
                    s.distance
                FROM seed_nodes s
                ORDER BY s.distance ASC
            """,
                (json.dumps(query_vector), top_k),
            )

            results = cursor.fetchall()
            db.close()

            if not results:
                return ""

            context_blocks = ["=== DEEP MEMORY CONTEXT (HINDSIGHT W-B-O-S) ==="]
            context_blocks.append(
                "INSTRUCTION: Pay attention to the [NETWORK] tags. [W] = Absolute Facts/Code. [B] = Past Experiences/Chats. [O] = Opinions/Beliefs. [S] = Summaries."
            )

            for i, (content, net_type, conf_score, distance) in enumerate(results):
                # Mapeo visual para el LLM
                net_label = {
                    "W": "WORLD FACT",
                    "B": "BIOGRAPHICAL EXP",
                    "O": f"OPINION (Conf: {conf_score})",
                    "S": "SUMMARY",
                }.get(net_type, "UNKNOWN")

                context_blocks.append(f"--- Fragment {i + 1} [{net_label}] ---")
                context_blocks.append(content.strip())

            logging.info(f"📚 RAG: {len(results)} fragmentos recuperados.")
            return "\n".join(context_blocks)
        except Exception as e:
            logging.error(f"Error en RAG: {e}")
            return ""


class RealTimeVectorizer(threading.Thread):
    """Vectoriza la interacción en segundo plano y la guarda en sqlite-vec."""

    def __init__(self, interaction_text: str, db_path: str, embedding_model: str):
        super().__init__()
        self.interaction_text = interaction_text
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.daemon = True

    def run(self):
        if not HAS_SQLITE_VEC or not os.path.exists(self.db_path):
            return
        try:
            api_key = vromlix.get_api_key()
            client = genai.Client(api_key=api_key)
            response = client.models.embed_content(
                model=self.embedding_model,
                contents=self.interaction_text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768
                ),
            )
            vector = response.embeddings[0].values

            db = sqlite3.connect(self.db_path)
            db.enable_load_extension(True)
            sqlite_vec.load(db)
            db.enable_load_extension(False)
            cursor = db.cursor()

            # Las interacciones en tiempo real se guardan estrictamente como Experiencia Biográfica (B)
            cursor.execute(
                "INSERT INTO vromlix_metadata (source_file, chunk_type, content, network_type) VALUES (?, ?, ?, ?)",
                ("LIVE_SESSION", "real_time_memory", self.interaction_text, "B"),
            )
            row_id = cursor.lastrowid
            cursor.execute(
                "INSERT INTO vromlix_vectors (id, embedding) VALUES (?, ?)",
                (row_id, json.dumps(vector)),
            )
            db.commit()
            db.close()
        except Exception as e:
            logging.error(f"Fallo en Vectorizer: {e}")


class SubconsciousUpdater(threading.Thread):
    """Analiza la charla buscando nuevos datos para el 01_Dynamic_Profile.xml."""

    def __init__(self, interaction_text: str, profile_path: Path, profiler_prompt: str):
        super().__init__()
        self.interaction_text = interaction_text
        self.profile_path = profile_path
        self.profiler_prompt = profiler_prompt
        self.model_id = vromlix.get_model("VOLUMEN")
        self.daemon = True

    def run(self):
        prompt = self.profiler_prompt.format(
            interaction_text=self.interaction_text,
            timestamp=datetime.now().strftime("%Y-%m-%d"),
        )
        try:
            api_key = vromlix.get_api_key()
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            result = response.text.strip()
            if result != "NONE" and "<user_fact" in result:
                # Escribir en el nuevo historial biográfico
                history_path = vromlix.paths.prompts / "sys_roger_historial_biografico.xml"
                if history_path.exists():
                    with open(history_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    if "</historical_archive>" in content:
                        updated_content = content.replace(
                            "</historical_archive>",
                            f"  {result}\n</historical_archive>",
                        )
                        with open(history_path, "w", encoding="utf-8") as f:
                            f.write(updated_content)
        except Exception as e:
            logging.error(f"Fallo en SubconsciousUpdater: {e}")


class DocumentForgeAgent:
    """Agente Operario SOTA (La Forja): Genera documentos completos asíncronamente en el Sandbox."""

    def __init__(self, forge_prompt: str):
        self.sandbox_dir = vromlix.paths.sandbox
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.model = vromlix.get_model("VOLUMEN")
        self.forge_prompt = forge_prompt

    def execute_missions(self, missions_json: str):
        try:
            missions = json.loads(missions_json)
        except json.JSONDecodeError as e:
            print(f"   ❌ Error parseando misiones JSON: {e}")
            return

        total = len(missions)
        for index, mission in enumerate(missions, 1):
            target = mission.get("target", "unknown.txt")
            source = mission.get("source", "NONE")
            instruction = mission.get("instruction", "")
            source_content = "No source provided."

            if source != "NONE":
                source_path = self._find_file(source)
                if source_path:
                    try:
                        with open(source_path, "r", encoding="utf-8") as f:
                            source_content = f.read()
                    except Exception as e:
                        source_content = f"Error: {e}"

            # Inyección de Dependencia: Formateamos el prompt que viene del XML
            prompt = self.forge_prompt.format(
                target=target, instruction=instruction, source_content=source_content
            )

            try:
                client = genai.Client(api_key=vromlix.get_api_key())
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.1),
                )
                content = response.text.strip()

                # Limpieza de backticks residuales si el LLM los incluye
                if content.startswith("```") and content.endswith("```"):
                    lines = content.split("\n")
                    if len(lines) > 2:
                        content = "\n".join(lines[1:-1])

                target_path = self.sandbox_dir / target
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(content)

                sys.stdout.write("\033[K")  # Limpiar línea actual en terminal
                print(f"   ⚡ [{index}/{total}] Misión completada -> SANDBOX/{target}")
            except Exception as e:
                sys.stdout.write("\033[K")
                print(f"   ❌ [{index}/{total}] Error forjando '{target}': {e}")

    def _find_file(self, filename: str) -> Path | None:
        for root, _, files in os.walk(vromlix.paths.base):
            if "venv" in root or ".git" in root or "vector_db" in root:
                continue
            if filename in files:
                return Path(root) / filename
        return None


class VromlixTerminalUI:
    """Bucle principal de la aplicación (PAIOS)."""

    def __init__(self):
        self.max_file_size = (
            getattr(vromlix.config, "MAX_FILE_SIZE_MB", 5) if vromlix.config else 5
        )
        print("\n" + "=" * 50)
        print(" 🧠 INICIALIZANDO VROMLIX PRIME OS (v2.0)")
        print("=" * 50)

        self.monitor = TokenMonitor()
        self.loader = VromlixContextLoader()
        self.sys_prompts = self.loader.load_system_prompts()
        self.master_prompt = self.loader.build_master_system_prompt()
        self.tracker = SessionTracker()

        moe_content = self.loader._read_file(self.loader.moe_file)
        self.router = MoERouter(
            moe_content, self.monitor, self.sys_prompts.get("moe_router", "")
        )
        self.executor = AgenticExecutor(
            self.master_prompt,
            self.tracker,
            self.monitor,
            self.loader.repo_file,
        )

        self.retriever = DeepMemoryRetriever()
        self.synthesizer = OckhamSynthesizer(
            self.master_prompt,
            self.monitor,
            self.sys_prompts.get("ockham_fusion", ""),
            self.sys_prompts.get("ockham_auditor", ""),
        )
        self.firewall = SandboxFirewall()

        self.pending_attachments = ""
        print(
            "✅ Sistemas Nominales. Enjambre Multi-Agente Activo. Memoria a Corto Plazo Online.\n"
        )

    def run_headless(self, task_query: str) -> str:
        """Ejecuta una sola tarea en segundo plano (Para Benchmarks de la Industria)."""
        logging.info("🤖 [Headless Mode] Procesando tarea de Benchmark...")
        recent_context = self.tracker.get_recent_context(max_turns=1)
        routing_data = self.router.determine_routing(task_query, recent_context)
        rag_context = self.retriever.retrieve_context(task_query)

        swarm_responses = self.executor.process_swarm(
            task_query, routing_data, recent_context, rag_context, ""
        )

        final_response = self.synthesizer.synthesize(
            task_query, swarm_responses, routing_data
        )
        return final_response

    def start(self):
        try:
            while True:
                user_input = input("\n👤 Tú: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "salir"]:
                    print("\n🛑 Cerrando Vromlix Prime. Sesión guardada.")
                    break

                # --- COMANDOS DE ARCHIVO ---
                if user_input.startswith("/leer "):
                    filepath = user_input.replace("/leer ", "").strip()
                    target_path = Path(filepath)
                    if not target_path.is_absolute():
                        target_path = vromlix.paths.base / filepath

                    if target_path.exists() and target_path.is_file():
                        # Límite de seguridad dinámico para evitar desbordamiento de RAM/Tokens
                        if (
                            target_path.stat().st_size
                            > self.max_file_size * 1024 * 1024
                        ):
                            print(
                                f"❌ Archivo demasiado grande (>{self.max_file_size}MB). Operación cancelada para proteger la memoria."
                            )
                            continue
                        try:
                            with open(target_path, "r", encoding="utf-8") as f:
                                file_content = f.read()
                            self.pending_attachments += f"\n=== ATTACHED FILE: {target_path.name} ===\n{file_content}\n"
                            print(
                                f"📎 Archivo '{target_path.name}' cargado en memoria temporal. (Usa /limpiar para removerlo)"
                            )
                        except Exception as e:
                            print(f"❌ Error leyendo archivo: {e}")
                    else:
                        print(f"❌ Archivo no encontrado: {target_path}")
                    continue

                if user_input.strip().lower() == "/limpiar":
                    self.pending_attachments = ""
                    print("🧹 Bandeja de archivos adjuntos limpiada.")
                    continue

                # --- FASE 4: AUTO-EVOLUCIÓN SEGURA (MAKER-CHECKER) ---
                if user_input.startswith("/evolucionar "):
                    target_expert = user_input.replace("/evolucionar ", "").strip()
                    print(
                        f"🧬 Iniciando evolución algorítmica para: {target_expert}..."
                    )

                    try:
                        with open(self.loader.moe_file, "r", encoding="utf-8") as f:
                            moe_data = json.load(f)

                        expert_idx = next(
                            (
                                i
                                for i, exp in enumerate(moe_data)
                                if exp["expert_id"] == target_expert
                            ),
                            None,
                        )
                        if expert_idx is None:
                            print(f"❌ Experto '{target_expert}' no encontrado.")
                            continue

                        expert_profile = moe_data[expert_idx]
                        recent_logs = self.tracker.get_recent_context(max_turns=5)
                        client = genai.Client(api_key=vromlix.get_api_key())

                        # 1. MAKER (Propone la evolución)
                        maker_prompt = f"Optimize the 'instructions' and 'constraints' of this expert based on recent friction. Return ONLY a JSON object with the updated arrays.\nPROFILE: {json.dumps(expert_profile)}\nFRICTION: {recent_logs}"
                        maker_resp = client.models.generate_content(
                            model=vromlix.get_model("PRECISION"),
                            contents=maker_prompt,
                            config=types.GenerateContentConfig(
                                temperature=0.4, response_mime_type="application/json"
                            ),
                        )
                        proposed_update = json.loads(maker_resp.text)

                        # 2. CHECKER (Audita la propuesta)
                        print("   🛡️ [Checker] Auditando la mutación propuesta...")
                        checker_prompt = f'Compare the ORIGINAL profile with the PROPOSED update. If the proposal deletes critical core identity rules or hallucinates, return {{"approved": false}}. If it safely adds value, return {{"approved": true}}.\nORIGINAL: {json.dumps(expert_profile)}\nPROPOSED: {json.dumps(proposed_update)}'
                        checker_resp = client.models.generate_content(
                            model=vromlix.get_model("PRECISION"),
                            contents=checker_prompt,
                            config=types.GenerateContentConfig(
                                temperature=0.0, response_mime_type="application/json"
                            ),
                        )
                        verdict = json.loads(checker_resp.text)

                        # 3. EJECUCIÓN
                        if verdict.get("approved"):
                            moe_data[expert_idx]["instructions"] = proposed_update.get(
                                "instructions", expert_profile["instructions"]
                            )
                            moe_data[expert_idx]["constraints"] = proposed_update.get(
                                "constraints", expert_profile["constraints"]
                            )
                            with open(self.loader.moe_file, "w", encoding="utf-8") as f:
                                json.dump(moe_data, f, indent=2, ensure_ascii=False)
                            print(
                                f"   ✅ [Evolución Aprobada] El experto '{target_expert}' ha mutado exitosamente."
                            )
                            # Actualizar caché de embeddings para el Semantic Router
                            self.router.expert_vectors = (
                                self.router._load_expert_vectors()
                            )
                        else:
                            print(
                                "   ❌ [Evolución Rechazada] El Auditor detectó riesgo de pérdida de identidad. Mutación abortada."
                            )

                    except Exception as e:
                        print(f"❌ Error en la evolución: {e}")
                    continue

                # --- PREPARACIÓN DEL PROMPT ---
                full_query = user_input
                if self.pending_attachments:
                    full_query = (
                        f"{self.pending_attachments}\n\nUSER QUERY:\n{user_input}"
                    )
                    # Ya no limpiamos la bandeja automáticamente. Se mantiene hasta usar /limpiar.

                # --- PIPELINE COGNITIVO ---
                recent_context = self.tracker.get_recent_context(max_turns=3)
                routing_data = self.router.determine_routing(full_query, recent_context)

                # Grounding Desacoplado (Google News RSS Deep Research)
                web_context = ""
                search_queries = routing_data.get("search_queries", [])
                if search_queries:
                    osint_prompt = self.sys_prompts.get("osint_synthesis", "")
                    web_context = OSINTGrounder.execute_deep_research(
                        search_queries, osint_prompt
                    )

                # Siempre buscar en RAG para cruzar memoria profunda con la consulta actual
                rag_context = self.retriever.retrieve_context(full_query)

                # Ejecución del Enjambre
                swarm_responses = self.executor.process_swarm(
                    full_query, routing_data, recent_context, rag_context, web_context
                )

                # Guardar en crudo los pensamientos de los agentes en el tracker para memoria a corto plazo
                for exp_id, resp in swarm_responses.items():
                    self.tracker.log_interaction(f"Raw_Expert_Data [{exp_id}]", resp)

                # Síntesis y Auditoría
                print("🧠 [Vromlix] -> Sintetizando respuesta maestra...")
                final_response = self.synthesizer.synthesize(
                    full_query, swarm_responses, routing_data
                )

                # Firewall
                firewall_status = self.firewall.execute_if_present(final_response)

                # --- LOGGING Y UI ---
                self.tracker.log_interaction("Vromlix", final_response)
                if "No OS/Code actions" not in firewall_status:
                    self.tracker.log_interaction("System_Firewall", firewall_status)

                # Extracción de Misiones de Forja
                missions_json = None
                mission_match = re.search(
                    r"::: VROMLIX_MISSIONS :::\s*(.*?)\s*::: END_MISSIONS :::",
                    final_response,
                    re.DOTALL,
                )
                if mission_match:
                    missions_json = mission_match.group(1)

                # Filtro Estético: Ocultar el Tracker, Misiones y JSONs de la pantalla
                display_response = re.sub(
                    r"::: VROMLIX_STATE_TRACKER :::.*?::: END_TRACKER :::",
                    "\n*[Tracker Guardado en Memoria]*",
                    final_response,
                    flags=re.DOTALL,
                )
                display_response = re.sub(
                    r"::: VROMLIX_MISSIONS :::.*?::: END_MISSIONS :::",
                    "\n*[Misiones de Forja Delegadas]*",
                    display_response,
                    flags=re.DOTALL,
                )
                display_response = re.sub(
                    r'```(?:json)?\s*\{.*?"vromlix_os_action".*?\}\s*```',
                    "\n*[Acción de Sistema Ejecutada]*",
                    display_response,
                    flags=re.DOTALL,
                )

                print(f"\n🤖 Vromlix:\n{display_response}")
                print(f"\n📊 {self.monitor.get_summary()}")

                # Ejecutar misiones en la Fábrica de Documentos (Obrero Asíncrono)
                if missions_json:
                    forge_prompt = self.sys_prompts.get("document_forge", "")
                    forge = DocumentForgeAgent(forge_prompt)
                    forge.execute_missions(missions_json)

                # --- BACKGROUND OPS ---
                interaction_log = f"USER: {full_query}\n\nVROMLIX: {final_response}"
                RealTimeVectorizer(
                    interaction_log,
                    self.retriever.db_path,
                    self.retriever.embedding_model,
                ).start()
                SubconsciousUpdater(
                    interaction_log,
                    self.loader.profile_file,
                    self.sys_prompts.get("subconscious_profiler", ""),
                ).start()

        except KeyboardInterrupt:
            print(
                "\n\n🛑 Interrupción manual (Ctrl+C). Sesión guardada de forma segura. Hasta luego."
            )
            sys.exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vromlix Prime OS")
    parser.add_argument(
        "--task", type=str, help="Ejecutar en modo Headless para Benchmarks"
    )
    args = parser.parse_args()

    ui = VromlixTerminalUI()
    if args.task:
        respuesta = ui.run_headless(args.task)
        print("\n=== OUTPUT FINAL DEL BENCHMARK ===")
        print(respuesta)
    else:
        ui.start()
