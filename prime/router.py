"""
prime.router.py — VROMLIX Prime: Mixture-of-Experts Semantic Router
MoERouter + leer_lineas_de_archivo utility.
Split from core_vromlix_prime.py (God Class refactor).
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

from prime.memory import TokenMonitor
from prime.models import RoutingResult, VromlixBackend
from vromlix_utils import vromlix


class MoERouter:
    """Advanced Semantic Router (Mixture of Experts)."""

    def __init__(
        self,
        moe_json_content: str,
        monitor: TokenMonitor,
        router_prompt: str,
        backend: VromlixBackend,
    ):
        self._backend = backend
        self.model_id = backend.get_model("VOLUMEN")
        embed_cfg = backend.get_secret("EMBEDDINGS")
        self.embed_model = (
            embed_cfg.get("model_id") or embed_cfg.get("primary")
            if embed_cfg
            else "gemini-embedding-2-preview"
        )
        self.monitor = monitor
        self.router_prompt = router_prompt
        self.cache_db_path = backend.paths.databases / "cache.sqlite"
        self._init_cache_db()

        try:
            self.moe_data = json.loads(moe_json_content)
        except json.JSONDecodeError:
            logging.error("CRITICAL: Failed to parse MoE Routing data")
            self.moe_data = []
        self.expert_vectors = self._load_expert_vectors()

    def _init_cache_db(self):
        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS moe_cache (expert_id TEXT PRIMARY KEY, vector TEXT)"
            )
            conn.commit()

    def _cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        import math

        mag1, mag2 = (
            math.sqrt(sum(a * a for a in v1)),
            math.sqrt(sum(b * b for b in v2)),
        )
        return (
            sum(a * b for a, b in zip(v1, v2, strict=False)) / (mag1 * mag2)
            if mag1 and mag2
            else 0.0
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=False,
    )
    def _load_expert_vectors(self) -> dict[str, list[float]]:
        vectors = {}

        # Intentar cargar desde SQLite
        with sqlite3.connect(self.cache_db_path) as conn:
            cursor = conn.cursor()
            rows = cursor.execute("SELECT expert_id, vector FROM moe_cache").fetchall()
            if len(rows) == len(self.moe_data) and len(self.moe_data) > 0:
                for row in rows:
                    vectors[row[0]] = json.loads(row[1])
                return vectors

        logging.info("🧠 [Semantic Router] Generating expert embeddings (Cache Miss)...")

        # Generar nuevos vectores
        for exp in self.moe_data:
            text_to_embed = (
                f"Role: {exp['expert_id']}. "
                f"Mechanics: {', '.join(exp.get('mechanics', []))}. "
                f"Instructions: {', '.join(exp.get('instructions', []))}."
            )
            vector = self._backend.get_embeddings(text_to_embed)
            vectors[exp["expert_id"]] = vector

        # Guardar en SQLite
        with sqlite3.connect(self.cache_db_path) as conn:
            for exp_id, vec in vectors.items():
                conn.execute(
                    "INSERT OR REPLACE INTO moe_cache (expert_id, vector) VALUES (?, ?)",
                    (exp_id, json.dumps(vec)),
                )
            conn.commit()

        return vectors

    def _get_expert_profile(self, expert_id: str) -> dict:
        profile = next((e for e in self.moe_data if e["expert_id"] == expert_id), None)
        if not profile:
            profile = next(
                (e for e in self.moe_data if e["expert_id"] == "ORCHESTRATE_SYSTEM_CORE"),
                {},
            )
        return dict(profile) if profile else {}

    def determine_routing(self, user_query: str, recent_context: str) -> dict:
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

        if self.expert_vectors and len(recent_context) < 500:
            try:
                query_vector = self._backend.get_embeddings(user_query)
                best_expert, max_sim = None, -1.0
                for exp_id, exp_vector in self.expert_vectors.items():
                    sim = self._cosine_similarity(query_vector, exp_vector)
                    if sim > max_sim:
                        max_sim, best_expert = sim, exp_id

                if max_sim > 0.75 and best_expert:
                    logging.info(
                        f"⚡ [Semantic Router] Short circuit: {best_expert} (Sim: {max_sim:.2f})"
                    )
                    step_dict = self._get_expert_profile(best_expert)
                    step_dict.update({"step_id": "step_1", "dynamic_files": [], "depends_on": []})
                    return {"execution_plan": [step_dict], "search_queries": []}
            except Exception as e:
                logging.warning(f"⚠️ Failure in Semantic Gateway, using LLM Fallback: {e}")

        routing_map = [
            {"id": exp["expert_id"], "cluster": exp["parent_cluster"]} for exp in self.moe_data
        ]
        prompt = self.router_prompt.format(
            recent_context=recent_context,
            user_query=user_query,
            routing_map=json.dumps(routing_map, indent=2),
            current_date=datetime.now().strftime("%Y-%m-%d"),
            current_year=datetime.now().strftime("%Y"),
        )

        try:
            result = self._backend.query_universal_llm(
                system_prompt="You are the MoE Router. Extract structured routing information.",
                user_prompt=prompt,
                role="VOLUMEN",
                response_model=RoutingResult,
            )
            logging.info(f"🧠 [Router CoT]: {result.internal_analysis}")

            execution_plan = []
            for step in result.execution_plan[:3]:
                step_dict = self._get_expert_profile(step.expert_id)
                step_dict.update(
                    {
                        "step_id": step.step_id,
                        "dynamic_files": step.required_files,
                        "depends_on": step.depends_on,
                    }
                )
                execution_plan.append(step_dict)

            dag_steps = [s["step_id"] + "(" + s["expert_id"] + ")" for s in execution_plan]
            logging.info(f"🔀 MoE Router: DAG Planned -> {dag_steps}")
            if result.search_queries:
                logging.info(f"🌐 MoE Router: Web Search requested -> {result.search_queries}")
            return {
                "execution_plan": execution_plan,
                "search_queries": result.search_queries,
            }
        except Exception as e:
            logging.warning(f"Failure in MoE Routing ({e}). Using Fallback.")
            fallback_profile = self._get_expert_profile("ORCHESTRATE_SYSTEM_CORE")
            fallback_profile.update({"step_id": "step_1", "dynamic_files": [], "depends_on": []})
            return {"execution_plan": [fallback_profile], "search_queries": []}


def leer_lineas_de_archivo(filepath: str, linea_inicio: int = 1, linea_fin: int = 100) -> str:
    """Read a specific range of lines from a local file to avoid memory saturation."""
    try:
        target_path = vromlix.paths.base / filepath
        if not target_path.exists() or not target_path.is_file():
            target_path = Path(vromlix.paths.config_xml) / filepath
            if not target_path.exists():
                return f"[ERROR TOOL]: The file {filepath} does not exist."

        with target_path.open(encoding="utf-8") as f:
            lineas = f.readlines()
        total_lineas = len(lineas)
        inicio_idx, fin_idx = max(0, linea_inicio - 1), min(total_lineas, linea_fin)

        if inicio_idx >= total_lineas:
            return (
                f"[ERROR TOOL]: linea_inicio ({linea_inicio}) is greater than"
                f" total lines ({total_lineas})."
            )

        # Satisfy IDE linter indexing
        slice_lines = [lineas[k] for k in range(inicio_idx, fin_idx)]
        header = f"--- FILE: {filepath} (Lines {linea_inicio} to {fin_idx} of {total_lineas}) ---"
        return f"{header}\n{''.join(slice_lines)}"
    except Exception as e:
        return f"[ERROR TOOL]: Failed to read {filepath} -> {e!s}"
