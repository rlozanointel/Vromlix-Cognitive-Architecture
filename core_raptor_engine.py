#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "tenacity>=9.0.0",
#     "sqlite-vec>=0.1.3",
#     "scikit-learn>=1.5.0",
#     "scikit-learn-intelex",
#     "umap-learn>=0.5.11",
#     "numpy>=1.24.0",
#     "jsonref",
#     "pydantic>=2.12.5",
#     "sumy>=0.11.0",
#     "yake>=0.4.8",
#     "nltk>=3.9.1",
#     "llama-cpp-python>=0.2.56"
# ]
# ///

# -*- coding: utf-8 -*-
# @description RAPTOR Memory Consolidation (v2.2) - SOTA Maker-Checker & Intel Optimized.

import argparse
import json
import logging
import os
import sqlite3
import sys
import warnings
from pathlib import Path

import nltk
import numpy as np
import yake
from pydantic import BaseModel, Field
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.utils import get_stop_words
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

sys.path.append(str(Path(__file__).parents[1]))
from vromlix_utils import vromlix

# --- SOTA SILENCE ---
warnings.filterwarnings("ignore", category=UserWarning, module="umap")
logging.getLogger("sklearnex").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# --- SOTA Intel Optimization ---
try:
    from sklearnex import patch_sklearn

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    with Path(os.devnull).open("w") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            patch_sklearn()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
except ImportError:
    pass


# --- MODELOS ESTRUCTURADOS ---
class RaptorSummaryNode(BaseModel):
    cluster_theme: str = Field(description="Título técnico conciso (3-5 palabras).")
    comprehensive_summary: str = Field(description="Resumen de alta densidad (máx 3 oraciones).")
    extracted_entities: list[str] = Field(
        description="Librerías, algoritmos, métricas o nombres propios."
    )
    critical_claims: list[str] = Field(description="Afirmaciones factuales directas.")


class RaptorAudit(BaseModel):
    approved: bool = Field(..., description="¿El resumen captura la esencia técnica sin paja?")
    feedback: str = Field(..., description="Instrucciones de mejora si fue rechazado.")


# --- MOTOR PRINCIPAL ---
class VromlixRaptorEngine:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.total_consolidated = 0

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except ImportError:
            pass
        return conn

    def reset_hierarchy(self):
        """Elimina la jerarquía previa para forzar una consolidación global."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM vromlix_metadata WHERE chunk_type = 'summary_node'")
            cursor.execute("UPDATE vromlix_metadata SET parent_id = NULL")
            cursor.execute(
                "DELETE FROM vromlix_vectors WHERE id NOT IN (SELECT id FROM vromlix_metadata)"
            )
            conn.commit()
        finally:
            conn.close()

    def get_unconsolidated_leaves(self, target_level: int = 0) -> list:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT m.id, m.content, v.embedding
                FROM vromlix_metadata m
                JOIN vromlix_vectors v ON m.id = v.id
                WHERE m.tree_level = ? AND m.parent_id IS NULL AND m.chunk_type != 'summary_node'
                """,
                (target_level,),
            )
            return cursor.fetchall()
        finally:
            conn.close()

    def determine_optimal_clusters(self, embeddings_matrix: np.ndarray, max_k: int = 50) -> tuple:
        """Clustering semántico usando UMAP y Gaussian Mixture Models."""
        import umap
        from sklearn.mixture import GaussianMixture

        N = len(embeddings_matrix)
        if N < 12:
            return (1, np.zeros(N, dtype=int)) if N > 0 else (None, None)

        reducer = umap.UMAP(
            n_components=min(10, N - 1), n_neighbors=min(15, N - 1), random_state=42
        )
        reduced_embeddings = np.float64(reducer.fit_transform(embeddings_matrix))

        bics, models = [], []
        k_range = range(1, min(max_k, N // 2) + 1)

        for k in k_range:
            gmm = GaussianMixture(
                n_components=k, covariance_type="full", random_state=42, reg_covar=1e-4
            )
            gmm.fit(reduced_embeddings)
            bics.append(gmm.bic(reduced_embeddings))
            models.append(gmm)

        optimal_k_idx = np.argmin(bics)
        return k_range[optimal_k_idx], models[optimal_k_idx].predict(reduced_embeddings)

    def audit_summary(self, summary: RaptorSummaryNode) -> RaptorAudit:
        """Fase Checker: Validacion por Reglas Duras (Sin LLM)."""
        texto = summary.comprehensive_summary

        # Regla 1: Longitud
        if len(texto) < 50:
            return RaptorAudit(
                approved=False,
                feedback="Rechazado: El resumen es demasiado corto para aportar valor.",
            )

        # Regla 2: Entidades
        entidades_encontradas = sum(
            1 for e in summary.extracted_entities if e.lower() in texto.lower()
        )
        if len(summary.extracted_entities) > 0 and entidades_encontradas == 0:
            feedback = (
                "Rechazado: El resumen no menciona ninguna de las entidades clave detectadas."
            )
            return RaptorAudit(approved=False, feedback=feedback)

        return RaptorAudit(approved=True, feedback="Aprobado por heurística.")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True
    )
    def generate_refined_summary(self, chunks: list[str]) -> RaptorSummaryNode:
        """Fase Maker: Generación Extractiva (Sin LLM)."""
        full_text = " ".join(chunks)

        # 1. Extraer entidades clave (YAKE)
        kw_extractor = yake.KeywordExtractor(lan="es", n=2, dedupLim=0.9, top=5, features=None)
        keywords = kw_extractor.extract_keywords(full_text)
        entities = [kw[0] for kw in keywords]

        cluster_theme = " | ".join(entities[:3]).upper() if entities else "CLUSTER_SIN_TEMA"

        # 2. Resumir (Sumy LexRank)
        parser = PlaintextParser.from_string(full_text, Tokenizer("spanish"))
        summarizer = LexRankSummarizer(Stemmer("spanish"))
        summarizer.stop_words = get_stop_words("spanish")

        sentences = summarizer(parser.document, sentences_count=3)
        comprehensive_summary = " ".join(str(s) for s in sentences)

        summary = RaptorSummaryNode(
            cluster_theme=cluster_theme,
            comprehensive_summary=comprehensive_summary,
            extracted_entities=entities,
            critical_claims=[],
        )

        # 3. Auditar (Checker)
        audit = self.audit_summary(summary)

        if not audit.approved:
            print(f"\n      ⚠️ Resumen rechazado. Feedback: {audit.feedback}")
            # Fallback simple: extraer más contexto
            sentences = summarizer(parser.document, sentences_count=5)
            summary.comprehensive_summary = " ".join(str(s) for s in sentences)

        return summary

    @retry(
        stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True
    )
    def embed_and_store_parent(
        self, summary_node: RaptorSummaryNode, child_ids: list[int], cluster_id: int
    ):
        summary_text = (
            f"THEME: {summary_node.cluster_theme}\n"
            f"SUMMARY: {summary_node.comprehensive_summary}\n"
            f"ENTITIES: {', '.join(summary_node.extracted_entities)}"
        )

        vector = vromlix.get_embeddings(summary_text, role="EMBEDDINGS")
        if not vector:
            raise ValueError("No embeddings returned from local engine")

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO vromlix_metadata "
                "(source_file, chunk_type, content, tree_level, cluster_id) "
                "VALUES (?, ?, ?, ?, ?)",
                ("RAPTOR_CONSOLIDATION", "summary_node", summary_text, 1, cluster_id),
            )
            parent_id = cursor.lastrowid
            cursor.execute(
                "INSERT INTO vromlix_vectors (id, embedding) VALUES (?, ?)",
                (parent_id, json.dumps(vector)),
            )
            placeholders = ",".join("?" * len(child_ids))
            cursor.execute(
                f"UPDATE vromlix_metadata SET parent_id = ? WHERE id IN ({placeholders})",
                (parent_id, *child_ids),
            )
            conn.commit()
        finally:
            conn.close()
        return True

    def run_consolidation(self, force_full: bool = False) -> None:
        if force_full:
            print("🧹 Reseteando jerarquía previa...")
            self.reset_hierarchy()

        records = self.get_unconsolidated_leaves(target_level=0)
        if not records:
            print("✅ No hay nodos huérfanos para consolidar.")
            return

        ids = [int(r[0]) for r in records]
        texts = [str(r[1]) for r in records]
        embeddings = np.array(
            [
                (
                    np.frombuffer(r[2], dtype=np.float32)
                    if isinstance(r[2], bytes)
                    else json.loads(r[2])
                )
                for r in records
            ],
            dtype=np.float32,
        )

        optimal_k, labels = self.determine_optimal_clusters(embeddings)
        if optimal_k is None:
            return

        print(f"🚀 Iniciando Consolidación Semántica ({optimal_k} nodos padre proyectados)...")
        for cluster_id in range(optimal_k):
            indices = np.where(labels == cluster_id)[0]
            cluster_texts = [str(texts[i]) for i in indices][:20]
            cluster_db_ids = [int(ids[i]) for i in indices]
            print(
                f"\r   ⏳ Sintetizando: [{cluster_id + 1}/{optimal_k}] "
                f"({(cluster_id + 1) / optimal_k * 100:.1f}%)",
                end="",
                flush=True,
            )

            try:
                summary_obj = self.generate_refined_summary(cluster_texts)
                if summary_obj:
                    self.embed_and_store_parent(summary_obj, cluster_db_ids, cluster_id)
                    self.total_consolidated += 1
            except Exception as e:
                print(f"\n   ❌ Error en cluster {cluster_id + 1}: {str(e)[:100]}")

        print(f"\n🎉 RAPTOR COMPLETADO. {self.total_consolidated} nodos consolidados exitosamente.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VROMLIX RAPTOR Consolidation Engine")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Delete current hierarchy and perform global consolidation.",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="vromlix_memory.sqlite",
        help="Nombre de la base de datos a consolidar",
    )
    args = parser.parse_args()

    db_target = str(vromlix.paths.databases / args.db)
    print(f"🦅 Lanzando RAPTOR SOTA Consolidator en {args.db}...")

    engine = VromlixRaptorEngine(db_path=db_target)
    try:
        engine.run_consolidation(force_full=args.full)
    except KeyboardInterrupt:
        print("\n🛑 Proceso interrumpido por el usuario.")
    except Exception as e:
        print(f"\n❌ Error Crítico: {e}")
