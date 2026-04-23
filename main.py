# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-genai>=1.68.0", "instructor>=1.7.0", "tenacity>=9.0.0",
#     "httpx>=0.28.1", "numpy>=2.2.6", "pydantic>=2.12.5", "sqlite-vec>=0.1.9",
#     "duckduckgo-search>=8.1.1", "feedparser>=6.0.12", "lxml>=5.1.0",
#     "tqdm>=4.67.3", "markitdown>=0.0.1a4"
# ]
# ///

#!/usr/bin/env -S uv run
# -*- coding: utf-8 -*-
"""
Vromlix Prime V2.0 — CLI Orchestrator (Facade Entry Point)
This file is the CLI entry point. All heavy logic is split into sub-modules:
  - prime.models.py   : Shared protocols and Pydantic models
  - prime.memory.py   : TokenMonitor, VromlixContextLoader, SessionTracker
  - prime.router.py   : MoERouter, leer_lineas_de_archivo
  - prime.executor.py : AgenticExecutor, OckhamSynthesizer, SandboxFirewall
  - prime.threads.py  : DeepMemoryRetriever, RealTimeVectorizer,
                        SubconsciousUpdater, DocumentForgeAgent
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from prime.executor import (
    AgenticExecutor,
    OckhamSynthesizer,
    SandboxFirewall,
)
from prime.memory import (
    SessionTracker,
    TokenMonitor,
    VromlixContextLoader,
)

# ── Sub-module imports (Facade pattern — backward compatibility preserved) ──
from prime.models import (
    ExecutionStep,
    RoutingResult,
    SimulatedPath,
    VromlixBackend,
)
from prime.router import MoERouter, leer_lineas_de_archivo
from prime.threads import (
    DeepMemoryRetriever,
    DocumentForgeAgent,
    RealTimeVectorizer,
    SubconsciousUpdater,
)

from vromlix_utils import OSINTGrounder, vromlix

try:
    import sqlite_vec  # noqa: F401

    HAS_SQLITE_VEC = True
except ImportError:
    HAS_SQLITE_VEC = False
    logging.warning("⚠️ sqlite-vec not installed. Deep memory (RAG) will be disabled.")

__all__ = [
    "AgenticExecutor",
    "DeepMemoryRetriever",
    "DocumentForgeAgent",
    "ExecutionStep",
    "MoERouter",
    "OckhamSynthesizer",
    "RealTimeVectorizer",
    "RoutingResult",
    "SandboxFirewall",
    "SessionTracker",
    "SimulatedPath",
    "SubconsciousUpdater",
    "TokenMonitor",
    "VromlixBackend",
    "VromlixContextLoader",
    "VromlixTerminalUI",
    "leer_lineas_de_archivo",
]


class VromlixTerminalUI:
    """Main application loop (PAIOS)."""

    def __init__(self):
        self.max_file_size = getattr(vromlix.config, "MAX_FILE_SIZE_MB", 5) if vromlix.config else 5
        print("\n" + "=" * 50 + "\n 🧠 INITIALIZING VROMLIX PRIME OS (v2.0)\n" + "=" * 50)

        self.monitor = TokenMonitor()
        self.loader = VromlixContextLoader()
        self.sys_prompts = self.loader.load_system_prompts()
        self.master_prompt = self.loader.build_master_system_prompt()
        self.tracker = SessionTracker()

        moe_content = self.loader._read_file(self.loader.moe_file)
        self.router = MoERouter(
            moe_content,
            self.monitor,
            self.sys_prompts.get("moe_router", ""),
            backend=vromlix,
        )
        self.executor = AgenticExecutor(
            self.master_prompt,
            self.tracker,
            self.monitor,
            self.loader.repo_file,
            backend=vromlix,
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
        print("✅ Systems Nominal. Multi-Agent Swarm Active. Short-Term Memory Online.\n")

    def run_headless(self, task_query: str) -> str:
        """Executes a single task in the background (For Industry Benchmarks)."""
        logging.info("🤖 [Headless Mode] Processing Benchmark task...")
        recent_context = self.tracker.get_recent_context(max_turns=1)
        routing_data = self.router.determine_routing(task_query, recent_context)
        rag_context = self.retriever.retrieve_context(task_query)
        swarm_responses = self.executor.process_swarm(
            task_query, routing_data, recent_context, rag_context, ""
        )
        return self.synthesizer.synthesize(task_query, swarm_responses, routing_data)

    def start(self):
        try:
            while True:
                user_input = input("\n👤 You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "salir"]:
                    print("\n🛑 Closing Vromlix Prime. Session saved.")
                    break

                if user_input.startswith("/leer "):
                    filepath = user_input.replace("/leer ", "").strip()
                    target_path = (
                        Path(filepath)
                        if Path(filepath).is_absolute()
                        else vromlix.paths.base / filepath
                    )

                    if target_path.exists() and target_path.is_file():
                        if target_path.stat().st_size > self.max_file_size * 1024 * 1024:
                            print(
                                f"❌ File too large (>{self.max_file_size}MB)."
                                " Operation cancelled to protect memory."
                            )
                            continue
                        try:
                            with target_path.open(encoding="utf-8") as f:
                                file_content = f.read()
                            self.pending_attachments += (
                                f"\n=== ATTACHED FILE: {target_path.name} ===\n{file_content}\n"
                            )
                            print(
                                f"📎 File '{target_path.name}' loaded into memory."
                                " (Use /limpiar to remove it)"
                            )
                        except Exception as e:
                            print(f"❌ Error reading file: {e}")
                    else:
                        print(f"❌ File not found: {target_path}")
                    continue

                if user_input.strip().lower() == "/limpiar":
                    self.pending_attachments = ""
                    print("🧹 Attachment tray cleared.")
                    continue

                if user_input.startswith("/evolucionar "):
                    target_expert = user_input.replace("/evolucionar ", "").strip()
                    print(f"🧬 Starting algorithmic evolution for: {target_expert}...")

                    try:
                        with self.loader.moe_file.open(encoding="utf-8") as f:
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
                            print(f"❌ Expert '{target_expert}' not found.")
                            continue

                        expert_profile = moe_data[expert_idx]
                        recent_logs = self.tracker.get_recent_context(max_turns=5)

                        maker_prompt = (
                            "Optimize the 'instructions' and 'constraints' of this expert based on"
                            " recent friction. Return ONLY a JSON object with the updated arrays.\n"
                            f"PROFILE: {json.dumps(expert_profile)}\nFRICTION: {recent_logs}"
                        )
                        maker_res = vromlix.query_universal_llm(
                            system_prompt=(
                                "You are an expert system optimizer. Return valid JSON only."
                            ),
                            user_prompt=maker_prompt,
                            role="PRECISION",
                        )
                        proposed_update = json.loads(maker_res.text)

                        print("   🛡️ [Checker] Auditing the proposed mutation...")
                        checker_prompt = (
                            "Compare the ORIGINAL profile with the PROPOSED update."
                            " If critical core identity rules are deleted or it hallucinates,"
                            ' return {"approved": false}; if it safely adds value,'
                            ' return {"approved": true}.'
                            f"\nORIGINAL: {json.dumps(expert_profile)}"
                            f"\nPROPOSED: {json.dumps(proposed_update)}"
                        )
                        checker_res = vromlix.query_universal_llm(
                            system_prompt="You are a safety auditor. Return valid JSON only.",
                            user_prompt=checker_prompt,
                            role="PRECISION",
                        )
                        verdict = json.loads(checker_res.text)

                        if verdict.get("approved"):
                            moe_data[expert_idx]["instructions"] = proposed_update.get(
                                "instructions", expert_profile["instructions"]
                            )
                            moe_data[expert_idx]["constraints"] = proposed_update.get(
                                "constraints", expert_profile["constraints"]
                            )
                            with self.loader.moe_file.open("w", encoding="utf-8") as f:
                                json.dump(moe_data, f, indent=2, ensure_ascii=False)
                            print(f"   ✅ [Evolution Approved] Expert '{target_expert}' mutated.")
                            self.router.expert_vectors = self.router._load_expert_vectors()
                        else:
                            print(
                                "   ❌ [Evolution Rejected] Identity-loss risk detected. Aborted."
                            )
                    except Exception as e:
                        print(f"❌ Error in evolution: {e}")
                    continue

                full_query = (
                    f"{self.pending_attachments}\n\nUSER QUERY:\n{user_input}"
                    if self.pending_attachments
                    else user_input
                )
                recent_context = self.tracker.get_recent_context(max_turns=3)
                routing_data = self.router.determine_routing(full_query, recent_context)

                web_context = ""
                search_queries = routing_data.get("search_queries", [])
                if search_queries:
                    web_context = OSINTGrounder.execute_deep_research(
                        search_queries, self.sys_prompts.get("osint_synthesis", "")
                    )

                rag_context = self.retriever.retrieve_context(full_query)
                swarm_responses = self.executor.process_swarm(
                    full_query, routing_data, recent_context, rag_context, web_context
                )

                for exp_id, resp in swarm_responses.items():
                    self.tracker.log_interaction(f"Raw_Expert_Data [{exp_id}]", resp)

                print("🧠 [Vromlix] -> Synthesizing master response...")
                final_response = self.synthesizer.synthesize(
                    full_query, swarm_responses, routing_data
                )
                firewall_status = self.firewall.execute_if_present(final_response)

                self.tracker.log_interaction("Vromlix", final_response)
                if "No OS/Code actions" not in firewall_status:
                    self.tracker.log_interaction("System_Firewall", firewall_status)

                missions_json = None
                mission_match = re.search(
                    r"::: VROMLIX_MISSIONS :::\s*(.*?)\s*::: END_MISSIONS :::",
                    final_response,
                    re.DOTALL,
                )
                if mission_match:
                    missions_json = mission_match.group(1)

                display_response = re.sub(
                    r"::: VROMLIX_STATE_TRACKER :::.*?::: END_TRACKER :::",
                    "\n*[Tracker Saved in Memory]*",
                    final_response,
                    flags=re.DOTALL,
                )
                display_response = re.sub(
                    r"::: VROMLIX_MISSIONS :::.*?::: END_MISSIONS :::",
                    "\n*[Forge Missions Delegated]*",
                    display_response,
                    flags=re.DOTALL,
                )
                display_response = re.sub(
                    r'```(?:json)?\s*\{.*?"vromlix_os_action".*?\}\s*```',
                    "\n*[System Action Executed]*",
                    display_response,
                    flags=re.DOTALL,
                )

                print(f"\n🤖 Vromlix:\n{display_response}")
                print(f"\n📊 {self.monitor.get_summary()}")

                if missions_json:
                    DocumentForgeAgent(self.sys_prompts.get("document_forge", "")).execute_missions(
                        missions_json
                    )

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
            print("\n\n🛑 Manual interruption (Ctrl+C). Session safely saved. Goodbye.")
            sys.exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vromlix Prime OS")
    parser.add_argument("--task", type=str, help="Execute in Headless mode for Benchmarks")
    args = parser.parse_args()

    ui = VromlixTerminalUI()
    if args.task:
        print("\n=== FINAL BENCHMARK OUTPUT ===")
        print(ui.run_headless(args.task))
    else:
        ui.start()
