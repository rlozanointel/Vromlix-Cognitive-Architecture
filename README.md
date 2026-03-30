
# 🧠 Vromlix Prime v4.0: Cognitive Edge Architecture

An advanced, deterministic Multi-Agent Orchestration system engineered for sovereign, high-performance execution in severely resource-constrained environments (Sub-8GB RAM edge computing). 

## 🏗️ System Architecture & Workflow

Vromlix operates under a strict deterministic delegation model. It replaces zero-shot LLM guessing with Monte Carlo Tree Search (MCTS) routing, and flat RAG with a topological GraphRAG memory system, ensuring zero-hallucination outputs.

```mermaid
graph TD;
    subgraph Interfaces[Input Layer]
        A[Terminal UI / Human] --> B{MCTS Pydantic Router};
        A2[FastAPI Headless Proxy] --> B;
    end

    subgraph Cognitive Core[Agentic Swarm Execution]
        B -->|DAG Plan| C(Parallel Expert Threads);
        C --> D[Ockham Synthesizer & Red Team Auditor];
    end

    subgraph Sovereign Memory[Hindsight W-B-O-S Architecture]
        C <--> E[GraphRAG 1-Hop Traversal];
        E <--> F[(SQLite-Vec Deep Memory)];
    end

    subgraph Edge Compute[Local Execution]
        C -->|LOCAL_ Prefix| G[llama.cpp 8-bit KV Quantization];
        D --> H[HitL Sandbox Firewall];
    end
```

## 🚀 Technical Highlights

* **MCTS-Driven MoE Routing:** Replaces fragile zero-shot routing with a lightweight Monte Carlo Tree Search (MCTS) heuristic. The system mathematically simulates and scores multiple Directed Acyclic Graph (DAG) execution paths via Pydantic before delegating tasks.
* **Edge-Optimized GraphRAG & W-B-O-S Memory:** Eliminates context amnesia by implementing a Hindsight epistemic memory taxonomy (World, Biographical, Opinion, Summary). Utilizes SQLite Recursive CTEs for 1-Hop neighborhood traversal, achieving GraphRAG capabilities without heavy JVM graph databases.
* **Sovereign Edge Compute (Anti-OOM):** Executes local LLMs (Phi-4) strictly within an 8GB RAM ceiling by implementing 8-bit Key-Value (KV) Cache quantization and dynamic context compression via `llama-cpp-python`.
* **Headless Benchmarking Proxy:** Features a decoupled FastAPI proxy server, allowing autonomous machine-to-machine evaluation against industry-standard benchmarks (SWE-bench, GAIA) without UI blocking.
* **Resilient API Multiplexing:** Native `google-genai` integration fortified with an Active Key Manager (Round-Robin) to effortlessly absorb HTTP 429/503 errors and maximize throughput under strict TPM quotas.

## 🛠️ Stack & Dependencies
- **Core Engine:** Python 3.12, `pydantic` (Deterministic DAGs), `FastAPI`
- **Edge AI & Memory:** `llama-cpp-python`, `sqlite-vec` (C++ WASM)
- **LLM Orchestration:** `google-genai` (Gemini 3.0 Flash/Pro)
- **OSINT & Scraping:** `feedparser`, `requests` (Map-Reduce Pipelines)

---
*Built for rigorous Personal Knowledge Graph compilation and continuous code refactoring.*

