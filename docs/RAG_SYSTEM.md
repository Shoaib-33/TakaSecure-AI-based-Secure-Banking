# TakaSecure model-routed adaptive RAG

This application uses LangChain and LangGraph components instead of handwritten
retrieval algorithms. The fine-tuned model served by vLLM makes the semantic
decisions: retrieval strategy, query rewriting, evidence grading, correction,
answer generation, and grounded-answer verification.

## Packaged components

- `ChatOpenAI` connects LangChain to vLLM's OpenAI-compatible endpoint.
- `QdrantVectorStore` performs dense BGE-M3 + sparse BM25 hybrid search with Qdrant fusion.
- `MultiQueryRetriever` performs model-generated query rewriting.
- `ContextualCompressionRetriever` and `CrossEncoderReranker` perform reranking.
- `StateGraph` implements the corrective retrieval and verification cycles.
- `UpstashRedisByteStore` stores SHA-256 exact-match verified responses.
- `PyPDFLoader` and `RecursiveCharacterTextSplitter` ingest the policy corpus.

The only handwritten code is configuration, typed schemas, prompts, graph wiring,
cache identity construction, and API boundaries. Safety limits and cache routing
remain deterministic; semantic decisions are made by the model.

All model decisions use vLLM JSON-schema constrained output through LangChain,
so no vLLM tool-call parser is required for routing, grading, or verification.

## Start

Use Python 3.11 on RunPod. First extract the downloaded adapter and serve it with
vLLM. Replace `BASE_MODEL` with the same base model used during fine-tuning.

```bash
unzip /workspace/TakaSecure-AI-based-Secure-Banking/artifacts/takasecure-adapter-v3.zip -d /workspace/takasecure-adapter-v3

vllm serve BASE_MODEL \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-lora \
  --lora-modules takasecure=/workspace/takasecure-adapter-v3/adapter \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

In a second terminal:

```bash
cd /workspace/TakaSecure-AI-based-Secure-Banking
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
uvicorn takasecure_rag.main:app --host 0.0.0.0 --port 8080
```

The first API startup downloads embedding/reranker models, reads the PDF, and
constructs the in-memory Qdrant collection. Later restarts rebuild it because
`:memory:` is intentionally non-persistent.

Test the API:

```bash
curl -X POST http://127.0.0.1:8080/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"What evidence is required for beneficial ownership verification?","user_role":"compliance_analyst","department":"compliance"}'
```

## Upstash cache

Create an Upstash Redis database and put its REST URL and token in `.env`:

```dotenv
UPSTASH_REDIS_REST_URL=https://...
UPSTASH_REDIS_REST_TOKEN=...
```

Only answers that pass the model verifier are cached. The exact-match SHA-256
identity includes the full request, role, department, corpus version, vLLM model,
and pipeline version. Changing any of these produces a different cache entry.

## Important behavior

- Cache availability never determines whether the RAG request succeeds.
- Cache hits bypass retrieval and vLLM; misses run the full graph.
- Query planning and evidence decisions come from structured model output.
- One corrective retrieval and one regeneration are hard safety bounds.
- The corpus is synthetic and must not be presented as real banking policy.
