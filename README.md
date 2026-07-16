# TakaSecure AI-based Secure Banking

TakaSecure is a synthetic banking AI portfolio project combining supervised
fine-tuning, vLLM serving, adaptive hybrid RAG, structured evaluation, and
verified-response caching. No content in this repository is real banking
policy, regulation, or customer data.

## What the project demonstrates

- A reproducible 7,000-example SFT dataset with isolated train, validation, and test policy families.
- A LoRA adapter trained for citations, structured JSON, abstention, policy conflicts, prompt-injection resistance, and tool routing.
- vLLM serving through an OpenAI-compatible endpoint.
- LangChain and LangGraph orchestration with model-selected retrieval strategy.
- BGE-M3 dense retrieval plus Qdrant BM25 hybrid retrieval and fusion.
- Multi-query rewriting, cross-encoder reranking, corrective retrieval, and answer verification.
- Catalog-backed role and department filtering before evidence reaches the model.
- Structured approved-tool routing with deterministic citation and tool validation.
- Upstash Redis SHA-256 exact-match caching for verified repeated responses.
- A 176-page synthetic policy corpus and a 166-question RAG benchmark.
- A responsive professional HTML, CSS, and JavaScript banking policy console.

## Repository layout

```text
.
|-- takasecure_rag/        FastAPI and LangGraph RAG application
|-- frontend/              HTML, CSS, and JavaScript interface
|-- scripts/               dataset and policy-corpus generators
|-- notebooks/             RunPod fine-tuning notebook
|-- data/
|   |-- sft/               versioned train/validation/test data
|   `-- rag/
|       `-- v4/            current corpus, manifest, and benchmark
|-- artifacts/             local LoRA archives; large files ignored by Git
|-- docs/                  RAG setup and operating guide
|-- tests/                 automated tests
|-- pyproject.toml         Python dependencies and tooling
`-- .env.example           environment configuration template
```

## Generate the datasets

```bash
python scripts/generate_banking_sft.py
python scripts/generate_enterprise_rag_corpus.py
```

The SFT generator writes to `data/sft/`. The enterprise corpus generator writes
to `data/rag/v4/`.

## Fine-tune on RunPod

Open `notebooks/TakaSecure_RunPod_Finetuning.ipynb` after cloning the repository.
The notebook discovers the repository root and reads the split files from
`data/sft/`.

## Run the RAG API

Serve the downloaded adapter with vLLM, then run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
uvicorn takasecure_rag.main:app --host 0.0.0.0 --port 8080
```

Open http://localhost:8080 for the policy console. FastAPI documentation is
available at http://localhost:8080/docs.

See `docs/RAG_SYSTEM.md` for vLLM, Upstash, retrieval, and API instructions.

## Dataset split

- Training: 5,600 examples
- Validation: 700 examples
- Test: 700 examples

Policy families are isolated across splits. A policy family appearing in
training does not appear in validation or test, including scope variants.

## Limitations

- All policies, cases, institutions, identifiers, and operational rules are synthetic.
- The UI role selector demonstrates authorization policy; it is not user authentication.
- The corpus has not been reviewed by banking, legal, compliance, fraud, or security professionals.
- Automated checks validate structure and provenance consistency, not real-world policy suitability.
- The project must not be represented as Bangladesh Bank guidance or used for real banking decisions.
- Production deployment would require identity-provider authentication, server-derived roles, institution-approved documents, privacy controls, and domain-expert review.
