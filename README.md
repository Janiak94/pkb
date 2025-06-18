# Personal Knowledge Base (PKB)

Simple RAG tool for answering questions based on my personal files
using langchain, huggingface, and llama-cpp.

The project is split into two parts:
1. Ingestion of files that are used as context for prompts.
2. Ask question to the LLM.

The project is still in its early stages and an MVP.

## Ingest context files

Files can be ingested with the script

```bash
uv run pkb ingest path/to/files/*
```

Ingested file metadata will be stored in a cache to make sure
the latest version of a file is ingested and does not need to be
ingested multiple times.

## Ask questions

```bash
uv run pkb ask "What type of database if used by the pkb project?"
```

### Cache

Files and artifacts are cached under `$HOME/.cache/pkb`.