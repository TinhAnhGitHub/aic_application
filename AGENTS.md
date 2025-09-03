# Repository Guidelines

## Project Structure & Module Organization
- `app/`: FastAPI app and domain logic.
  - `app/api/`: Route handlers (e.g., `health`, `search`).
  - `app/controller/`: Request orchestration.
  - `app/services/`: Model/search services (Unilm and related).
  - `app/repository/`: Mongo, Milvus, Elasticsearch data access.
  - `app/schemas/`: Pydantic v2 request/response models.
  - `app/core/`: Config, DI, logging.
  - `app/migration/`: Typer CLI and ingestion helpers.
- `docker_services/`: Local Elasticsearch, MongoDB, Milvus stack.
- `script/`: Convenience scripts for ingestion and initialization.

## Build, Test, and Development Commands
- Setup deps: `uv sync`
- Run API (dev): `uv run uvicorn app.main:app --reload`
- Start infra: `docker compose -f docker_services/docker-compose.yml up -d`
- Initialize Milvus: `uv run python app/migration/cli.py init --keyframe-embedding-path <.npy> --caption-embedding-path <.npy>`
- Ingest embeddings: `uv run python app/migration/cli.py ingest_embedding --keyframes-dir <dir> --captions-dir <dir> --keyframe-embedding-path <.npy> --caption-embedding-path <.npy>`
- Ingest metadata: `uv run python app/migration/cli.py ingest_meta --keyframes-dir <dir> --captions-dir <dir>`

## Coding Style & Naming Conventions
- Python 3.10; 4-space indentation; PEP 8; type hints required.
- Modules/vars/functions: `snake_case`; classes: `PascalCase`.
- Keep Pydantic models in `app/schemas`; ensure API responses are typed.
- Place I/O in repository layer; keep controllers thin and composable.

## Testing Guidelines
- Current: no test suite. Add pytest tests under `tests/` mirroring `app/`.
- Naming: `tests/test_<module>.py`; test functions `test_*`.
- Focus: controllers, repositories (with fakes), and service boundaries.
- Run (once added): `uv run pytest -q`

## Commit & Pull Request Guidelines
- Commits: concise, imperative mood (e.g., "add search controller"); group related changes.
- Prefer scopes: `feat`, `fix`, `refactor`, `chore`, `docs` when applicable.
- PRs: include purpose, changes, how to run, and screenshots/curl of key endpoints. Link issues. Note migration/ingestion impacts.

## Security & Configuration Tips
- Config via `.env`; see `app/core/config.py` for keys (`mongo_uri`, `es_hosts`, `milvus_uri`, etc.). Do not commit secrets or absolute local paths.
- CORS is permissive for dev; restrict in production.
- Ensure `docker_services` are healthy before ingestion and search.

