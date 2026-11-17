# Agentic Insight Analyst

A bounded feedback-analysis agent with explicit tools, working memory, a stopping heuristic, and a local FastAPI interface. It uses TF-IDF clustering and score summaries alongside an LLM planner and report writer.

## Run (Python 3.11 or 3.12)

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
# Set OPENAI_API_KEY in your shell for the optional live run.
python scripts/run_agent.py --goal "Identify the main customer concerns" --data data/sample/sample_survey.csv
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Tests use mock model responses and local sample data; no key or network is needed. `/docs` describes the API. Submit exactly one CSV `data_path` or `raw_text` (one feedback response per line). Raw text is converted to a temporary CSV and removed after the run.

## Behavior and boundaries

Provider failures, malformed model responses, invalid summaries and failed tools do not become successful results. Requests have bounded provider timeouts/retries and an iteration limit. An output can be structurally valid without being factually correct: `success` describes execution/validation, not independently evaluated analytical quality. The stopping score is a heuristic, not a calibrated confidence or proof of goal completion.

The synchronous endpoint runs in FastAPI's worker thread pool. The optional background-job endpoint retains at most 100 jobs **in one process**; completed jobs are evicted at capacity and all jobs disappear on restart. Keep this a local, single-worker service: it accepts local file paths and has no authentication or durable queue. Persistence, distributed workers, and a generic orchestration framework are intentionally out of scope.

The test suite covers tool execution, memory, bounded control flow, provider failures, malformed/invalid output, raw-text cleanup and failed-job status. No live model quality or production reliability benchmark is claimed. Historical design notes below describe the original experiment; this README defines the maintained interface.

## CI and tested dependencies

`requirements.lock.txt` records the isolated Linux/Python 3.12 test environment. CI installs that lock, runs `pip check`, and executes the offline suite on pushes and pull requests; it does not deploy. For exact replay, use the lock in place of `requirements.txt` (Depth still needs the CPU Torch install first), then `pip install --no-deps -e .` for packaged repos. No credentials are required by CI.
