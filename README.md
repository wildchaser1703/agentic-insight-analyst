# Agentic Insight Analyst

A production-grade agentic system for synthesizing actionable insights from messy qualitative and quantitative data. Built for leadership decision-making with explicit reasoning traces and controlled autonomy.

---

## Problem Statement

Organizations collect vast amounts of feedback—surveys, reviews, support tickets, open-ended comments—yet struggle to translate this raw data into actionable insights. Traditional analysis pipelines are brittle: they follow fixed steps regardless of data characteristics, produce opaque outputs, and require constant manual intervention.

**Agentic Insight Synthesis** addresses this by introducing a system that:
- Interprets high-level goals ("summarize key pain points for leadership")
- Plans an appropriate analysis workflow based on data characteristics
- Dynamically selects tools (summarization, clustering, retrieval, recommendations)
- Iterates until goals are satisfied—not forever
- Produces structured, human-readable outputs with full reasoning transparency

This is not a demo. This is controlled autonomy with explicit guardrails.

---

## What Makes This System Agentic

This system implements a genuine agentic loop, not a linear prompt chain:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   GOAL → PLAN → ACT → OBSERVE → DECIDE → SYNTHESIZE    │
│     ▲                                        │          │
│     └────────────────────────────────────────┘          │
│                    (iterate)                            │
│                                                         │
│                        ↓                                │
│                      STOP                               │
│              (when criteria met)                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Key Agentic Properties

| Property | Implementation |
|----------|---------------|
| **Goal Interpretation** | Agent parses natural language objectives into structured analysis targets |
| **Explicit Planning** | Agent generates a step-by-step plan before execution, visible to users |
| **Dynamic Tool Selection** | Agent chooses tools based on current state, not a hardcoded sequence |
| **Iterative Execution** | Agent observes results and adjusts strategy mid-run |
| **Controlled Stopping** | Evaluator enforces termination on goal satisfaction or diminishing returns |
| **Reasoning Transparency** | Every decision is logged with human-readable justification |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         API Layer                             │
│                      (FastAPI /analyze)                       │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                      Agent Controller                         │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│   │   Planner   │ →  │   Executor  │ →  │  Evaluator  │      │
│   └─────────────┘    └─────────────┘    └─────────────┘      │
│          ↑                  │                  │              │
│          └──────────────────┴──────────────────┘              │
│                        (feedback loop)                        │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                       Tool Registry                           │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│   │  Summarize   │  │   Cluster    │  │   Retrieve   │       │
│   └──────────────┘  └──────────────┘  └──────────────┘       │
│   ┌──────────────┐  ┌──────────────┐                         │
│   │  Recommend   │  │   Analyze    │                         │
│   └──────────────┘  └──────────────┘                         │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                      Working Memory                           │
│        Plans | Observations | Intermediate Outputs            │
└──────────────────────────────────────────────────────────────┘
```

---

## Example Workflow

**Input:**
```json
{
  "goal": "Identify the top 3 customer pain points and recommend improvements",
  "data_path": "data/sample/sample_survey.csv"
}
```

**Agent Execution Trace:**

```
[STEP 1] Planning
  → Goal parsed: Extract pain points, prioritize, generate recommendations
  → Plan generated:
      1. Load and analyze data structure
      2. Cluster qualitative feedback into themes
      3. Summarize each cluster
      4. Cross-reference with quantitative scores
      5. Generate prioritized recommendations

[STEP 2] Tool Selection: analyze_data_structure
  → Reasoning: Need to understand data shape before processing
  → Result: 50 rows, 5 Likert columns, 1 open-text column

[STEP 3] Tool Selection: cluster_feedback
  → Reasoning: Open-text column detected, clustering will reveal themes
  → Result: 4 clusters identified (onboarding, pricing, support, features)

[STEP 4] Tool Selection: summarize_text
  → Reasoning: Each cluster needs a digestible summary
  → Result: Summaries generated for each cluster

[STEP 5] Evaluator Check
  → New insights: Yes (4 distinct themes found)
  → Goal progress: 60% (pain points identified, recommendations pending)
  → Decision: Continue

[STEP 6] Tool Selection: generate_recommendations
  → Reasoning: Themes and quantitative data sufficient for recommendations
  → Result: 3 prioritized recommendations with evidence

[STEP 7] Evaluator Check
  → Goal progress: 100%
  → Decision: Stop and synthesize

[OUTPUT] Final Report Generated
```

---

## What This System Does

✅ Interprets natural language analysis goals  
✅ Generates explicit, reviewable plans  
✅ Dynamically selects appropriate tools  
✅ Maintains transparent working memory  
✅ Stops when goals are satisfied  
✅ Produces structured, leadership-ready outputs  
✅ Logs all reasoning for auditability  

## What This System Does NOT Do

❌ Run indefinitely (strict iteration limits and stopping conditions)  
❌ Make autonomous decisions without transparency  
❌ Require specific LLM vendors (abstracted interfaces)  
❌ Hide intermediate reasoning (all steps are logged)  
❌ Execute arbitrary code (tools are predefined and sandboxed)  

---

## Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API key (or compatible LLM endpoint)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/agentic-insight-analyst.git
cd agentic-insight-analyst

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API key
```

### Running the Agent (CLI)

```bash
python scripts/run_agent.py \
  --goal "Summarize key pain points for leadership" \
  --data data/sample/sample_survey.csv
```

### Running the API

```bash
uvicorn api.main:app --reload --port 8000
```

**API Request:**
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Identify top customer concerns",
    "data_path": "data/sample/sample_survey.csv"
  }'
```

### Running Tests

```bash
pytest tests/ -v
```

---

## Configuration

All configuration is centralized in `src/config.py` and can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `gpt-4` | Model identifier |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature |
| `MAX_ITERATIONS` | `10` | Maximum agent loop iterations |
| `MIN_IMPROVEMENT_THRESHOLD` | `0.1` | Minimum progress to continue |

---

## Project Structure

```
agentic-insight-analyst/
├── README.md                 # This file
├── .env.example              # Environment template
├── requirements.txt          # Python dependencies
│
├── data/
│   └── sample/
│       └── sample_survey.csv # Demo data
│
├── src/
│   ├── agent.py              # Core agentic loop (planner + controller)
│   ├── tools.py              # Tool registry (callable functions)
│   ├── memory.py             # Working memory (plans, observations)
│   ├── prompts.py            # Structured system prompts
│   ├── evaluator.py          # Stopping conditions & quality checks
│   └── config.py             # Centralized configuration
│
├── api/
│   └── main.py               # FastAPI interface
│
├── scripts/
│   └── run_agent.py          # CLI entry point
│
└── tests/
    └── test_agent_flow.py    # Integration tests
```

---

## Limitations & Next Steps

### Current Limitations

- **Tool set is fixed**: Currently supports summarization, clustering, retrieval, and recommendations. Extending requires code changes.
- **Single-session memory**: Memory resets between runs. No persistent context.
- **LLM dependency**: Core reasoning requires LLM access. Offline mode not supported.
- **English only**: Prompts and analysis optimized for English text.

### Roadmap

1. **Tool extensibility**: Plugin architecture for custom tools
2. **Persistent memory**: Cross-session context with vector storage
3. **Multi-modal support**: Image and document analysis
4. **Human-in-the-loop**: Approval gates for high-stakes decisions
5. **Evaluation framework**: Automated benchmarking against ground truth

---

## Design Philosophy

This system embodies **controlled autonomy**:

- **Autonomy**: The agent makes real decisions about tool selection and execution order
- **Control**: Strict iteration limits, explicit stopping conditions, and full transparency

This is not a wrapper around prompts. This is not an infinite loop hoping for a good answer. This is a structured reasoning system with clear boundaries.

---

## License

MIT License. See LICENSE for details.

---

## Author

Built as a portfolio demonstration of production-grade agentic systems. Suitable for consulting engagements, enterprise pilots, or extension into domain-specific applications.