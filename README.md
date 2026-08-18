# agent-core

**A tool-using agent framework built from scratch, with a benchmark that measures which control strategy actually earns its cost instead of assuming one does.**

[![CI](https://github.com/Aphelion07/agent-core/actions/workflows/ci.yml/badge.svg)](https://github.com/Aphelion07/agent-core/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Coverage 97%](https://img.shields.io/badge/coverage-97%25-brightgreen)](#testing)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-blue)](http://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Why not another LangChain wrapper

Most agent projects import a framework, wire up a couple of tools, and call it done. What actually varies between agent frameworks is the control loop: does the model re-decide its next move after every observation, commit to a plan up front, or check its own answer before handing it back? Every framework picks one and rarely says why.

So this project has no framework dependency. It implements three control strategies from the primitives up (a tool registry, an LLM-backend contract, a context trimmer, a run trace) and then runs all three against the same labelled task suite to find out whether the extra machinery in the more elaborate strategies is actually buying anything.

```
[react]        14 tasks, 3 runs each: 100% success, 2.2 LLM calls/task,  1687 tokens/task, 4.2s/task
[reflexion]    14 tasks, 3 runs each: 100% success, 3.2 LLM calls/task,  2247 tokens/task, 5.6s/task
[plan_execute] 14 tasks, 3 runs each: 100% success, 3.6 LLM calls/task,  2570 tokens/task, 9.0s/task
```

Same accuracy, up to 62% more LLM calls. That gap is the finding.

---

## The three strategies

| | control flow | designed weakness |
|---|---|---|
| **ReAct** | One LLM call per step, re-deciding the next action from scratch after every tool result. | Never commits to a plan, so it can also wander - re-checking something it already knows, or drifting off-task over a long run. |
| **Plan-and-execute** | One call produces an ordered plan; each step then gets exactly one execution turn, in order, no matter what earlier steps returned. | A plan built on a wrong assumption executes to completion anyway - nothing re-plans after step three's tool result contradicts step one. |
| **Reflexion** | ReAct, but the last call in the step budget is reserved for a self-critique pass: show the model its own candidate answer and ask it to confirm or correct it. | The reserved call is one fewer step available for actual tool use, so on a task that needs every step of the budget, reflexion can run out of room where plain ReAct would not. |

All three share the same tool registry, the same context-budget trimmer, and the same `RunTrace` instrumentation (LLM calls, tool calls, tokens, wall time, step-limit hits) - see [Design decisions](#design-decisions) for how.

---

## The task suite

14 tasks against a small fixed tool set: `calculator`, `catalog_lookup`, `unit_convert`, `date_diff`, and `get_weather` - the last one is a pure distractor, never the right tool for anything, included to see whether a strategy reaches for it anyway.

Twelve tasks are solvable and get scored against the right number (`contains_number`, tolerant to rounding but not to a wrong tool result). Two are deliberately **not** solvable with the tools on offer - asking for a price in a currency there's no converter for, and for stock data the catalog doesn't carry. The correct behaviour there is to say so, not to guess; a verifier checking only for numeric accuracy would score a confidently wrong currency conversion as a pass.

---

## Findings

All figures: `qwen3:8b` via Ollama, RTX 5080, `max_steps=8`, 3 runs per task per strategy (126 runs total). Reproduce with `python benchmarks/bench.py --runs 3`.

### 1. On this suite, all three strategies are equally correct

42/42 runs succeeded for every strategy: every solvable task got the right number, and every refusal on the two unsolvable tasks was genuine (no strategy ever fabricated a currency rate or a stock count). `hit_step_limit_rate` was 0.0 across the board too - an 8-step budget was never once exhausted, and `distractor_call_rate` was 0.0 - `get_weather` was never called, by any strategy, on any task.

That's a real result, not a placeholder: this task suite, at this model size, doesn't create a failure condition sharp enough to separate the strategies on correctness. The honest reading is in [Limitations](#limitations), not buried here.

### 2. So the strategies separate entirely on cost

| strategy | success | LLM calls/task | tool calls/task | tokens/task | wall time/task |
|---|---|---|---|---|---|
| react | 100% | 2.24 | 1.40 | 1687 | 4.2s |
| reflexion | 100% | 3.21 (+44%) | 1.38 | 2247 (+33%) | 5.6s (+33%) |
| plan_execute | 100% | 3.62 (+62%) | 1.40 | 2570 (+52%) | 9.0s (+113%) |

Tool calls per task are within noise of each other across all three (1.38-1.40) - the cost difference is pure LLM-orchestration overhead, not extra tool use. It's structural: plan-and-execute always pays for a planning call and a dedicated final-answer call on top of execution, and reflexion always pays for one critique call, regardless of whether the task needed either. ReAct pays for neither, because it only calls the model again when there's a reason to.

Reflexion's self-critique round never once changed an answer in this run (0 corrections across 42 calls) - every candidate ReAct already produced was already right. That's the same 33% token premium either way: on tasks this size, the safety net cost real tokens and caught nothing.

### 3. A verifier bug undercounted refusals by up to 3x

The first pass at scoring the two unsolvable tasks used a keyword list (`"cannot"`, `"unable"`, `"don't have"`, ...) that missed how qwen3 actually phrases a refusal: *"the available tools do not **support** converting USD to JPY"*, *"the stock quantity **could not be retrieved**"*. Neither contains any of the original markers. The first run reported `unsolvable_refusal_rate` of 0.67 (react), 0.50 (reflexion), and 0.33 (plan_execute) - a plausible-looking result that plan-and-execute's no-replanning design was hurting it exactly where you'd expect.

Reading the raw transcripts showed all 18 answers across all three strategies were genuine, correct refusals; the verifier just didn't recognise the phrasing. Broadened the marker list (`"not support"`, `"could not be"`, `"isn't available"`, ...) and re-scored the stored transcripts without re-running the model - refusal rate for every strategy went to 1.0. `benchmarks/tasks.py` keeps the current list, plus the comment explaining why it grew.

### 4. A transient timeout cost the first attempt the whole run

The first full benchmark attempt died about ninety seconds in on `httpx.ReadTimeout` - a single slow response from the local Ollama daemon - taking every already-computed result down with it, since the script only wrote output once at the very end. Fixed two ways: `OllamaBackend` now retries a timed-out or dropped connection with exponential backoff (`max_retries=2`, configurable), and `benchmarks/bench.py` checkpoints the accumulated results to disk after every single run, not just at the end. The second attempt hit no further errors, but a benchmark that can lose an hour of GPU time to one dropped connection was worth not shipping.

---

## Quickstart

```bash
git clone https://github.com/Aphelion07/agent-core
cd agent-core
uv venv && uv pip install -e ".[dev]"
```

Run one task through one strategy against a local Ollama model:

```bash
ollama pull qwen3:8b
agent-core "What is (12 * 7) + (144 / 12)?" --strategy react --model qwen3:8b
```

Or point it at any OpenAI-compatible endpoint - including [`llm-gateway`](https://github.com/Aphelion07/llm-gateway) from this same portfolio, which adds caching and provider failover in front of whatever it proxies to:

```bash
agent-core "..." --backend openai_compatible --base-url http://localhost:8000/v1 --model qwen3:8b
```

Run the full benchmark:

```bash
python benchmarks/bench.py --model qwen3:8b --runs 3
```

Or with no model and no network at all - exercises every strategy's full control flow (including the planning and critique tool calls) against a scripted backend, which is what CI's smoke job runs:

```bash
python benchmarks/bench.py --backend fake --runs 1
```

---

## Design decisions

**Tools are generic over their own argument model.** `Tool[ModelT]` rather than `Tool` taking a bare `BaseModel`, so a tool's `func` can declare the concrete pydantic subclass it actually receives instead of widening every tool to the same base type. The schema advertised to the model comes straight from `parameters.model_json_schema()` - there is exactly one place a tool's argument shape is defined.

**One JSON schema serves both backends.** Ollama's native tool-calling API and the OpenAI-compatible wire format both accept the same `{"type": "function", "function": {...}}` shape, so `Tool.schema()` doesn't need a backend-specific variant. What differs between the two backends is smaller than it looks: Ollama returns tool-call arguments as a parsed object, OpenAI-compatible endpoints return them as a JSON string that needs decoding.

**Strategies communicate control decisions through synthetic tools, not parsed text.** Plan-and-execute's plan and reflexion's verdict are both extracted from a forced-looking tool call (`submit_plan`, `submit_verdict`) rather than parsed out of free-form text with a regex. When the model answers in plain text instead of calling the tool, both strategies degrade gracefully - plan-and-execute falls back to a single-step plan, reflexion just keeps its ReAct candidate - rather than crashing on a parse failure.

**The step budget counts LLM calls, not tool calls.** One response can request several tool calls at once without spending extra budget, since tool execution is local and cheap; what's actually scarce is round trips to the model.

**`FakeBackend` takes either a fixed script or a callable.** The fixed-list form is for strategy tests that just need a specific sequence of responses; the callable form lets a test inspect the full conversation so far and assert on it - e.g. that a failed tool call's error message actually reached the model as an observation before it tried again.

---

## Testing

```bash
pytest -q --cov=agent_core --cov-report=term-missing
ruff check . && ruff format --check . && mypy
```

51 tests, 97% coverage, mypy `strict`. Nothing touches the network: both real backends are mocked at the HTTP layer with `respx` (including a test that a scripted timeout gets retried, and one that it eventually gives up), and every strategy test runs against `FakeBackend`.

The failure paths are asserted, not assumed: a tool that raises becomes an observation the model can react to rather than an exception that unwinds the run (`test_failed_tool_call_is_visible_to_the_model` checks the error text actually reaches the next LLM call), and a strategy that never gets a final answer within its budget reports `hit_step_limit` rather than returning an empty string that would silently score as wrong for the wrong reason.

---

## Limitations

- **The task suite didn't stress-test the designed weaknesses.** Plan-and-execute's no-replanning problem and reflexion's step-budget tension are real properties of the code (see [The three strategies](#the-three-strategies)), but none of the 14 tasks contains a tool result that contradicts an earlier assumption or a task hard enough to exhaust an 8-step budget. Proving those weaknesses actually costs accuracy would need adversarial tasks: a tool that returns a plausible-but-wrong intermediate value, or a task that only resolves after a strategy notices something upstream was wrong.
- **One model, one hardware target.** All findings are `qwen3:8b` on one RTX 5080. A weaker model would likely show real accuracy gaps between the strategies where this one shows none; a stronger one might close even the cost gap if it reliably answers in one call regardless of strategy.
- **14 tasks is a small suite.** Enough to see a 100% vs 100% tie clearly, not enough to bootstrap a meaningful confidence interval when every strategy is at the same extreme.
- **No parallel tool calls.** A response requesting three tool calls at once executes them sequentially; nothing here measures whether concurrent execution would change the cost picture.
- **Reflexion's critique is one round, not iterative.** A real Reflexion-style loop can critique-and-retry multiple times; this implementation spends exactly one reserved call and stops, which is why it never actually corrected an answer in 42 runs - there was no second chance for it to matter.

---

## License

MIT, see [LICENSE](LICENSE).
