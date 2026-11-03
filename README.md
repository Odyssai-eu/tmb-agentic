# TMB Agentic

Deterministic worlds and a mechanical verdict for tool-using LLM agents.

TMB Agentic runs the agent loop itself (call the model, execute its tool
calls against an in-memory world, send the results back, repeat) and judges
the final state with closed predicates. No LLM judge, no network besides
the model endpoint, standard library only (Python 3.9+).

It ships the engine and three example tasks. **It ships no benchmark**: you
write the tasks that matter to you.

## Install

From a checkout of this repository:

```bash
pip install .
```

No dependencies besides Python 3.9+.

## Web interface

```bash
tmb-agentic ui
```

Opens http://127.0.0.1:8790 in your browser. Everything is done from there:

1. **Endpoint**: base URL, model, API key. **Test tool calls** checks that
   the endpoint returns tool calls the loop can use.
2. **Tasks**: the task folder (the bundled examples by default), each task
   with its prompt, assertions and validity.
3. **Run**: pick tasks, repetitions and parallel runs, then follow progress.
4. **History**: every past run with its results. Open a run, then any
   repetition, to see the full trajectory turn by turn. **Export** gives a ZIP
   of one run, **Export all** of every run.

The interface is in English and Simplified Chinese. The server listens on
127.0.0.1 only. The API key stays in the server's memory and is never written
to disk; leave it empty to use `$OPENAI_API_KEY`. Stop the server with Ctrl-C.

| Tool-call check | Meaning |
|---|---|
| Native | Structured `tool_calls` with parsable arguments. Ready. |
| Text | The model writes its calls as text. Runs use the text adapter; add its dialect to `parse_text_calls` if needed. |
| No usable tool call | Often a serving issue (chat template, tool parser), not the model. |
| Unreachable | Transport error. Says nothing about the model. |

## Command line

The same steps without the browser, for scripts and CI:

```bash
tmb-agentic gate --base http://localhost:8000/v1 --model my-model
tmb-agentic selftest path/to/tasks
tmb-agentic run path/to/tasks --base http://localhost:8000/v1 --model my-model --reps 4 --concurrency 4
```

Without a task path, `selftest` and `run` use the bundled examples. The API
key is read from `$OPENAI_API_KEY` (change with `--api-key-env`). Vendor
fields go through `--extra-body '{"reasoning_effort": "low"}'`.

## Runs and exports

Runs from the interface and from the command line go to the same place,
`~/.tmb-agentic/runs/<started>-<model>/` (change with `$TMB_AGENTIC_HOME` or
`--home`), so they all appear in History:

- `meta.json`: model, endpoint, settings, status (`running`, `done`,
  `stopped`, `incomplete` when the process died, `error`).
- `progress.ndjson`: one line per finished repetition.
- `<task>.r<k>.json`: one full trajectory: every turn, every call and result
  as the model saw it, faults fired, final world state, assertion results.
- `summary.json`: per task, `passed` / `runs`, `outcomes` (`stopped`,
  `tool_loop`, `budget_exhausted`, `wall_timeout`, `error`) and `flags`.

A repetition passes when the model stopped on its own, every assertion holds
and no flag is raised.

An export is a ZIP with the raw run folders, `report.csv` (one row per run
and task) and `report.html` (a readable summary with the failed repetitions).

## Writing tasks

See [AGENTS.md](AGENTS.md). In short: a JSON file with a prompt, a world,
assertions on the final state, and a golden trajectory that `selftest`
replays.

## Tests

```bash
python -m unittest discover -s tests
```

## What results mean

A pass rate is a property of your task set, your serving stack and your
model together. It is not comparable with any other benchmark or task set.
A pass rate on your own tasks is not a TMB Bench score: do not present it
as one, nor as a score of any other benchmark.

## License

Copyright 2026 Monoklbr OÜ - Sophie Dupont.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).
