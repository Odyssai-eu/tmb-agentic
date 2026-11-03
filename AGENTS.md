# Writing tasks for TMB Agentic

This file is for whoever writes tasks: a person, or a coding agent asked to
"add a task that checks X". Follow it to the letter; `selftest` enforces most
of it.

## The one rule

**A task is valid only if its golden trajectory passes it.** Write the world,
the assertions and the golden together, then run:

```bash
tmb-agentic selftest path/to/tasks
```

`BROKEN` means the task is wrong, never the model. Fix the task until every
line says `OK`, before running any model on it.

## Task file

One JSON file per task. The file name (without `.json`) is the task id unless
`id` is set.

| Key | Required | Meaning |
|---|---|---|
| `prompt` | yes | The user message. State the expected final format if an assertion reads the final answer. |
| `world` | yes | `{"type": "fs" \| "db" \| "api", "init": {...}}`, see below. |
| `assertions` | yes | Closed predicates on the final state, the call log and the final message. |
| `golden` | yes | A reference trajectory that solves the task. |
| `tools` | no | Subset of the world's tools to expose. Default: all. |
| `budget` | no | `max_steps` (default 20), `max_tool_calls` (default 2 × steps), `wall_s` (default 1800). |
| `faults` | no | Faults injected on given calls, see below. |
| `forbidden_tools` | no | Tools that are offered but answer "permission denied". |
| `extra_tools` | no | Extra OpenAI function schemas the world does not implement (usually also forbidden). |
| `system` | no | Text appended to the uniform agent system prompt. |

### Worlds

- **fs**: `{"files": {"path/to/file": "content"}}`. Tools: `list_dir`,
  `read_file`, `write_file`, `delete_file`, `search`. Snapshot: `{"files": {...}}`.
- **db**: `{"tables": {"name": [row, ...]}}`. Tools: `db_tables`, `db_select`,
  `db_insert`, `db_update`, `db_delete` (`where` = equality on columns).
  Snapshot: `{"tables": {...}}`.
- **api**: `{"routes": {...}, "state": {...}}`. Tools: `http_get`, `http_post`,
  `http_delete`. Route kinds: `static` (fixed `payload`), `read_store`,
  `append` (`field` of the body appended to `store`), `remove`
  (`{param}` from the path removed from `store`). Snapshot: `{"state": {...}}`.

A new world is a subclass of `tmb_agentic.World` with `TOOLS`, `dispatch()` and
`snapshot()`, registered in `tmb_agentic.worlds.WORLDS`. It must stay in memory
and deterministic.

### Assertions

`path` is a dotted path into the snapshot (`tables.accounts`, `state.tickets`).
Use `key` when the last key contains dots or slashes (`"path": "files", "key": "app/config.yaml"`).

| kind | fields | passes when |
|---|---|---|
| `state_matches` | `path`, `key?`, `op`, `value` | `op` is `eq`, `set_eq` (lists, order ignored), `contains`, `contains_item`, `len_eq` |
| `tool_called_at_least` | `tool`, `n` | the tool was attempted at least `n` times (calls eaten by a fault count) |
| `tool_never_called` | `tool` | the tool was never attempted |
| `final_contains` | `value` | the final message contains `value` |
| `final_not_contains` | `value` | it does not |
| `final_number_equals` | `regex`, `path` | the last match of `regex` (group 1) equals the resolved value, or its length for a list |

Add `"flag": "<name>"` to an assertion to make its failure a named flag on the
trajectory (for example `constraint_violation` on a `tool_never_called`). A
flagged failure always fails the task.

**Check the effect, not the path.** Assert on the final state and on the
answer, not on the exact sequence of calls. A model that reaches the right
state by another valid route must pass.

### Faults

```json
{"on_tool": "db_update", "nth": 1, "kind": "timeout", "once": true}
```

The fault fires on the `nth` call of that tool, whatever happened before.
Kinds: `timeout`, `http_404`, `http_500`, `empty_result` (call not executed);
`malformed_json`, `truncated_result` (call executed, result corrupted);
`inject` (real result plus `payload` appended).

When a task has a fault, the golden must meet it too, and recover.

### Golden trajectory

```json
[
  {"calls": [{"tool": "read_file", "args": {"path": "VERSION"}}]},
  {"calls": [{"tool": "write_file", "args": {"path": "CHANGELOG.md", "content": "..."}}]},
  {"final": "Done. RESULT=3"}
]
```

Each `calls` step is one model turn (several calls = parallel calls). The
last step is the final answer.

## Checklist before running a model

1. `selftest` says `OK` for the task.
2. The prompt alone is enough to solve it: no hidden convention.
3. Every value an assertion checks is derivable from the world, and the
   expected answer format is stated in the prompt.
4. A wrong but plausible answer fails (add a `final_not_contains` for the
   obvious trap when there is one).
5. Faults fire on a call every reasonable solution makes.
6. The task does not copy an existing public benchmark task.

See `tmb_agentic/examples/` for one task per world. The Tasks tab of
`tmb-agentic ui` shows the same validity check for a whole folder.
