# Harnesses

A *harness* is any CLI that runs a language model non-interactively.
BenchKit treats each harness uniformly through an **adapter** that
translates `(model, prompt, system_prompt)` into a concrete CLI
invocation.

## Built-in harnesses

### Default family — `ohp`, `punkin`, `opencode`, `monopi`, `omp`

Any CLI following this shape:

```
BIN -p --no-session --model X @promptfile [--system-prompt FILE]
```

Prompt written to a temp file, passed with `@file` syntax so the OS
`ARG_MAX` limit (~256 KB on macOS) doesn't truncate long-context
prompts. System prompt — when supplied — goes through `--system-prompt
FILE`. Any unknown `--harness` name falls through to this adapter with
the name used as the executable, so custom CLIs following the shape work
out of the box.

### `claude` — Anthropic Claude Code

```
claude -p --model X --no-session-persistence --permission-mode bypassPermissions
       [--append-system-prompt-file FILE]
       < promptfile
```

Prompt via stdin. System prompt always uses the
`--append-system-prompt-file` form so heavy prompts (IFEval's ~3 KB
heavy condition) bypass ARG_MAX. `--permission-mode bypassPermissions`
suppresses the trust dialog for unattended runs.

### `codex` — OpenAI Codex

```
codex exec --skip-git-repo-check --ephemeral --color never
           -o LASTFILE -m MODEL -
           < promptfile
```

Codex's stdout is streamed chrome (banner, user echo, token counters);
the clean agent message is read from the `--output-last-message` file.
Empty LASTFILE signals auth failure or refusal — the adapter emits
`[ERROR codex] empty last-message; stderr=…`.

Codex has no dedicated system-prompt flag on `exec`, so system content
is prepended with delimiters:

```
[SYSTEM INSTRUCTIONS]
<system prompt text>

[USER REQUEST]
<user prompt text>
```

## Adapter interface

```python
@dataclass
class HarnessInvocation:
    cmd: list[str]                              # argv
    stdin: str | None = None                    # piped to subprocess stdin
    tempfiles: list[str] = []                   # cleaned up after run
    extract: Callable[[CompletedProcess[str]], str] = lambda p: p.stdout.strip()
```

`run_harness` is a generic executor: build the invocation, run
`subprocess.run(cmd, input=stdin, …)`, call `extract` on success, map
nonzero exits / timeouts / `FileNotFoundError` to `[ERROR …]`-prefixed
strings, clean up temp files in `finally`.

## Adding a new harness

Three steps:

1. **Write an adapter** in `src/benchkit_for_harnesses/harnesses/dispatch.py`:

   ```python
   def _build_myharness(
       model: str, prompt: str, system_prompt: str | None
   ) -> HarnessInvocation:
       prompt_path = _write_temp(prompt)
       cmd = ["myharness", "--run", "--model", model, "--input", prompt_path]
       tempfiles = [prompt_path]
       # ... handle system_prompt per the CLI's shape ...
       return HarnessInvocation(cmd=cmd, tempfiles=tempfiles)
   ```

   If the CLI's stdout needs post-processing, supply an `extract`:

   ```python
   def _extract(p: CompletedProcess[str]) -> str:
       return p.stdout.split("===RESPONSE===")[1].strip()
   return HarnessInvocation(cmd=cmd, extract=_extract, tempfiles=...)
   ```

2. **Wire it** in `build_invocation`:

   ```python
   if harness == "myharness":
       return _build_myharness(model, prompt, system_prompt)
   ```

3. **Test it** in `tests/test_harness_adapters.py` — assert the
   constructed command without invoking subprocess.

No changes to `run_harness`, the core loop, any runner, or the CLI needed.

## Error discipline

All failure paths return a response starting with `[ERROR …]`:

| Condition | Response prefix |
|---|---|
| Nonzero exit | `[ERROR rc=N] <stderr>\n---STDOUT---\n<stdout>` |
| Timeout | `[ERROR timeout=Ns] harness did not return within timeout` |
| Binary not in PATH | `[ERROR] Harness 'NAME' not found in PATH` |
| Codex empty last-message | `[ERROR codex] empty last-message; stderr=…` |
| Codex last-file unreadable | `[ERROR codex] could not read last-message file: …` |

Benchmark `eval_fn` implementations score `[ERROR …]`-prefixed strings
as incorrect. Operators see the exact failure reason in the archive
record's `response` field.

## Responder factory

Inside the kit, every harness is turned into a `SyncResponder` via:

```python
from benchkit_for_harnesses.responders import harness_responder
responder = harness_responder("claude", "sonnet", timeout_sec=120)
# responder(prompt, system) -> (response, latency_ms)
```

This is what `runner.py` and `ifeval/experiment.py` use internally.
External callers can reach for the same factory when they want to run
custom benchmarks through the unified substrate.
