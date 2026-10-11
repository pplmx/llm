# Test Architecture

How tests are organized, how they behave on CPU-only hosts, and how CI splits
fast per-push feedback from the slow deep gate. See also
`.agents/skills/run-ci/SKILL.md` for the runnable commands and
[`Justfile`](../../Justfile) for the recipes.

## Core problem (why this design exists)

The repo has a large test surface (3000+ tests). CI used to run the **full
suite** (every marker) + coverage on GPU-less 3-OS runners on **every push**,
which:

- **timed out at 6 h** (the heavy/e2e/GPU tests never finished), so CI was red
  for months without anyone noticing the real failures underneath; and
- **failed on GPU-dependent tests** because GPU-less runners don't have GPUs —
  tests either failed outright or were skipped inconsistently.

Two performance pushes (fused penalty kernels, vectorized GAE, AWQ chunking)
shortened the full suite enough (6 h → ~15 min) that the **real** pre-existing
failures surfaced: 6 tests that were never reached before the timeout.

The architecture below makes three things true at once: per-push CI is *fast*
and *CPU-clean*, GPU-less hosts *skip* GPU tests instead of failing, and the
slow full-suite gate still runs (just not on every push).

## Marker taxonomy (`pytest.toml` `[tool.pytest.ini_options].markers`)

| Marker         | Meaning                                     | In per-push core? |
| -------------- | ------------------------------------------- | ----------------- |
| `quick`        | < 1 s unit test                             | ✅                |
| `slow`         | 1–30 s, mid-weight                          | ✅                |
| `heavy`        | > 30 s or resource-heavy (paged attn, dist) | ❌ (deep gate)    |
| `e2e`          | end-to-end workflow                         | ❌ (deep gate)    |
| `gpu`          | needs ≥ 1 usable GPU                        | ❌ auto-skip      |
| `multi_gpu`    | needs ≥ 2 usable GPUs                       | ❌ auto-skip      |
| `need_gpu(n)`  | needs exactly ≥ n usable GPUs               | ❌ auto-skip      |
| `full_cluster` | needs 8 GPUs                                | ❌ auto-skip      |

Markers are registered in `tests/conftest.py::pytest_configure` and the
per-marker counts live in AGENTS.md.

## GPU handling (the "how to run without a GPU" contract)

The GPU strategy is a strict fallback ladder. Each test falls into exactly one
bucket:

1. **GPU-only / too-slow-on-CPU** — marked `gpu` / `multi_gpu` / `need_gpu(n)` /
   `full_cluster`. These *require* GPU semantics or would be impractically slow
   on CPU (distributed `tensor_parallel`/`pipeline`/`tp_dp` e2e, DDP). Without
   enough GPUs they **skip** — a CPU-only host never wastes time running them.
2. **Fast on CPU** — unmarked / `quick` / `slow`. They run on CPU everywhere
   (and pick a GPU device when present for speed).
3. **Slow CPU-resource tests** — `heavy` (convergence, full task trians). Not
   GPU-dependent; excluded from the per-push core gate and run in the deep gate.

`tests/conftest.py::pytest_collection_modifyitems` is the **single** owner of
GPU availability. Using `tests.support.devices.cuda_device_count()` (visible
**and** enough free VRAM), it auto-adds a skip to every GPU-requiring marker
when the host is short of usable GPUs:

- `need_gpu(n)` → skip if `< n` (default `n=1`)
- `gpu` → skip if `< 1`
- `multi_gpu` → skip if `< 2`
- `full_cluster` → skip if `< 8`

So a CPU-only CI runner or laptop **skips** (never fails, never hangs) GPU
tests — including currently `need_gpu(4..8)` distributed/e2e tests that would
otherwise crawl or crash on CPU. The per-push core gate (`just test-fast` /
`just test-cov-fast`) also **excludes** all four GPU markers so a CPU host
never even collects them. GPU tests run via `just test` / `just test-gpu` on a
GPU host or self-hosted GPU runner; when GPUs are present, `just test-parallel`
spreads CPU-clean work across them with pytest-xdist.

### CUDA cached memory note

PyTorch's caching allocator keeps a process's GPU memory allocated for its
whole lifetime and only returns it to the driver at process exit. During a GPU
suite, `nvidia-smi` therefore shows *our* pytest process holding memory — that
is expected, not a leak; the memory is freed when the process exits. Running
GPU tests as a separate short `test-gpu` process (rather than inside the giant
full suite) shortens how long GPUs appear "reserved" on a shared box.

## Test profiles (`Justfile`)

| Recipe          | Marker selection                                    | Use                                          |
| --------------- | --------------------------------------------------- | -------------------------------------------- |
| `test-quick`    | `quick`                                             | < 1 min iterative smoke                      |
| `test-fast`     | `not heavy/e2e/gpu/multi_gpu/need_gpu/full_cluster` | CPU-clean daily dev gate (~7 min)            |
| `test-cov-fast` | same as `test-fast` + coverage                      | **per-push CI gate** (`--cov-fail-under=77`) |
| `test`          | all markers                                         | full suite on a GPU host                     |
| `test-gpu`      | `gpu OR multi_gpu OR need_gpu OR full_cluster`      | GPU-only, on a GPU host / self-hosted runner |
| `test-heavy`    | `heavy`                                             | slow resource tests (separate job)           |
| `test-e2e`      | `e2e`                                               | end-to-end workflows                         |
| `test-cov`      | all markers + coverage                              | authoritative deep gate (scheduled)          |

All recipes run `uv run --frozen`.

## CI split (`.github/workflows/ci.yml`)

- **Per-push** — `lint`, `doctest`, `build-and-test` (ubuntu/macos/windows):
  `build-and-test` runs **`just test-cov-fast`** — the CPU-clean core suite +
  coverage. Fast (minutes), green on GPU-less runners.
- **Scheduled / manual** — `test-full` (ubuntu): `just test-cov` (full suite +
  coverage). Heavy/e2e included; too slow for per-push, so it runs on the
  weekly schedule + `workflow_dispatch`.
- **GPU** — no CI runner has GPUs; GPU tests auto-skip there. A self-hosted GPU
  runner runs `just test` / `just test-gpu` (tracked in ROADMAP Tier 3).

### Why terminal width matters (CLI help tests)

Rich/typer render `--help` at the detected terminal width. On a non-tty or CI
runner that width can be narrow, truncating each option to a couple of
characters so the option *names* (`--model`, `--host`, …) disappear from the
output — breaking CLI-help assertions that pass locally. `tests/conftest.py`
has an autouse `_force_wide_terminal` fixture that sets `COLUMNS=120`/`LINES=40`
for every test, so CLI-help rendering is deterministic. This fixed 5 surfaced CI
failures and hardens the whole CLI-help family.

## Runtime optimization

- Full suite was 6 h (CI timeout) → now ~15 min (perf work).
- The per-push core gate (`test-cov-fast`) runs in ~7 min with coverage, so CI
  gives fast green/red on every push.
- GPU tests isolated into `test-gpu` so they only tax GPU hosts and can be
  scheduled independently.
