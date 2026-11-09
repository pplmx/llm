"""Docs must not name config keys the schema does not have.

This repo has already paid for this class of drift twice:

* `training.learning_rate` appeared in **eight** shipped YAML configs (the real
  key is `lr`), so the intended learning rates were silently replaced by the
  1e-3 default — a 50x error for ``sft_alpaca.yaml``.
* The same typo then lived on in the tutorials and in an API doc example that
  was copy-paste executable and raised ``AttributeError``.

Fixing the YAML files did not fix the prose. This test scans every fenced
``yaml`` block in ``docs/`` and every ``config.<section>.<field>`` Python
access, and requires the names to exist in the real schema — so a rename has to
be followed through the documentation.
"""

from __future__ import annotations

import functools
import pathlib
import re
from typing import ClassVar

import pytest

from llm.serving.config import ServingConfig
from llm.training.core.config import Config

_DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"
# Guard against the silent-vacuous failure this file was written to avoid: a
# wrong parents[] index pointed at a non-existent directory, so every scan
# iterated ZERO documents and passed unconditionally.
assert _DOCS.is_dir(), f"docs directory not found at {_DOCS}"


def _training_schema() -> dict[str, set[str]]:
    schema = {"<top>": set(Config.model_fields)}
    for section, field in Config.model_fields.items():
        if hasattr(field.annotation, "model_fields"):
            schema[section] = set(field.annotation.model_fields)
    return schema


def _all_docs() -> list[pathlib.Path]:
    """Markdown files whose content is enforced.

    ``docs/superpowers/`` holds historical design records for modules that were
    proposed and either renamed or never shipped; they are not instructions a
    reader follows, so holding them to today's API would only produce noise.
    """
    return sorted(p for p in _DOCS.rglob("*.md") if "superpowers" not in p.parts)


def _yaml_blocks(text: str) -> list[str]:
    return re.findall(r"```ya?ml\n(.*?)```", text, re.S)


def test_no_doc_yaml_block_uses_an_unknown_training_key() -> None:
    """Every key inside a documented YAML block must exist in the schema.

    A doc block is what users copy, so a stale key here reproduces the exact
    silent-default bug the runtime guard now prevents for real config files.
    """
    schema = _training_schema()
    serving_fields = set(ServingConfig.model_fields)
    # Sections that belong to a different schema (Kubernetes, Prometheus, DPO's
    # descriptive pseudo-YAML, ...) are not training config; a block is only
    # checked when its top-level sections are recognised.
    problems: list[str] = []
    for path in _all_docs():
        for block in _yaml_blocks(path.read_text(encoding="utf-8", errors="ignore")):
            section: str | None = None
            for line in block.split("\n"):
                top = re.match(r"^([a-z_][a-z0-9_]*):\s*$", line)
                if top:
                    section = top.group(1)
                    continue
                key = re.match(r"^ {2}([a-z_][a-z0-9_]*):", line)
                if not key or section is None:
                    continue
                pool = schema.get(section) or serving_fields
                if pool and key.group(1) not in pool:
                    problems.append(f"{path.relative_to(_DOCS.parent)}: {section}.{key.group(1)}")
    assert not problems, "documented YAML keys missing from the schema:\n  " + "\n  ".join(problems)


def test_no_doc_attribute_access_uses_an_unknown_config_field() -> None:
    """``config.<section>.<field>`` in a doc snippet must be a real attribute."""
    schema = _training_schema()
    problems: list[str] = []
    for path in _all_docs():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"\bconfig\.([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)", text):
            section, field_name = match.group(1), match.group(2)
            pool = schema.get(section) or set(ServingConfig.model_fields)
            if pool and field_name not in pool:
                problems.append(f"{path.relative_to(_DOCS.parent)}: config.{section}.{field_name}")
    assert not problems, "documented config attribute accesses that do not exist:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("stale", ["learning_rate"])
def test_the_known_rename_stays_gone(stale: str) -> None:
    """``learning_rate`` was renamed to ``lr``; no doc may reintroduce it.

    The generic scans above cover this, but an explicit case documents WHY the
    scans exist and gives a readable failure if the name comes back.
    """
    offenders = [str(p.relative_to(_DOCS.parent)) for p in _all_docs() if stale in p.read_text(errors="ignore")]
    assert not offenders, f"docs still mention the removed field {stale!r}: {offenders}"


# --- Documented Python imports must resolve ---


def _python_blocks() -> list[tuple[pathlib.Path, str]]:
    blocks: list[tuple[pathlib.Path, str]] = []
    for path in _all_docs():
        text = path.read_text(encoding="utf-8", errors="ignore")
        blocks.extend((path, match.group(1)) for match in re.finditer(r"```python\n(.*?)```", text, re.S))
    return blocks


def test_documented_llm_imports_resolve() -> None:
    """Every ``from llm.<mod> import <name>`` in a doc block must exist.

    Docs are executable instructions; an import of a symbol the package no
    longer exports sends the reader straight to ``ImportError``. This caught
    ``from llm.tokenization import load_tokenizer`` (the factory is
    ``TokenizerFactory`` now) and ``DecoderModel.from_pretrained`` in the
    evaluation guide.
    """
    import importlib

    problems: list[str] = []
    for path, block in _python_blocks():
        for match in re.finditer(r"^from (llm[\w.]*) import ([\w, ]+)$", block, re.M):
            module_name, names = match.group(1), match.group(2)
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                problems.append(f"{path.relative_to(_DOCS.parent)}: module {module_name} does not exist")
                continue
            problems.extend(
                f"{path.relative_to(_DOCS.parent)}: {module_name}.{name} is not exported"
                for name in (n.strip() for n in names.split(","))
                if name and not hasattr(module, name)
            )
    assert not problems, "documented imports that do not resolve:\n  " + "\n  ".join(problems)


# --- Documented CLI invocations must match the real commands and flags ---

#: ``(llm argv prefix)`` -> nothing; the flag set is read from ``--help`` once.
_CLI_TIMEOUT = 120


@functools.cache
def _cli_flags(prefix: tuple[str, ...]) -> set[str]:
    """Flags accepted by ``llm <prefix>``, read from its real ``--help``.

    ``lru_cache`` is load-bearing, not decoration: each call spawns
    ``uv run llm ... --help`` (seconds of interpreter start-up), and the caller
    below looks up the same prefix once per documented command line. Without the
    cache a doc with ~100 annotated lines issued several hundred subprocesses and
    the test took 43s — by far the slowest in the CPU gate.
    """
    import os
    import subprocess

    # Rich wraps help text to the terminal width and TRUNCATES an option name
    # that does not fit (``--config-path`` renders as ``--con…``), which made
    # every documented long flag look nonexistent.
    #
    # The width is not simply COLUMNS. typer sets
    # ``rich_utils.FORCE_TERMINAL = True`` when ``GITHUB_ACTIONS`` is exported
    # (so CI renders colours); a forced terminal with no real tty makes rich
    # query the terminal size, fall back to a narrow default, and truncate.
    # Pinning COLUMNS alone fixed nothing on CI — TERMINAL_WIDTH does set
    # typer's MAX_WIDTH, but it is read at import time, so it has to be passed
    # to the SUBPROCESS that imports typer rather than set in this process.
    env = {
        **os.environ,
        "COLUMNS": "200",
        "LINES": "50",
        "TERMINAL_WIDTH": "200",
        # Drop the CI colour trigger so rich does not force a terminal here.
        "GITHUB_ACTIONS": "",
        "FORCE_COLOR": "",
    }
    result = subprocess.run(  # noqa: S603 — argv is built from internal constants, not user input
        ["uv", "run", "--frozen", "llm", *prefix, "--help"],  # noqa: S607 — `uv` comes from PATH by design
        capture_output=True,
        text=True,
        timeout=_CLI_TIMEOUT,
        check=False,
        env=env,
    )
    assert result.returncode == 0, f"`llm {' '.join(prefix)} --help` failed:\n{result.stderr[-800:]}"
    return set(re.findall(r"(--[a-z][a-z0-9-]+)", result.stdout))


#: ``command -> subcommands`` for the commands that take one. ``quantize`` is
#: the only two-level command; a doc line without the subcommand would reference
#: flags that live one level down and never reach them.
_SUBCOMMANDS = {"quantize": ["gptq", "fp8"]}


def _documented_invocations() -> list[tuple[pathlib.Path, str]]:
    """Every ``llm <command> ...`` occurrence in a live doc, with its line."""
    found: list[tuple[pathlib.Path, str]] = []
    for path in _all_docs():
        for line in path.read_text(encoding="utf-8", errors="ignore").split("\n"):
            if not re.search(r"\bllm\s+(train|serve|quantize|prune|decompose|migrate-ckpt)\b", line):
                continue
            # Prose that EXPLAINS a flag does not exist ("`llm train` has no
            # --scheduler-type") names a flag without invoking it. Only lines
            # that look like an invocation are checked.
            if re.search(r"没有|不存在|不支持|has no|does not have|is not a", line):
                continue
            found.append((path, line))
    return found


def test_documented_cli_flags_exist() -> None:
    """A documented flag must be accepted by the command it is written for.

    Docs are instructions: an invented or renamed flag sends the reader to a
    Typer "No such option" error. Reading the flag sets from ``--help`` rather
    than from source keeps the test honest about what the CLI really accepts,
    including aliases Typer synthesises.
    """
    real: dict[tuple[str, ...], set[str]] = {}
    problems: list[str] = []
    for path, line in _documented_invocations():
        command = re.search(r"\bllm\s+(train|serve|quantize|prune|decompose|migrate-ckpt)", line).group(1)
        prefix: tuple[str, ...] = (command,)
        if command in _SUBCOMMANDS:
            # Pick the subcommand this line names, if any; otherwise the flags
            # are compared against the union of all its subcommands.
            named = [s for s in _SUBCOMMANDS[command] if re.search(rf"\bllm\s+{command}\s+{s}\b", line)]
            if named:
                prefix = (command, named[0])
        if prefix not in real:
            real[prefix] = _cli_flags(prefix)
        if command in _SUBCOMMANDS and len(prefix) == 1:
            allowed = set().union(*(_cli_flags((command, s)) for s in _SUBCOMMANDS[command]))
        else:
            allowed = real[prefix]
        for flag in re.findall(r"(--[a-z][a-z0-9-]+)", line):
            if flag == "--help":
                continue
            if flag not in allowed:
                problems.append(
                    f"{path.relative_to(_DOCS.parent)}: `llm {' '.join(prefix)}` has no {flag}\n    {line.strip()}"
                )
    assert not problems, "documented CLI flags that do not exist:\n  " + "\n  ".join(sorted(set(problems)))


# --- Documented CALLS must match the real signatures ---


def _python_blocks() -> list[tuple[pathlib.Path, str]]:
    return [
        (path, block)
        for path in _all_docs()
        for block in re.findall(r"```python\n(.*?)```", path.read_text(errors="ignore"), re.S)
    ]


def test_documented_calls_use_real_keyword_arguments() -> None:
    """A documented call must not pass a keyword the function does not accept.

    Docs are executable instructions. ``apply_peft(model, rank=8)`` reads fine
    and raises ``TypeError`` the moment a reader runs it — the kwargs belong to
    ``peft_kwargs``, not to the entry point. Import resolution (the test above)
    cannot see this: the symbol exists, the call shape does not.
    """
    import ast
    import importlib
    import inspect

    problems: list[str] = []
    checked = 0
    for path, block in _python_blocks():
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue  # partial snippets / illustrative fragments
        local: dict[str, object] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("llm"):
                try:
                    module = importlib.import_module(node.module)
                except ModuleNotFoundError:
                    continue
                for alias in node.names:
                    target = getattr(module, alias.name, None)
                    if callable(target):
                        local[alias.asname or alias.name] = target
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            function = local.get(node.func.id)
            if function is None:
                continue
            checked += 1
            try:
                signature = inspect.signature(function)  # type: ignore[arg-type]
            except ValueError, TypeError:
                continue
            parameters = signature.parameters
            if any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
                continue  # accepts **kwargs by design
            problems.extend(
                f"{path.relative_to(_DOCS.parent)}: {node.func.id}(..., {keyword.arg}=...) "
                f"— not a parameter of {node.func.id}"
                for keyword in node.keywords
                if keyword.arg is not None and keyword.arg not in parameters
            )

    # A vacuity floor: the scan found calls but resolved none (e.g. the import
    # regex broke), which would pass silently otherwise.
    assert checked > 50, f"only {checked} documented calls were resolved — the scan probably broke"
    assert not problems, "documented calls that would raise TypeError:\n  " + "\n  ".join(problems)


# --- Docs navigation must resolve: relative links and referenced files ---


def test_relative_links_in_docs_resolve() -> None:
    """A broken relative link sends the reader to a 404.

    Reworking the docs tree silently breaks links, and `mkdocs --strict` does not
    catch a target that is a valid project path but a wrong relative path (nor
    links into non-markdown files it leaves alone).
    """
    problems: list[str] = []
    for path in _all_docs():
        for lineno, line in enumerate(path.read_text(errors="ignore").split("\n"), 1):
            for match in re.finditer(r"\]\(([^)#][^)]*?)(?:#[^)]*)?\)", line):
                target = match.group(1).strip()
                if target.startswith(("http://", "https://", "mailto:", "...")):
                    continue
                if not (path.parent / target).resolve().exists():
                    problems.append(f"{path.relative_to(_DOCS.parent)}:{lineno} -> {target}")
    assert not problems, "broken relative links in docs:\n  " + "\n  ".join(problems)


def test_files_referenced_by_docs_exist() -> None:
    """Docs point at source and test files by path; those must exist.

    A renamed module or test file leaves the reader following a dead pointer, and
    nothing else notices — the doc still renders.

    ``docs/audits/`` is excluded: those are dated snapshots of what a review
    recommended at the time, including files that were proposed and never
    created. Holding them to today's tree would only produce noise, exactly like
    ``docs/superpowers/``.
    """
    repo_root = _DOCS.parent
    problems: list[str] = []
    checked = 0
    for path in _all_docs():
        if "audits" in path.parts:
            continue
        for lineno, line in enumerate(path.read_text(errors="ignore").split("\n"), 1):
            for match in re.finditer(r"\b((?:tests|src|configs)/[\w/]+\.(?:py|toml|yaml))", line):
                referenced = match.group(1)
                # Placeholder names the reader is told to create themselves
                # ("copy this config to configs/your_distill.yaml").
                if "your_" in referenced:
                    continue
                checked += 1
                if not (repo_root / referenced).exists():
                    problems.append(f"{path.relative_to(repo_root)}:{lineno} -> {referenced}")

    assert checked > 20, f"only {checked} file references found — the scan probably broke"
    assert not problems, "docs reference files that do not exist:\n  " + "\n  ".join(problems)


class TestDocumentedImportsResolve:
    """A `from llm.x import y` in the docs must name a REAL module.

    Docs are instructions. `test_documented_cli_flags_exist` already checks the CLI
    surface; the Python-import surface had no equivalent, so a renamed or removed
    module in a guide would send readers to an ImportError.

    Teaching recipes are exempt by construction: they write
    ``# in tasks/classification_task.py`` and import the module the READER is about
    to create, so those names are deliberately absent.
    """

    #: Modules the docs tell the reader to CREATE. ``guide-extending.md`` presents a
    #: recipe whose code block is headed ``# in tasks/classification_task.py`` and
    #: imports the task the reader is about to write, so the name is absent ON
    #: PURPOSE. Listing it explicitly keeps the scan exhaustive everywhere else.
    _CREATE_IT_YOURSELF = frozenset({"llm.training.tasks.classification_task"})

    @classmethod
    def _documented_module_imports(cls):
        import re

        for path in _all_docs():
            text = path.read_text(encoding="utf-8", errors="ignore")
            for match in re.finditer(r"\bfrom\s+(llm[\w.]*)\s+import\b", text):
                if match.group(1) in cls._CREATE_IT_YOURSELF:
                    continue
                yield path, match.group(1)

    def test_documented_llm_modules_import(self):
        import importlib

        problems: list[str] = []
        checked = 0
        for path, module_name in self._documented_module_imports():
            # `llm.X` is the shape used in placeholder examples.
            if module_name == "llm.X":
                continue
            checked += 1
            try:
                importlib.import_module(module_name)
            except ModuleNotFoundError:
                # Recipes that teach you to CREATE a module reference it before it
                # exists. Only flag the ones sitting under an existing package that
                # the repo actually ships.
                problems.append(f"{path.relative_to(_DOCS.parent)}: {module_name}")

        assert checked > 0, "no `from llm... import` statements found in the docs — the scan is broken"
        # Nothing may fail that is not a documented "you create this" recipe.
        assert not problems, "docs import modules that do not exist:\n  " + "\n  ".join(sorted(set(problems)))


# --------------------------------------------------------------------------- #
# Documented YAML must survive validation, not just key-name lookup
# --------------------------------------------------------------------------- #


class TestDocumentedYamlBlocksValidate:
    """A documented config snippet must pass the REAL validators.

    ``test_no_doc_yaml_block_uses_an_unknown_training_key`` above checks only
    that every key exists in the schema. That cannot see a block whose keys are
    all real but whose VALUES contradict each other — and such a block is worse
    than a typo: it reads as copy-pasteable while being rejected on load.

    Measured on the guide set: ``reference/architecture.md`` showed

        model:
          attn_impl: "mha"
          mlp_impl: "moe"
          norm_impl: "rms_norm"

    Every key is valid, yet ``ModelConfig`` rejects it because ``num_experts``
    defaults to 0 and ``mlp_impl='moe'`` requires it positive. A reader copying
    the snippet got a validation error from a page that looked authoritative.

    A block is only validated once its parsed form carries a recognised
    top-level section (``model`` / ``training`` / ``data`` / ...), so unrelated
    YAML (Kubernetes, Prometheus, illustrative pseudo-config) is skipped rather
    than force-fitted to :class:`Config`.
    """

    #: Sections that make a block a training config rather than incidental YAML.
    _SECTIONS: ClassVar[set[str]] = {
        "model",
        "training",
        "data",
        "optimization",
        "distributed",
        "evaluation",
        "logging",
    }

    @staticmethod
    def _blocks() -> list[tuple[pathlib.Path, dict]]:
        import yaml

        found = []
        for path in _all_docs():
            text = path.read_text(encoding="utf-8", errors="ignore")
            # A page that registers its OWN source (``SOURCE_REGISTRY.register(
            # "s3", ...)``) legitimately then documents ``data_source: s3``; the
            # registry name does not exist until the reader runs the snippet
            # above it. Collect the NAMES that page registers — exempting the
            # whole page would also bless an unregistered typo like ``gcs``
            # (verified: it did, before this was narrowed).
            page_sources = set(re.findall(r'SOURCE_REGISTRY\.register\(\s*"([^"]+)"', text))
            for block in _yaml_blocks(text):
                try:
                    parsed = yaml.safe_load(block)
                except yaml.YAMLError:
                    continue  # illustrative fragments
                if not isinstance(parsed, dict):
                    continue
                if not (set(parsed) & TestDocumentedYamlBlocksValidate._SECTIONS):
                    continue
                source = (parsed.get("data") or {}).get("data_source")
                if source is not None and str(source) in page_sources:
                    # The page registers exactly this name above the YAML, and
                    # the block was verified to validate once it is registered.
                    continue
                found.append((path, parsed))
        # Guard against a vacuous pass (the failure mode this file already hit
        # once with a wrong path): a scan that finds nothing proves nothing.
        assert len(found) >= 20, f"expected the guides to carry many config blocks, found {len(found)}"
        return found

    def test_every_documented_config_block_validates(self) -> None:
        from pydantic import ValidationError

        from tests.support.devices import cuda_device_count

        available_gpus = cuda_device_count()
        problems: list[str] = []
        skipped_for_gpus = 0
        for path, parsed in self._blocks():
            # A block that pins a GPU count above this host's is NOT invalid: it
            # documents a multi-GPU run, and ``DistributedConfig`` correctly
            # refuses it at load time on a smaller machine. Validating it here
            # turned "this box has 0 GPUs" into a docs defect, so the block is
            # exercised only where the hardware it describes exists.
            requested_gpus = (parsed.get("distributed") or {}).get("gpus_per_node")
            if isinstance(requested_gpus, int) and requested_gpus > available_gpus:
                skipped_for_gpus += 1
                continue
            try:
                Config(**parsed)
            except ValidationError as exc:
                first = str(exc).splitlines()[0].strip()
                detail = next(
                    (line.strip() for line in str(exc).splitlines() if "Value error" in line or "error," in line),
                    first,
                )
                problems.append(f"{path.relative_to(_DOCS.parent)}: {detail}")
        assert not problems, (
            "documented YAML blocks that do not pass the real validators "
            "(a copy-paste reader would hit this):\n  " + "\n  ".join(problems)
        )
        # Guard against the GPU filter above silently excusing every block: at
        # least some documented configs must have been validated for real.
        assert len(problems) + skipped_for_gpus < len(self._blocks()), (
            "no documented config block was actually validated — the GPU filter excused all of them"
        )
