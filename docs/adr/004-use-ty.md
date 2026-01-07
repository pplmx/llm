# 004. Use ty for Type Checking

Date: 2025-12-21

## Status

Accepted

## Context

Static type checking is crucial for maintaining code quality in Python projects. While `mypy` has been the industry standard, newer alternatives like `ty` (from Astral) offer significant performance improvements and better developer experience.

Key considerations:

- **Type checking speed**: Affects development workflow and CI time
- **Error messages**: Quality of type error reporting
- **Configuration complexity**: Ease of setup and maintenance
- **Ecosystem fit**: Consistency with project toolchain (uv, ruff)
- **Feature parity**: Must support necessary type checking features

## Decision

We adopt **ty** as our type checker, replacing mypy.

**Rationale**:

- **Performance**: 5-10x faster than mypy, written in Rust
- **Better errors**: More readable and actionable error messages
- **Zero config**: Works out of the box with sensible defaults
- **Ecosystem alignment**: Part of Astral's toolchain alongside uv and ruff
- **Modern features**: Supports latest Python type hints including PEP 695
- **Active development**: Rapidly evolving with strong backing

**Usage**:

```bash
make ty  # Run ty type checking
```

## Consequences

### Positive

- **Faster CI**: Type checking 5-10x faster, reducing CI time
- **Better DX**: Clearer error messages help developers fix issues faster
- **Simpler config**: No complex mypy.ini needed
- **Tool consistency**: All dev tools from same ecosystem
- **Future-proof**: Active development ensures modern Python support
- **Lower barrier**: Easier for new contributors to understand errors

### Negative

- **Newer tool**: Less mature than mypy (fewer years in production)
- **Smaller community**: Fewer StackOverflow answers and blog posts
- **Plugin ecosystem**: Mypy has more third-party plugins
- **Some edge cases**: May have different behavior in corner cases
- **Migration effort**: Need to address any differences from mypy

### Neutral

- **Different strictness**: May flag different issues than mypy
- **Configuration format**: Uses different config format (though simpler)
- **IDE integration**: VS Code support through Pylance (which we already use)

## References

- [ty GitHub Repository](https://github.com/astral-sh/ty)
- [Astral's vision for Python tooling](https://astral.sh/blog)
- Alternative considered: mypy (<https://mypy-lang.org/>)
- Configured in: `pyproject.toml` and `Makefile`
