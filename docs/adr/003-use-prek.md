# 003. Use prek for Git Hooks

Date: 2025-12-21

## Status

Accepted

## Context

Git hooks are essential for maintaining code quality by running checks before commits and pushes. The traditional solution is `pre-commit`, but newer alternatives like `prek` offer better performance and integration with modern Python tooling.

Key considerations:

- **Performance**: Hook execution speed affects developer productivity
- **Tool integration**: Need seamless integration with uv, ruff, ty
- **Developer experience**: Simple configuration and clear error messages
- **Ecosystem alignment**: Consistency with project's modern toolchain

## Decision

We adopt **prek** as our Git hook manager, replacing the traditional pre-commit framework.

**Rationale**:

- **Performance**: Written in Rust, significantly faster than Python-based pre-commit
- **Native integration**: Built-in support for uv, ruff, ty without wrapper scripts
- **Simpler configuration**: More intuitive and less verbose than pre-commit YAML
- **Modern tooling**: Aligns with our choice of Astral tools (uv, ruff, ty)
- **Active development**: Well-maintained and rapidly evolving

**Configuration approach**:

- Use prek's native hook system
- Integrate with Makefile for consistency
- Run ruff, ty, and tests before commits

## Consequences

### Positive

- **Faster hooks**: Hooks run 2-5x faster than pre-commit
- **Better DX**: Simpler configuration, clearer output
- **Tool consistency**: All tools from the same ecosystem (Astral)
- **Less complexity**: No need for hook wrappers or virtualenvs
- **Future-proof**: Aligned with modern Python tooling direction

### Negative

- **Less mature**: prek has smaller community than pre-commit
- **Fewer plugins**: Not as many pre-built hooks available
- **Learning curve**: Team needs to learn new tool (though it's simpler)
- **Migration cost**: Need to convert existing pre-commit config

### Neutral

- **Different paradigm**: Hooks configured differently than pre-commit
- **Documentation**: Need to document prek usage for contributors

## References

- [prek GitHub Repository](https://github.com/astral-sh/prek)
- [Why we're migrating from pre-commit](https://astral.sh/blog/prek)
- Alternative considered: pre-commit (<https://pre-commit.com/>)
