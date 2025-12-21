# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the LLM project.

## What is an ADR?

An Architecture Decision Record (ADR) is a document that captures an important architectural decision made along with its context and consequences.

## Format

Each ADR follows this structure:

```markdown
# [Number]. [Title]

Date: YYYY-MM-DD

## Status

[Proposed | Accepted | Deprecated | Superseded]

## Context

What is the issue that we're seeing that is motivating this decision or change?

## Decision

What is the change that we're proposing and/or doing?

## Consequences

What becomes easier or more difficult to do because of this change?
```

## Index

- [ADR-001: Use Grouped Query Attention (GQA)](001-use-gqa.md)
- [ADR-002: Use SwiGLU Activation Function](002-use-swiglu.md)
- [ADR-003: Use prek for Git Hooks](003-use-prek.md)
- [ADR-004: Use ty for Type Checking](004-use-ty.md)

## Creating a New ADR

1. Copy the template: `cp template.md 00X-title.md`
2. Fill in the sections
3. Update this index
4. Submit a PR
