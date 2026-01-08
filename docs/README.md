# LLM Project Documentation

Welcome to the LLM project documentation. This index provides quick navigation to all available guides and references.

## Quick Links

| Document | Description |
| -------- | ----------- |
| [Usage Guide](usage.md) | Training, inference, and serving instructions |
| [Architecture](architecture.md) | System design and component overview |
| [Tutorial](tutorial-cpu-llm.md) | Step-by-step CPU LLM training tutorial |
| [Development](development.md) | Setting up the development environment |
| [FAQ](faq.md) | Frequently asked questions |
| [Troubleshooting](troubleshooting.md) | Common issues and solutions |

## Training Framework

Detailed documentation for the training system:

| Document | Description |
| -------- | ----------- |
| [Overview](training/README.md) | Training framework introduction |
| [Components](training/components.md) | Core training components |
| [Training Flow](training/training-flow.md) | End-to-end training process |
| [Callbacks](training/flow-callbacks.md) | Callback system for extensibility |
| [Configuration](training/guide-configuration.md) | Configuration guide |
| [Extending](training/guide-extending.md) | How to extend the framework |
| [DDP Deep Dive](training/deep-dive-ddp.md) | Distributed training details |
| [Troubleshooting](training/troubleshooting.md) | Training-specific issues |

## Architecture Decision Records (ADR)

Design decisions and their rationale:

| ADR | Topic |
| --- | ----- |
| [001](adr/001-use-gqa.md) | Grouped Query Attention (GQA) |
| [002](adr/002-use-swiglu.md) | SwiGLU Activation |
| [003](adr/003-use-prek.md) | Using prek for Git hooks |
| [004](adr/004-use-ty.md) | Using ty for type checking |

## Deep Dives

In-depth technical explorations:

- [Attention Mechanisms Part 1](../_learning/01_concepts/attn-01.md)
- [Attention Mechanisms Part 2](../_learning/01_concepts/attn-02.md)

---

> **Contributing**: See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.
>
> **Issues**: Report bugs at [GitHub Issues](https://github.com/pplmx/llm/issues).
