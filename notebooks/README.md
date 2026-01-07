# Notebooks

Interactive Jupyter notebooks demonstrating the LLM framework capabilities.

## Available Notebooks

| Notebook | Description |
| -------- | ----------- |
| [quick_start.ipynb](quick_start.ipynb) | Complete tutorial: model building, training, inference, advanced features |

## Quick Start

```bash
# Install Jupyter if needed
uv pip install jupyter

# Launch Jupyter
cd notebooks
jupyter notebook
```

## Topics Covered

### quick_start.ipynb

1. **Building a Model** - Create a Decoder-only Transformer
2. **Training** - Train on synthetic data with loss visualization
3. **Inference** - Text generation (streaming and non-streaming)
4. **Advanced Features**:
   - Gradient Checkpointing (memory-efficient training)
   - Grouped Query Attention (GQA)
   - SwiGLU Activation
   - Mixture of Experts (MoE)
5. **E2E Pipeline** - Full train → evaluate → inference workflow
