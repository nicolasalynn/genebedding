# Testing Genebeddings Wrappers

This document describes how to test the standardized wrapper APIs.

## Quick Smoke Test

The fastest way to verify the API structure is correct:

```bash
python quick_test.py
```

This tests:
- ✓ All wrappers can be imported
- ✓ All wrappers inherit from `BaseWrapper`
- ✓ Method signatures are correct
- ✓ Capability discovery works

**This does NOT load actual models** - it just verifies the code structure.

## Comprehensive Model Testing

To test actual model functionality with real inference:

```bash
python test_wrappers.py
```

### Interactive Mode (Default)

When run without arguments, the script presents an interactive menu:

```
Available Models
1. Nucleotide Transformer (nt)
2. Caduceus (caduceus)
3. SpeciesLM (specieslm)
4. RiNALMo (rinalmo)
5. Borzoi (borzoi)
6. ConvNova (convnova) (requires manual setup)
7. Test all models
0. Exit

Select models to test (comma-separated numbers or 'all'):
```

Just enter the numbers of models you want to test (e.g., `1,2,4` or `all`).

### Command-Line Options

Test all models automatically:
```bash
python test_wrappers.py --all
```

Test specific models:
```bash
python test_wrappers.py --models nt caduceus
```

Quick test with shorter sequences (faster):
```bash
python test_wrappers.py --models nt --quick
```

### What Gets Tested

For each model, the script tests:

1. **Capability Discovery**
   - `supports_capability(name)` returns correct values
   - `get_capabilities()` lists all capabilities

2. **Embedding (`embed`)**
   - Mean pooling: returns (hidden_dim,) array
   - CLS pooling: returns (hidden_dim,) array
   - Token pooling: returns (num_tokens, hidden_dim) array
   - `return_numpy=False` returns torch.Tensor

3. **Nucleotide Prediction (`predict_nucleotides`)** *(if supported)*
   - Dict output: list of `{'A': p_A, 'C': p_C, 'G': p_G, 'T': p_T}`
   - Array output: (num_positions, 4) numpy array
   - Probabilities sum to 1.0

4. **Track Prediction (`predict_tracks`)** *(if supported)*
   - Returns (num_tracks, num_positions) numpy array
   - Values in reasonable range

## Model Requirements

Different models have different dependencies:

| Model | Dependencies | Notes |
|-------|-------------|-------|
| NT | `transformers` | ~2GB model |
| Caduceus | `transformers` | Various sizes |
| SpeciesLM | `transformers` | ~1GB model |
| RiNALMo | `rinalmo` | RNA/DNA model |
| Borzoi | `borzoi_pytorch` | Large model, needs 524k bp sequences |
| ConvNova | Custom | Requires repo path + config files |

Install missing dependencies as needed:
```bash
pip install transformers
pip install rinalmo
pip install borzoi-pytorch
```

## Expected Output

### Successful Test

```
================================================================================
                        Testing Nucleotide Transformer
================================================================================

ℹ Initializing model...
✓ Model initialized: NTWrapper(capabilities=[embed, predict_nucleotides])
ℹ Generated test sequence of length 1000
ℹ Testing capability discovery for Nucleotide Transformer...
✓ Capabilities: ['embed', 'predict_nucleotides']
✓ Supports 'embed': True
✓ Supports 'predict_nucleotides': True
✓ Does not support 'predict_tracks': False
ℹ Testing embed() for Nucleotide Transformer...
✓ Mean pooling: shape (512,)
✓ CLS pooling: shape (512,)
✓ Token embeddings: shape (167, 512)
✓ Torch tensor output: shape torch.Size([512])
ℹ Testing predict_nucleotides() for Nucleotide Transformer...
✓ Dict output: 3 positions predicted
ℹ   Example at pos 10: A=0.234, C=0.187, G=0.301, T=0.278
✓ Array output: shape (3, 4)

✓ All 3 tests passed for Nucleotide Transformer!
```

### Skipped Model

```
================================================================================
                               Testing ConvNova
================================================================================

⚠ ConvNova requires manual setup (config files, checkpoints, etc.)
⚠ Skipping this model. Please configure it manually to test.
```

## Manual Testing

You can also test wrappers manually in Python:

```python
from genebeddings.wrappers import NTWrapper
import numpy as np

# Initialize
model = NTWrapper()

# Check capabilities
print(model.get_capabilities())  # ['embed', 'predict_nucleotides']
print(model.supports_capability('predict_tracks'))  # False

# Test embedding
seq = "ACGTACGT" * 100
emb = model.embed(seq, pool='mean')
print(emb.shape)  # (512,)

# Test nucleotide prediction
probs = model.predict_nucleotides(seq, positions=[10, 20, 30])
print(probs[0])  # {'A': 0.23, 'C': 0.19, 'G': 0.31, 'T': 0.27}
```

## Troubleshooting

### Import Errors

```
ImportError: cannot import name 'NTWrapper'
```

**Solution**: Make sure you're running from the genebeddings directory or have it installed:
```bash
pip install -e .
```

### Missing Dependencies

```
⚠ Missing dependencies: transformers
⚠ Install with: pip install transformers
```

**Solution**: Install the required package:
```bash
pip install transformers
```

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**:
- Use `--quick` flag for shorter sequences
- Set smaller batch sizes
- Run on CPU by setting device manually:
```python
model = NTWrapper(device='cpu')
```

### Model Download Issues

If models fail to download:
- Check internet connection
- Try setting HuggingFace cache directory: `export HF_HOME=/path/to/cache`
- Some models require authentication: `huggingface-cli login`

## Adding New Tests

To add tests for new functionality:

1. Add test function in `test_wrappers.py`:
```python
def test_my_feature(model: Any, seq: str, model_name: str) -> bool:
    """Test my new feature."""
    print_info(f"Testing my_feature() for {model_name}...")

    try:
        result = model.my_feature(seq)
        assert result is not None
        print_success("my_feature() works!")
        return True
    except Exception as e:
        print_error(f"my_feature() failed: {e}")
        return False
```

2. Add to model test sequence:
```python
# In test_model() function
if 'my_feature' in config['capabilities']:
    results['my_feature'] = test_my_feature(model, seq, config['name'])
```

3. Update model config:
```python
MODEL_CONFIGS['mymodel']['capabilities'].append('my_feature')
```
