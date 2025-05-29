import pytest
import torch

from llm.core.attn import MultiHeadAttention

# Default hidden_size for most tests
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_HEADS = 8


@pytest.fixture
def mha_props(request):
    """Fixture to allow parameterizing MHA properties for tests."""
    props = {
        "hidden_size": DEFAULT_HIDDEN_SIZE,
        "num_heads": DEFAULT_NUM_HEADS,
        "p": 0.1,
        "bias": False,
        "eps": 1e-5,
        "norm_first": True,
        "is_causal": False,
        "include_norm_residual": True,  # Default to new True behavior
    }
    # Update with any markers passed via request
    if hasattr(request, "param"):
        props.update(request.param)
    return props


@pytest.fixture
def mha(mha_props):
    """Creates an MHA instance based on mha_props."""
    return MultiHeadAttention(**mha_props)


@pytest.fixture
def input_tensor(mha_props):  # Depend on mha_props to get hidden_size
    """Creates a default input tensor."""
    batch_size = 2
    seq_len = 10
    hidden_size = mha_props.get("hidden_size", DEFAULT_HIDDEN_SIZE)
    return torch.randn(batch_size, seq_len, hidden_size)


# Helper to create MHA with specific config for testing
def create_mha_for_test(**kwargs):
    base_params = {
        "hidden_size": 64,
        "num_heads": 8,
        "p": 0.0,  # Default to no dropout for deterministic tests unless specified
        "norm_first": True,
        "include_norm_residual": True,
    }
    base_params.update(kwargs)
    mha_instance = MultiHeadAttention(**base_params)
    mha_instance.eval()  # Default to eval mode

    input_data = torch.randn(2, 10, base_params["hidden_size"])  # B, S, H
    return mha_instance, input_data


def test_mha_initialization(mha, mha_props):
    """Test if MHA module is initialized correctly based on mha_props."""
    assert isinstance(mha, MultiHeadAttention)
    assert mha.hidden_size == mha_props["hidden_size"]
    assert mha.num_heads == mha_props["num_heads"]
    assert mha.head_dim == mha_props["hidden_size"] // mha_props["num_heads"]
    assert mha.p == mha_props["p"]
    assert mha.is_causal == mha_props["is_causal"]
    assert mha.include_norm_residual == mha_props["include_norm_residual"]

    if mha_props["include_norm_residual"]:
        assert mha.norm is not None
        assert mha.norm_first == mha_props["norm_first"]
    else:
        assert mha.norm is None

    if mha_props["bias"]:
        assert mha.qkv_proj.bias is not None
        assert mha.out_proj.bias is not None
    else:
        assert mha.qkv_proj.bias is None
        assert mha.out_proj.bias is None


@pytest.mark.parametrize(
    "mha_props",
    [
        {"include_norm_residual": True, "bias": True},
        {"include_norm_residual": True, "bias": False},
        {"include_norm_residual": False, "bias": True},
        {"include_norm_residual": False, "bias": False},
    ],
    indirect=True,
)
def test_mha_forward_shape(mha, input_tensor):
    """Test if forward pass maintains correct shape for norm/residual and bias modes."""
    output = mha(input_tensor)
    assert output.shape == input_tensor.shape


def test_mha_with_mask(mha, input_tensor):  # Uses default mha_props (norm_res=True)
    """Test MHA with attention mask."""
    # Create a simple attention mask (True means masked for scaled_dot_product_attention)
    # Masking out the last 5 tokens for all heads, all sequences in batch
    attn_mask = torch.zeros(input_tensor.size(0), input_tensor.size(1), input_tensor.size(1), dtype=torch.bool)
    attn_mask[:, :, -5:] = True  # Mask last 5 tokens
    # Reshape for MHA: [B, N, S, S] or broadcastable like [B, 1, S, S]
    # The scaled_dot_product_attention takes [B, N, S, S] or certain broadcastable forms.
    # A common padding mask is [B, 1, 1, S_key]
    # For this test, a full [B, N, S, S] mask:
    mask_for_mha = attn_mask.unsqueeze(1).repeat(1, mha.num_heads, 1, 1)

    output = mha(input_tensor, attn_mask=mask_for_mha)
    assert output.shape == input_tensor.shape

    # More detailed check: output for masked positions might differ
    # This is complex to verify without knowing exact weights. Shape check is primary here.


@pytest.mark.parametrize(
    "mha_props", [{"include_norm_residual": True}, {"include_norm_residual": False}], indirect=True
)
def test_mha_gradients(mha, input_tensor):
    """Test if gradients are computed correctly for both norm/residual modes."""
    input_tensor.requires_grad_(True)
    output = mha(input_tensor)
    loss = output.sum()
    loss.backward()
    assert input_tensor.grad is not None
    assert not torch.isnan(input_tensor.grad).any()
    # Check gradients for bias terms if they exist
    if mha.qkv_proj.bias is not None:
        assert mha.qkv_proj.bias.grad is not None
        assert not torch.isnan(mha.qkv_proj.bias.grad).any()
    if mha.out_proj.bias is not None:
        assert mha.out_proj.bias.grad is not None
        assert not torch.isnan(mha.out_proj.bias.grad).any()


@pytest.mark.parametrize(
    "mha_props",
    [{"is_causal": True, "include_norm_residual": True}, {"is_causal": True, "include_norm_residual": False}],
    indirect=True,
)
def test_mha_causal(mha, input_tensor):
    """Test causal MHA for both norm/residual modes."""
    assert mha.is_causal is True
    output = mha(input_tensor)
    assert output.shape == input_tensor.shape
    # Further checks could involve comparing outputs with a manually created causal mask.


def test_mha_different_num_heads(mha_props):
    """Test MHA with different number of heads for both norm/residual modes."""
    num_heads_list = [2, 4]  # Assuming hidden_size=64 from default mha_props
    for num_heads in num_heads_list:
        props = mha_props.copy()
        props["num_heads"] = num_heads
        current_mha = MultiHeadAttention(**props)
        current_input_tensor = torch.randn(2, 10, props["hidden_size"])
        output = current_mha(current_input_tensor)
        assert output.shape == current_input_tensor.shape


@pytest.mark.parametrize(
    "mha_props", [{"include_norm_residual": True}, {"include_norm_residual": False}], indirect=True
)
@pytest.mark.parametrize("hidden_size_test", [32, 128])  # Test different hidden sizes
def test_mha_different_hidden_sizes(mha_props, hidden_size_test):
    """Test MHA with different hidden sizes for both norm/residual modes."""
    props = mha_props.copy()
    # Ensure num_heads is compatible with new hidden_size, e.g., hidden_size must be div by num_heads
    # For simplicity, let's use a fixed num_heads that works for common hidden_sizes or adjust it.
    num_heads_test = 4
    if hidden_size_test % num_heads_test != 0:
        pytest.skip(f"Hidden size {hidden_size_test} not compatible with {num_heads_test} heads.")

    props["hidden_size"] = hidden_size_test
    props["num_heads"] = num_heads_test  # Adjust num_heads if necessary

    current_mha = MultiHeadAttention(**props)
    current_input_tensor = torch.randn(2, 10, props["hidden_size"])
    output = current_mha(current_input_tensor)
    assert output.shape == current_input_tensor.shape


@pytest.mark.parametrize(
    "mha_props", [{"include_norm_residual": True}, {"include_norm_residual": False}], indirect=True
)
def test_mha_different_batch_sizes(mha_props, mha):  # mha here uses the parametrized mha_props
    """Test MHA with different batch sizes for both norm/residual modes."""
    batch_sizes = [1, 2, 4]
    for batch_size in batch_sizes:
        input_tensor_varied_batch = torch.randn(batch_size, 10, mha_props["hidden_size"])
        output = mha(input_tensor_varied_batch)
        assert output.shape == input_tensor_varied_batch.shape


@pytest.mark.parametrize(
    "mha_props", [{"include_norm_residual": True}, {"include_norm_residual": False}], indirect=True
)
def test_mha_different_sequence_lengths(mha_props, mha):  # mha here uses the parametrized mha_props
    """Test MHA with different sequence lengths for both norm/residual modes."""
    seq_lengths = [5, 10, 20]
    for seq_len in seq_lengths:
        input_tensor_varied_seq = torch.randn(2, seq_len, mha_props["hidden_size"])
        output = mha(input_tensor_varied_seq)
        assert output.shape == input_tensor_varied_seq.shape


@pytest.mark.parametrize("norm_first_val", [True, False])
def test_mha_internal_norm_first_when_norm_residual_active(norm_first_val):
    """Test MHA's internal norm_first behavior when include_norm_residual is True."""
    mha_instance, input_data = create_mha_for_test(
        norm_first=norm_first_val,
        include_norm_residual=True,
        p=0.0,  # Disable dropout for comparability
    )
    assert mha_instance.norm is not None
    assert mha_instance.norm_first == norm_first_val

    output = mha_instance(input_data)
    assert output.shape == input_data.shape

    # Compare outputs of Pre-LN and Post-LN MHA (when norm/residual is active)
    # They should differ if hidden_size > 1
    mha_opposite_norm, _ = create_mha_for_test(norm_first=not norm_first_val, include_norm_residual=True, p=0.0)
    # Copy weights for fair comparison
    mha_opposite_norm.load_state_dict(mha_instance.state_dict())

    output_opposite = mha_opposite_norm(input_data)

    if input_data.shape[-1] > 1:  # LayerNorm makes a difference if hidden_size > 1
        assert not torch.allclose(output, output_opposite, atol=1e-5), (
            f"Outputs of Pre-LN and Post-LN MHA (norm_res=True) should differ for norm_first={norm_first_val}"
        )


def test_mha_no_norm_residual_output():
    """Test MHA behavior and output when include_norm_residual is False."""
    torch.manual_seed(0)
    mha_instance, input_data = create_mha_for_test(
        include_norm_residual=False,
        p=0.0,  # No dropout for exact check
    )
    mha_instance.eval()

    assert mha_instance.norm is None
    assert mha_instance.include_norm_residual is False

    # Manually compute expected output: QKV -> SDPA -> OutProj
    # 1. QKV projection
    qkv = mha_instance.qkv_proj(input_data)
    batch_size, seq_len, _ = input_data.size()
    qkv = qkv.reshape(batch_size, seq_len, 3, mha_instance.num_heads, mha_instance.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    # 2. Scaled Dot-Product Attention (assuming no mask, not causal, p=0 for SDPA)
    # The internal SDPA call in MHA uses self.p if training, 0 if eval. MHA is in eval.
    attn_output_sdpa = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=mha_instance.is_causal
    )
    attn_output_reshaped = attn_output_sdpa.transpose(1, 2).reshape(batch_size, seq_len, mha_instance.hidden_size)

    # 3. Output projection (dropout is 0.0 as p=0.0 for mha_instance)
    expected_output = mha_instance.out_proj(attn_output_reshaped)
    # mha_instance.dropout is nn.Dropout(0.0), so it's an identity op.

    actual_output = mha_instance(input_data)

    assert torch.allclose(actual_output, expected_output, atol=1e-6), (
        "Output with no norm/residual does not match manual computation."
    )

    # Compare with MHA that *does* include norm and residual (default Pre-LN)
    mha_with_norm_res, _ = create_mha_for_test(
        include_norm_residual=True,
        norm_first=True,  # Default Pre-LN
        p=0.0,
    )
    # Copy relevant weights (qkv_proj, out_proj) for fair comparison
    # Norm weights are not copied as mha_instance doesn't have them.
    mha_with_norm_res.qkv_proj.load_state_dict(mha_instance.qkv_proj.state_dict())
    mha_with_norm_res.out_proj.load_state_dict(mha_instance.out_proj.state_dict())

    output_with_norm_res = mha_with_norm_res(input_data)

    if input_data.shape[-1] > 1:  # Norm/residual usually make a difference if H > 1
        assert not torch.allclose(actual_output, output_with_norm_res, atol=1e-5), (
            "Output with no norm/residual should differ from output with norm/residual."
        )


def test_mha_no_norm_residual_dropout_active():
    """Test that output dropout is still active in MHA when include_norm_residual is False."""
    dropout_p_test = 0.5
    mha_instance, input_data = create_mha_for_test(
        include_norm_residual=False,
        p=dropout_p_test,  # Set dropout for MHA's output projection
    )

    # Eval mode (dropout disabled)
    mha_instance.eval()
    output_eval_1 = mha_instance(input_data)
    output_eval_2 = mha_instance(input_data)
    assert torch.allclose(output_eval_1, output_eval_2, atol=1e-7), (
        "Outputs in eval mode should be identical (no norm/res)."
    )

    # Train mode (dropout active)
    mha_instance.train()
    torch.manual_seed(10)
    output_train_1 = mha_instance(input_data)
    torch.manual_seed(11)  # Ensure different dropout mask potentially
    output_train_2 = mha_instance(input_data)

    if dropout_p_test > 0:
        # Note: scaled_dot_product_attention also has dropout_p. If p > 0, it's also active in train.
        # The MHA module passes self.p to scaled_dot_product_attention IF self.training.
        # So, both attention dropout and output_proj dropout are active.
        assert not torch.allclose(output_train_1, output_train_2, atol=1e-6), (
            "Outputs in train mode should differ due to dropout (no norm/res)."
        )


@pytest.mark.parametrize(
    "mha_props", [{"p": 0.5, "include_norm_residual": True}, {"p": 0.5, "include_norm_residual": False}], indirect=True
)
def test_mha_dropout_train_eval_modes(mha, input_tensor, mha_props):
    """
    Tests MHA dropout behavior in train vs eval modes, parameterized for include_norm_residual.
    Dropout (p > 0) is applied in two places: within scaled_dot_product_attention and in the output projection.
    """
    dropout_p_test = mha_props["p"]
    assert dropout_p_test > 0, "This test requires dropout_p > 0"

    # Eval mode: Dropout should be disabled, outputs should be identical.
    mha.eval()
    with torch.no_grad():
        output_eval_1 = mha(input_tensor)
        output_eval_2 = mha(input_tensor)
    assert torch.allclose(output_eval_1, output_eval_2, atol=1e-7), (
        f"Outputs in eval mode should be identical (include_norm_residual={mha_props['include_norm_residual']})"
    )

    # Train mode: Dropout should be active, outputs should differ.
    mha.train()
    # As MHA applies dropout in two places (SDPA and output projection),
    # subsequent calls in train mode should produce different results.
    with torch.no_grad():
        output_train_1 = mha(input_tensor)
        output_train_2 = mha(input_tensor)

    assert not torch.allclose(output_train_1, output_train_2, atol=1e-6), (
        f"Outputs in train mode should differ due to dropout (include_norm_residual={mha_props['include_norm_residual']})"
    )


def test_mha_initialization_invalid_hidden_size_num_heads():
    """Test MHA initialization with hidden_size not divisible by num_heads."""
    with pytest.raises(ValueError, match="hidden_size .* must be divisible by num_heads .*"):
        MultiHeadAttention(hidden_size=60, num_heads=8)  # 60 is not divisible by 8

    with pytest.raises(ValueError, match="hidden_size .* must be divisible by num_heads .*"):
        MultiHeadAttention(hidden_size=32, num_heads=3)  # 32 is not divisible by 3
