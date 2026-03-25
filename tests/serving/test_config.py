from llm.serving.config import ServingConfig


def test_prefix_cache_config_defaults():
    """Test default prefix cache config values."""
    config = ServingConfig()
    assert config.enable_prefix_cache is False
    assert config.max_prefixes == 10


def test_prefix_cache_config_override():
    """Test overriding prefix cache config."""
    config = ServingConfig(
        enable_prefix_cache=True,
        max_prefixes=5,
    )
    assert config.enable_prefix_cache is True
    assert config.max_prefixes == 5
