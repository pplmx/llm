"""Tests for :class:`llm.training.core.logger.Logger`."""

from __future__ import annotations

import logging

from llm.training.core.config import LoggingConfig
from llm.training.core.logger import Logger


def _clear_logger(rank: int):
    log = logging.getLogger(f"rank_{rank}")
    for handler in log.handlers[:]:
        handler.close()
        log.removeHandler(handler)
    log.propagate = True


class TestLogger:
    def setup_method(self):
        for r in range(20):
            _clear_logger(r)

    def teardown_method(self):
        for r in range(20):
            _clear_logger(r)

    def test_init_rank_zero_creates_stream_handler(self, tmp_path):
        config = LoggingConfig(log_interval=10, log_level="INFO", save_logs=False, log_dir=str(tmp_path))
        logger_obj = Logger(rank=0, config=config)
        assert logger_obj.rank == 0
        assert logger_obj.logger.level == logging.INFO
        assert any(isinstance(h, logging.StreamHandler) for h in logger_obj.logger.handlers)

    def test_init_non_zero_rank_no_stream_handler(self, tmp_path):
        config = LoggingConfig(log_interval=10, log_level="INFO", save_logs=False, log_dir=str(tmp_path))
        logger_obj = Logger(rank=1, config=config)
        assert logger_obj.rank == 1
        assert not any(isinstance(h, logging.StreamHandler) for h in logger_obj.logger.handlers)

    def test_save_logs_creates_log_directory_and_file_handler(self, tmp_path):
        log_dir = tmp_path / "logs"
        config = LoggingConfig(log_interval=10, log_level="DEBUG", save_logs=True, log_dir=str(log_dir))
        logger_obj = Logger(rank=0, config=config)
        assert logger_obj.logger.level == logging.DEBUG
        assert log_dir.exists()
        assert any(isinstance(h, logging.FileHandler) for h in logger_obj.logger.handlers)

    def test_proxies_logging_methods(self, tmp_path):
        config = LoggingConfig(log_interval=10, log_level="INFO", save_logs=False, log_dir=str(tmp_path))
        logger_obj = Logger(rank=0, config=config)
        assert callable(logger_obj.info)
        assert callable(logger_obj.warning)
        assert callable(logger_obj.error)
        assert callable(logger_obj.debug)

    def test_rank_nonzero_has_no_file_handler(self, tmp_path):
        log_dir = tmp_path / "logs"
        config = LoggingConfig(log_interval=10, log_level="INFO", save_logs=True, log_dir=str(log_dir))
        logger_obj = Logger(rank=1, config=config)
        assert not any(isinstance(h, logging.FileHandler) for h in logger_obj.logger.handlers)

    def test_respects_log_level_string(self, tmp_path):
        config = LoggingConfig(log_interval=10, log_level="WARNING", save_logs=False, log_dir=str(tmp_path))
        logger_obj = Logger(rank=0, config=config)
        assert logger_obj.logger.level == logging.WARNING

    def test_can_log_messages(self, tmp_path):
        log_dir = tmp_path / "logs"
        config = LoggingConfig(log_interval=10, log_level="DEBUG", save_logs=True, log_dir=str(log_dir))
        logger_obj = Logger(rank=0, config=config)
        logger_obj.info("test message %d", 42)
        log_files = list(log_dir.glob("training_*.log"))
        assert len(log_files) == 1
        content = log_files[0].read_text()
        assert "test message 42" in content
