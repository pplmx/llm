"""Enhanced logging manager with distributed-training support."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from llm.training.core.config import LoggingConfig


class Logger:
    """Enhanced logging manager with distributed training support."""

    def __init__(self, rank: int, config: LoggingConfig):
        self.rank = rank
        self.config = config
        self.logger = logging.getLogger(f"rank_{rank}")
        # Only setup handlers if they haven't been added for this logger name yet
        if not self.logger.hasHandlers():
            self._setup_logging()
        else:
            # Ensure level is set even if handlers are already there
            # This could happen if a default logger existed before our custom setup
            self.logger.setLevel(getattr(logging, self.config.log_level.upper()))

    def _setup_logging(self):
        # This method is now only called if self.logger.hasHandlers() was false
        self.logger.setLevel(getattr(logging, self.config.log_level.upper()))
        formatter = logging.Formatter(
            f"[%(asctime)s] [%(levelname)s] [Rank {self.rank}] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        if self.rank == 0:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            if self.config.save_logs:
                Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
                # Use human-friendly timestamp format
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                log_path = Path(self.config.log_dir) / f"training_{timestamp}.log"
                file_handler = logging.FileHandler(log_path)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
        else:
            self.logger.addHandler(logging.NullHandler())

    def __getattr__(self, name):
        # Proxy all logger methods (info, warning, error, etc.)
        return getattr(self.logger, name)
