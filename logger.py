### FILE: logger.py
import logging
import sys
from pathlib import Path
from typing import cast
import colorlog
from dotenv import load_dotenv
import os

# Mapping des niveaux de log possibles
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class Logger:
    _instance = None
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._logger is None:
            self._setup_logger()

    def _get_log_level(self) -> int:
        """
        Récupère le niveau de log depuis les variables d'environnement
        avec INFO comme niveau par défaut
        """
        load_dotenv()
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        if log_level not in LOG_LEVELS:
            print(f"Warning: Invalid LOG_LEVEL '{log_level}'. Using INFO as default.")
            return logging.INFO
        return LOG_LEVELS[log_level]

    def _setup_logger(self):
        """Configure the logger with both file and console handlers"""
        self._logger = logging.getLogger("o1_engineer")

        # Définir le niveau de log depuis l'environnement
        log_level = self._get_log_level()
        self._logger.setLevel(log_level)

        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # File handler - pas de couleurs dans le fichier
        file_handler = logging.FileHandler("logs/o1_engineer.log")
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        # Console handler avec couleurs
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            reset=True,
        )
        console_handler.setFormatter(console_formatter)

        # Add handlers if they don't exist
        if not self._logger.handlers:
            self._logger.addHandler(file_handler)
            self._logger.addHandler(console_handler)

            # Log le niveau de log configuré
            self._logger.info(
                f"Logger initialized with level: {logging.getLevelName(log_level)}"
            )

    @classmethod
    def get_logger(cls):
        """Get the singleton logger instance"""
        if cls._instance is None:
            cls()
        return cast(logging.Logger, cls._instance._logger)  # type: ignore


# Initialize the singleton on module import
logger = Logger.get_logger()
