"""
CommitGen - AI-powered git commit message generator
"""

__version__ = "0.1.0"
__author__ = "Sourav Tripathy"
__license__ = "MIT"

# Use lazy loading or ensure dependencies are there
from commitgen.core.message_generator import generate_commit_message
from commitgen.core.config_manager import ConfigManager

__all__ = ["generate_commit_message", "ConfigManager"]
