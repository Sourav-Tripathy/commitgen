from abc import ABC, abstractmethod
from typing import List, Optional

class BaseProvider(ABC):
    """Base class for AI providers"""
    
    def __init__(self, config: dict):
        self.config = config
    
    @abstractmethod
    def generate_commit_message(
        self, 
        diff: str, 
        file_summary: str,
        context: Optional[str] = None
    ) -> str:
        """
        Generate commit message from diff
        
        Args:
            diff: Git diff content
            file_summary: High-level summary of changed files
            context: Optional additional context
            
        Returns:
            Generated commit message
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and configured"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model being used"""
        pass
