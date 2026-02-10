from typing import Optional
from commitgen.core.git_handler import GitHandler
from commitgen.core.diff_processor import DiffProcessor
from commitgen.core.config_manager import ConfigManager
from commitgen.providers.ollama_provider import OllamaProvider
from commitgen.providers.claude_provider import ClaudeProvider
from commitgen.providers.openrouter_provider import OpenRouterProvider

class MessageGenerator:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager.load()
        self.git_handler = GitHandler()
        self.diff_processor = DiffProcessor(
            max_tokens_per_chunk=self.config.ollama.max_tokens_per_chunk
        )
        
    def _get_provider(self, provider_override: Optional[str] = None):
        """Get appropriate provider based on config or override"""
        provider_name = provider_override or self.config.general.provider
        
        if provider_name == "ollama":
            return OllamaProvider(self.config.ollama.model_dump())
        elif provider_name == "claude":
            return ClaudeProvider(self.config.claude.model_dump())
        elif provider_name == "openrouter":
            return OpenRouterProvider(self.config.openrouter.model_dump())
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
    
    def generate(
        self, 
        provider_override: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> str:
        """
        Main generation flow:
        1. Get staged diff
        2. Filter and process diff
        3. Generate file summary
        4. Call AI provider
        5. Validate and format result
        """
        # Check for staged changes
        if not self.git_handler.has_staged_changes():
            raise ValueError("No staged changes found. Use 'git add' first.")
        
        # Get diff and files
        raw_diff = self.git_handler.get_staged_diff()
        staged_files = self.git_handler.get_staged_files()
        
        # Filter if enabled
        if self.config.advanced.filter_generated_files:
            diff = self.diff_processor.filter_diff(raw_diff, staged_files)
        else:
            diff = raw_diff
        
        # Generate summary
        file_summary = self.diff_processor.summarize_changes(staged_files)
        
        # Get provider
        provider = self._get_provider(provider_override)
        
        # Override model if specified
        if model_override:
            provider.model = model_override
        
        # Check availability
        if not provider.is_available():
            raise ValueError(
                f"Provider '{provider_override or self.config.general.provider}' "
                f"is not available. Check configuration."
            )
        
        # Generate message
        commit_message = provider.generate_commit_message(diff, file_summary)
        
        # Validate and format
        commit_message = self._validate_and_format(commit_message)
        
        return commit_message
    
    def _validate_and_format(self, message: str) -> str:
        """Ensure message follows conventions and length limits"""
        lines = message.strip().split('\n')
        first_line = lines[0]
        
        # Enforce max length on first line
        if len(first_line) > self.config.general.max_length:
            first_line = first_line[:self.config.general.max_length]
            lines[0] = first_line
        
        # Ensure conventional commits format if enabled
        if self.config.general.conventional_commits:
            if not self._is_conventional_format(first_line):
                # Try to fix by adding "chore: " prefix if completely missing type
                # But be careful not to double prefix if it's just malformed
                first_line = f"chore: {first_line}"
                lines[0] = first_line
        
        return '\n'.join(lines)
    
    def _is_conventional_format(self, line: str) -> bool:
        """Check if line follows conventional commits format"""
        import re
        pattern = r'^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)(\(.+\))?: .+'
        return bool(re.match(pattern, line))


def generate_commit_message(
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> str:
    """
    Convenience function for generating commit messages
    """
    config_manager = ConfigManager()
    generator = MessageGenerator(config_manager)
    return generator.generate(provider_override=provider, model_override=model)
