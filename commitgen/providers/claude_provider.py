from typing import Optional
from anthropic import Anthropic
from commitgen.providers.base import BaseProvider
from commitgen.prompts.system_prompts import BASE_SYSTEM_PROMPT

class ClaudeProvider(BaseProvider):
    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = config.get('api_key', '')
        self.model = config.get('model', 'claude-sonnet-4-5-20250929')
        self.max_tokens = config.get('max_tokens', 1024)
        
        if self.api_key:
            self.client = Anthropic(api_key=self.api_key)
        else:
            self.client = None
    
    def is_available(self) -> bool:
        """Check if API key is configured"""
        return bool(self.api_key and self.client)
    
    def get_model_name(self) -> str:
        return self.model
    
    def generate_commit_message(
        self, 
        diff: str, 
        file_summary: str,
        context: Optional[str] = None
    ) -> str:
        """Generate commit message using Claude API"""
        if not self.is_available():
            raise ValueError("Claude API key not configured")
        
        user_prompt = f"""File changes summary:
{file_summary}

Git diff:
{diff}

Generate a conventional commit message for these changes."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            system=BASE_SYSTEM_PROMPT
        )
        
        return response.content[0].text.strip()
