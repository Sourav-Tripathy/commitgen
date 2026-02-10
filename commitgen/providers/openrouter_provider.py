from typing import Optional
from openai import OpenAI
from commitgen.providers.base import BaseProvider
from commitgen.prompts.system_prompts import BASE_SYSTEM_PROMPT

class OpenRouterProvider(BaseProvider):
    """
    OpenRouter provider - access to free models like:
    - google/gemma-2-9b-it:free
    - meta-llama/llama-3.1-8b-instruct:free
    - microsoft/phi-3-mini-128k-instruct:free
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = config.get('api_key', '')
        self.model = config.get('model', 'google/gemma-3-27b-it:free')
        self.site_url = config.get('site_url', '')
        self.site_name = config.get('site_name', 'commitgen')
        
        if self.api_key:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key
            )
        else:
            self.client = None
    
    def is_available(self) -> bool:
        return bool(self.api_key and self.client)
    
    def get_model_name(self) -> str:
        return self.model
    
    def generate_commit_message(
        self, 
        diff: str, 
        file_summary: str,
        context: Optional[str] = None
    ) -> str:
        """Generate commit message using OpenRouter API"""
        if not self.is_available():
            raise ValueError("OpenRouter API key not configured")
        
        user_prompt = f"""File changes:
{file_summary}

Diff:
{diff[:3000]}

Generate a conventional commit message."""

        extra_headers = {}
        if self.site_url:
            extra_headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            extra_headers["X-Title"] = self.site_name

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": BASE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            extra_headers=extra_headers,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
