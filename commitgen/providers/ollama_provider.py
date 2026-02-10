from typing import List, Optional
import ollama
from commitgen.providers.base import BaseProvider
from commitgen.core.diff_processor import DiffProcessor
from commitgen.prompts.system_prompts import (
    BASE_SYSTEM_PROMPT, 
    SMALL_MODEL_FILE_PROMPT, 
    SMALL_MODEL_COMBINE_PROMPT
)

class OllamaProvider(BaseProvider):
    def __init__(self, config: dict):
        super().__init__(config)
        self.host = config.get('host', 'http://localhost:11434')
        self.client = ollama.Client(host=self.host)
        self.model = config.get('model', 'llama3.2:1b')
        self.max_tokens_per_chunk = config.get('max_tokens_per_chunk', 1000)
        self.diff_processor = DiffProcessor(self.max_tokens_per_chunk)
        
    def is_available(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            self.client.list()
            return True
        except Exception:
            return False
    
    def get_model_name(self) -> str:
        return self.model
    
    def _is_small_model(self) -> bool:
        """
        Detect if model is small (<3B parameters) based on name
        Small models: llama3.2:1b, qwen2.5:0.5b, phi3:mini
        """
        small_indicators = ['1b', '2b', '0.5b', 'mini', 'small']
        return any(indicator in self.model.lower() for indicator in small_indicators)
    
    def _generate_with_chunking(
        self, 
        diff: str, 
        file_summary: str
    ) -> str:
        """
        For small models: chunk diff, generate messages per chunk, combine
        
        Strategy:
        1. Split diff by file
        2. Generate commit message for each file
        3. Combine into single coherent message
        """
        chunks = self.diff_processor.chunk_diff(diff)
        chunk_summaries = []
        
        for chunk in chunks:
            if not chunk.strip(): continue
            prompt = SMALL_MODEL_FILE_PROMPT.format(diff=chunk)
            response = self._call_ollama(prompt, "You are a helpful coding assistant.")
            chunk_summaries.append(response.strip())
            
        final_prompt = SMALL_MODEL_COMBINE_PROMPT.format(
            file_descriptions="\n".join([f"- {s}" for s in chunk_summaries])
        )
        
        return self._call_ollama(final_prompt, BASE_SYSTEM_PROMPT)
    
    def _generate_direct(
        self, 
        diff: str, 
        file_summary: str
    ) -> str:
        """For large models: send full diff"""
        prompt = f"""File changes summary:
{file_summary}

Git diff:
{diff}

Generate a conventional commit message for these changes."""
        return self._call_ollama(prompt, BASE_SYSTEM_PROMPT)
    
    def generate_commit_message(
        self, 
        diff: str, 
        file_summary: str,
        context: Optional[str] = None
    ) -> str:
        """
        Main entry point - routes to chunking or direct based on model size
        """
        # If diff is empty, maybe only file renames or deletions without content change
        if not diff.strip() and file_summary:
             # Just use file summary
             return self._call_ollama(
                 f"Generate a commit message based on this summary:\n{file_summary}", 
                 BASE_SYSTEM_PROMPT
             )

        if self._is_small_model(): # and token count > context window
            # simple check for now: always chunk for small models if diff is large enough
            # but wait, logic should be robust.
            return self._generate_with_chunking(diff, file_summary)
        else:
            return self._generate_direct(diff, file_summary)
    
    def _call_ollama(self, prompt: str, system_prompt: str) -> str:
        """Make API call to Ollama"""
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': 0.7,
                    'num_predict': 200,  # Max tokens for commit message
                }
            )
            return response['message']['content']
        except Exception as e:
            return f"Error generating message: {str(e)}"
