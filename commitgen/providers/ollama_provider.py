from typing import List, Optional
import ollama
from commitgen.providers.base import BaseProvider
from commitgen.core.diff_processor import DiffProcessor
import subprocess
import time
import shutil
import sys
from commitgen.prompts.system_prompts import (
    BASE_SYSTEM_PROMPT, 
    SMALL_MODEL_FILE_PROMPT, 
    SMALL_MODEL_COMBINE_PROMPT
)

class OllamaProvider(BaseProvider):
    @staticmethod
    def get_installed_models(host: str = 'http://localhost:11434') -> List[str]:
        """Get list of installed Ollama models"""
        try:
            client = ollama.Client(host=host)
            models = client.list()
            # client.list() returns dictionary with 'models' key which is a list
            # Each item in list has 'name'
            if hasattr(models, 'models'): # Latest python client
                 return [m.model for m in models.models]
            elif isinstance(models, dict) and 'models' in models:
                 # Older format or raw API response
                 return [m['name'] for m in models['models']]
            return []
        except Exception:
            return []

    @staticmethod
    def pull_model(model_name: str, host: str = 'http://localhost:11434') -> bool:
        """Pull a model from Ollama library"""
        try:
            client = ollama.Client(host=host)
            # This might be slow and blocking. 
            # Ideally we want progress but the python client might stream.
            # For simplicity in this CLI tool, we just call it and wait or stream if possible.
            # client.pull returns an iterator of progress objects
            current_digest = ''
            for progress in client.pull(model_name, stream=True):
                # We could print progress here if we passed a callback
                # But for now just consume it
                pass
            return True
        except Exception as e:
            print(f"Error pulling model: {e}")
            return False

    def __init__(self, config: dict):
        super().__init__(config)
        self.host = config.get('host', 'http://localhost:11434')
        self.client = ollama.Client(host=self.host)
        self.model = config.get('model', 'llama3.2:1b')
        self.max_tokens_per_chunk = config.get('max_tokens_per_chunk', 1000)
        self.diff_processor = DiffProcessor(self.max_tokens_per_chunk)
        
    def is_available(self) -> bool:
        """Check if Ollama is running, if not try to start it."""
        try:
            self.client.list()
            return True
        except Exception:
            # Try to start it
            return self._start_ollama()

    def _start_ollama(self) -> bool:
        
        if not shutil.which("ollama"):
            return False
            
        print("[yellow]Ollama is not running. Attempting to start...[/yellow]")
        try:
             # Start in background, redirect output to devnull to avoid clutter
             subprocess.Popen(
                 ["ollama", "serve"], 
                 stdout=subprocess.DEVNULL, 
                 stderr=subprocess.DEVNULL,
                 start_new_session=True 
             )
             
             # Wait for it to come up
             print("Waiting for Ollama to initialize...", end="", flush=True)
             for _ in range(20): # Wait up to 20 seconds
                 time.sleep(1)
                 try:
                     self.client.list()
                     print(" Done.")
                     return True
                 except:
                     print(".", end="", flush=True)
             print("\nFailed to start Ollama (timeout).")
             return False
        except Exception as e:
            print(f"\nFailed to start Ollama: {e}")
            return False
    
    def get_model_name(self) -> str:
        return self.model
    
    def _is_small_model(self) -> bool:
        """
        Detect if model is small (<3B parameters) based on name
        Small models: llama3.2:1b, qwen2.5:0.5b, phi3:mini, deepseek-r1:1.5b
        """
        small_indicators = ['1b', '2b', '0.5b', '1.5b', 'mini', 'small']
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
        # Define base options
        options = {
            'temperature': 0.7,
            'num_predict': 200,
        }
        
        # Check if CPU mode is forced in config or if we should auto-retry
        # For now, let's just make the call.
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                options=options
            )
            return response['message']['content']
        except Exception as e:
            error_msg = str(e).lower()
            if "memory" in error_msg or "vram" in error_msg:
                # If memory error, try falling back to CPU
                from rich.console import Console
                Console().print("[yellow]Memory error detected. Retrying on CPU...[/yellow]")
                try:
                    # num_gpu=0 forces CPU
                    options['num_gpu'] = 0
                    response = self.client.chat(
                        model=self.model,
                        messages=[
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': prompt}
                        ],
                        options=options
                    )
                    return response['message']['content']
                except Exception as cpu_e:
                    raise cpu_e
            else:
                # Re-raise other errors
                raise e
