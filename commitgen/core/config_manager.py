from pathlib import Path
from typing import Dict, Any, Optional
import toml
from pydantic import BaseModel, Field
import os

class OllamaConfig(BaseModel):
    model: str = "llama3.2:1b"
    host: str = "http://localhost:11434"
    timeout: int = 30
    max_tokens_per_chunk: int = 1000

class ClaudeConfig(BaseModel):
    api_key: str = ""
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 1024

class OpenRouterConfig(BaseModel):
    api_key: str = ""
    model: str = "google/gemma-2-9b-it:free"
    site_url: str = ""
    site_name: str = "commitgen"

class GeneralConfig(BaseModel):
    provider: str = "ollama"
    max_length: int = 72
    conventional_commits: bool = True
    language: str = "en"

class AdvancedConfig(BaseModel):
    filter_generated_files: bool = True
    filter_lock_files: bool = True
    filter_minified: bool = True
    chunk_large_diffs: bool = True

class Config(BaseModel):
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    openrouter: OpenRouterConfig = Field(default_factory=OpenRouterConfig)
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)

class ConfigManager:
    def __init__(self):
        self.config_dir = Path.home() / ".config" / "commitgen"
        self.config_file = self.config_dir / "config.toml"
        self._config_cache: Optional[Config] = None
        
    def load(self) -> Config:
        """Load configuration from file or create default"""
        if self._config_cache:
            return self._config_cache

        if not self.config_file.exists():
            return self._create_default()

        try:
            with open(self.config_file, "r") as f:
                data = toml.load(f)
            self._config_cache = Config(**data)
            return self._config_cache
        except Exception:
            # If load fails, return default but warn? For now just default
            return self._create_default()
    
    def _create_default(self) -> Config:
        """Create and save default configuration"""
        default_config = Config()
        self.save(default_config)
        return default_config

    def save(self, config: Config) -> None:
        """Save configuration to file"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, "w") as f:
            toml.dump(config.model_dump(), f)
        self._config_cache = config
    
    def get(self, key: str) -> Any:
        """Get a specific config value (dot notation)"""
        config = self.load()
        parts = key.split(".")
        value = config
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            elif isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set a specific config value (dot notation)"""
        config = self.load()
        parts = key.split(".")
        target = config
        
        # Navigate to the parent of the target key
        for part in parts[:-1]:
            if hasattr(target, part):
                target = getattr(target, part)
            else:
                raise ValueError(f"Invalid config key: {key}")
        
        final_key = parts[-1]
        if hasattr(target, final_key):
            # Try to cast value to the correct type
            current_value = getattr(target, final_key)
            if isinstance(current_value, bool) and isinstance(value, str):
                value = value.lower() == "true"
            elif isinstance(current_value, int) and isinstance(value, str):
                value = int(value)
                
            setattr(target, final_key, value)
            self.save(config)
        else:
            raise ValueError(f"Invalid config key: {key}")
