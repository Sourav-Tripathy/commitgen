import click
from rich.console import Console
from rich.prompt import Prompt, Confirm
from commitgen.core.message_generator import generate_commit_message
from commitgen.core.config_manager import ConfigManager
from commitgen.core.git_handler import GitHandler
import ollama
import os
import stat

console = Console()

@click.group()
@click.version_option()
def main():
    """AI-powered git commit message generator"""
    pass

@main.command()
@click.option('--auto', is_flag=True, help='Automatically commit with generated message')
@click.option('--amend', is_flag=True, help='Amend the last commit')
@click.option('--provider', type=str, help='Override configured provider')
@click.option('--model', type=str, help='Override configured model')
def commit(auto, amend, provider, model):
    """Generate a commit message for staged changes"""
    try:
        if amend:
            console.print("[yellow]Wait, amend logic not fully implemented in CLI yet, generic generation for now...[/yellow]")
        
        with console.status("[bold green]Generating commit message..."):
            message = generate_commit_message(provider=provider, model=model)
        
        console.print("\n[bold]Generated Commit Message:[/bold]")
        console.print(f"[green]{message}[/green]\n")
        
        if auto:
            # Auto commit
            handler = GitHandler()
            if handler.commit_with_message(message):
                 console.print("[bold green]✓ Commit successful![/bold green]")
            else:
                 console.print("[bold red]✗ Commit failed![/bold red]")
        else:
            # Interactive confirmation
            action = Prompt.ask(
                "Do you want to commit with this message?",
                choices=["y", "n", "e"],
                default="y"
            )
            
            if action == "y":
                handler = GitHandler()
                if handler.commit_with_message(message):
                     console.print("[bold green]✓ Commit successful![/bold green]")
                else:
                     console.print("[bold red]✗ Commit failed![/bold red]")
            elif action == "e":
                # Edit functionality
                # Open editor with message
                # For now just prompt for new message
                new_message = click.edit(message)
                if new_message:
                    handler = GitHandler()
                    if handler.commit_with_message(new_message.strip()):
                         console.print("[bold green]✓ Commit successful![/bold green]")
                    else:
                         console.print("[bold red]✗ Commit failed![/bold red]")
                else:
                    console.print("[yellow]Commit aborted.[/yellow]")
            else:
                 console.print("[yellow]Commit aborted.[/yellow]")

    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
    except Exception as e:
        console.print(f"[bold red]Unexpected Error:[/bold red] {str(e)}")

@main.command()
@click.option('--set', 'set_value', type=str, help='Set config value (key=value)')
@click.option('--get', 'get_key', type=str, help='Get config value')
@click.option('--reset', is_flag=True, help='Reset configuration to defaults')
def config(set_value, get_key, reset):
    """Manage commitgen configuration"""
    manager = ConfigManager()
    
    if reset:
        if Confirm.ask("Are you sure you want to reset all configuration to defaults?"):
            manager._create_default()
            console.print("[green]Configuration reset to defaults.[/green]")
        else:
            console.print("[yellow]Reset aborted.[/yellow]")
            
    elif set_value:
        try:
            key, val = set_value.split('=', 1)
            manager.set(key, val)
            console.print(f"[green]Set {key} = {val}[/green]")
        except ValueError:
            console.print("[red]Invalid format. Use key=value[/red]")
        except Exception as e:
            console.print(f"[red]Error setting config: {e}[/red]")
            
    elif get_key:
        val = manager.get(get_key)
        console.print(f"{get_key} = {val}")
    else:
        # Show simplified config
        cfg = manager.load()
        console.print(cfg.model_dump_json(indent=2))

@main.command()
def init():
    """Interactive configuration wizard"""
    manager = ConfigManager()
    console.print("[bold]Welcome to CommitGen Setup![/bold]")
    
    provider = Prompt.ask("Choose default provider", choices=["ollama", "openrouter"], default="ollama")
    manager.set("general.provider", provider)
    
    if provider == "ollama":
        from commitgen.providers.ollama_provider import OllamaProvider
        
        console.print("[cyan]Checking Ollama status...[/cyan]")
        host = "http://localhost:11434"
        
        # Check if Ollama is running
        try:
            client = ollama.Client(host=host)
            client.list()
            is_running = True
        except Exception:
            is_running = False
            
        if is_running:
            models = OllamaProvider.get_installed_models(host)
            if models:
                # Add "Other" option or just allow typing if Prompt supports it?
                # Rich Prompt choices restricts input.
                # Let's show list and ask.
                console.print(f"[green]Found {len(models)} models:[/green]")
                for m in models:
                    console.print(f"- {m}")
                    
                model = Prompt.ask("Select Ollama model", choices=models, default=models[0])
            else:
                if Confirm.ask("Ollama is running but no models found. Download base model (llama3.2:1b)?", default=True):
                    console.print("[cyan]Downloading llama3.2:1b (this may take a while)...[/cyan]")
                    # We can use subprocess to call 'ollama pull' for better progress bar visibility
                    # because the python client streaming is basic here.
                    import subprocess
                    try:
                        subprocess.run(["ollama", "pull", "llama3.2:1b"], check=True)
                        console.print("[green]Model downloaded![/green]")
                        model = "llama3.2:1b"
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        # Fallback to python client or error
                        console.print("[yellow]CLI pull failed, trying API...[/yellow]")
                        if OllamaProvider.pull_model("llama3.2:1b", host):
                            console.print("[green]Model downloaded![/green]")
                            model = "llama3.2:1b"
                        else:
                            console.print("[red]Download failed.[/red]")
                            model = Prompt.ask("Enter model name manually", default="llama3.2:1b")
                else:
                    model = Prompt.ask("Enter model name manually", default="llama3.2:1b")
        else:
            console.print("[yellow]Ollama service not detected on localhost:11434[/yellow]")
            
            import shutil
            if shutil.which("ollama"):
                 console.print("Ollama is installed but not running.")
                 if Confirm.ask("Start Ollama service now?", default=True):
                     import subprocess
                     # Starting in background is tricky across platforms.
                     # "ollama serve" blocks.
                     console.print("Please start 'ollama serve' in another terminal.")
            else:
                 console.print("[red]Ollama is NOT installed.[/red]")
                 console.print("Please install Ollama from [link=https://ollama.ai]https://ollama.ai[/link]")
            
            model = Prompt.ask("Enter model name to use once Ollama is running", default="llama3.2:1b")

        manager.set("ollama.model", model)

    elif provider == "openrouter":
        key = Prompt.ask("OpenRouter API Key", password=True)
        manager.set("openrouter.api_key", key)
        
    console.print("[green]Configuration saved![/green]")

@main.command()
def install_hook():
    """Install git pre-commit hook"""
    # Create .git/hooks/prepare-commit-msg
    
    hook_content = """#!/bin/sh
# specific hook logic to call commitgen
# usually prepare-commit-msg or commit-msg
# For now, just a placeholder message
echo "CommitGen hook not fully implemented yet."
"""
    # Logic to find .git dir and write hook
    console.print("[yellow]Hook installation not fully implemented yet.[/yellow]")

if __name__ == "__main__":
    main()
