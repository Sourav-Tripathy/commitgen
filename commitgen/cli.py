import click
from rich.console import Console
from rich.prompt import Prompt, Confirm
from commitgen.core.message_generator import generate_commit_message
from commitgen.core.config_manager import ConfigManager
from commitgen.core.git_handler import GitHandler

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
def config(set_value, get_key):
    """Manage commitgen configuration"""
    manager = ConfigManager()
    
    if set_value:
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
    
    provider = Prompt.ask("Choose default provider", choices=["ollama", "claude", "openrouter"], default="ollama")
    manager.set("general.provider", provider)
    
    if provider == "ollama":
        model = Prompt.ask("Ollama model", default="llama3.2:1b")
        manager.set("ollama.model", model)
    elif provider == "claude":
        key = Prompt.ask("Claude API Key", password=True)
        manager.set("claude.api_key", key)
    elif provider == "openrouter":
        key = Prompt.ask("OpenRouter API Key", password=True)
        manager.set("openrouter.api_key", key)
        
    console.print("[green]Configuration saved![/green]")

@main.command()
def install_hook():
    """Install git pre-commit hook"""
    # Create .git/hooks/prepare-commit-msg
    import os
    import stat
    
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
