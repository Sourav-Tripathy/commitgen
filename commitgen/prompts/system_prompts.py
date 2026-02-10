"""System prompts for different AI providers and scenarios"""

# Base prompt for all providers
BASE_SYSTEM_PROMPT = """You are an expert at writing git commit messages.
Generate concise, informative commit messages following the conventional commits format.

Format: <type>(<scope>): <description>

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- style: Code style changes (formatting, etc.)
- refactor: Code refactoring
- test: Adding or updating tests
- chore: Maintenance tasks
- perf: Performance improvements
- ci: CI/CD changes
- build: Build system changes

Rules:
1. Keep the first line under 72 characters
2. Use imperative mood ("add" not "added")
3. Don't capitalize the first letter after the colon
4. No period at the end of the subject line
5. Be specific but concise
6. Focus on WHAT changed and WHY, not HOW"""

# For small models with limited context
SMALL_MODEL_FILE_PROMPT = """Given this git diff for a single file, generate a brief description of what changed.
Focus on the main purpose of the change. Keep it to one sentence, under 10 words.

Example outputs:
- "Add user authentication"
- "Fix null pointer in login"
- "Update README with examples"
- "Refactor database queries"

Diff:
{diff}

Brief description:"""

SMALL_MODEL_COMBINE_PROMPT = """Given these file-level change descriptions, generate a single conventional commit message.

Changes:
{file_descriptions}

Generate a commit message in this format: <type>(<scope>): <description>

Commit message:"""

# For combining multiple chunks
CHUNK_COMBINE_PROMPT = """You are combining information from multiple code changes.
Generate a single, coherent conventional commit message that summarizes all changes.

Individual change descriptions:
{chunk_summaries}

Generate one conventional commit message that captures the overall change:"""

# For amending commits
AMEND_PROMPT = """Improve this commit message to be more clear and follow conventional commits format.

Current message:
{current_message}

Recent changes:
{file_summary}

Generate an improved commit message:"""
