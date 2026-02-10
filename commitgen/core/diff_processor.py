from typing import List, Dict, Tuple
import re
import tiktoken

class DiffProcessor:
    def __init__(self, max_tokens_per_chunk: int = 1000):
        self.max_tokens_per_chunk = max_tokens_per_chunk
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT tokenizer
        except ValueError:
            self.encoding = tiktoken.get_encoding("gpt2") # Fallback
        
        # Patterns for filtering
        self.generated_patterns = [
            r'.*\.min\.js$',
            r'.*\.min\.css$',
            r'package-lock\.json$',
            r'yarn\.lock$',
            r'poetry\.lock$',
            r'Pipfile\.lock$',
            r'.*-lock\.json$',
            r'.*\.bundle\.js$',
            r'dist/.*',
            r'build/.*',
            r'.*\.generated\..*',
            r'__pycache__/.*'
        ]
    
    def filter_diff(self, diff: str, files: List[Tuple[str, str]]) -> str:
        """
        Filter out generated files, lock files from diff
        Args:
            diff: Full git diff
            files: List of (filename, change_type)
        Returns:
            Filtered diff
        """
        if not diff:
            return ""

        # Basic string based filtering for now based on file list
        # We need to construct a new diff string excluding filtered files.
        # This is tricky because diff string is one big blob.
        # Smarter way: iterate over hunks.
        
        # But wait, GitPython gives diff objects.
        # However, the input here is a string `diff`.
        
        # A simple approach: split by `diff --git a/`
        chunks = diff.split("diff --git a/")
        filtered_chunks = []
        
        for chunk in chunks:
            if not chunk.strip(): continue
            
            # Re-add prefix stripped by split
            full_chunk = f"diff --git a/{chunk}"
            
            # Extract filename from first line
            # format: path/to/file b/path/to/file
            first_line = chunk.split('\n')[0]
            filename = first_line.split(' b/')[0]
            
            # Check against patterns
            should_exclude = False
            for pattern in self.generated_patterns:
                if re.match(pattern, filename):
                    should_exclude = True
                    break
            
            if not should_exclude:
                filtered_chunks.append(full_chunk)
                
        return "".join(filtered_chunks)

    def extract_file_diffs(self, diff: str) -> Dict[str, str]:
        """
        Extract individual file diffs from combined diff
        Returns: Dict mapping filename -> file diff
        """
        chunks = diff.split("diff --git a/")
        file_diffs = {}
        
        for chunk in chunks:
            if not chunk.strip(): continue
            
            full_chunk = f"diff --git a/{chunk}"
            first_line = chunk.split('\n')[0]
            try:
                filename = first_line.split(' b/')[0]
                file_diffs[filename] = full_chunk
            except IndexError:
                continue
                
        return file_diffs

    def chunk_diff(self, diff: str) -> List[str]:
        """
        Split diff into chunks that fit in small model context
        Intelligently splits by file, then by hunk if needed
        
        Args:
            diff: Filtered git diff
        Returns:
            List of diff chunks
        """
        file_diffs = self.extract_file_diffs(diff)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for filename, file_diff in file_diffs.items():
            file_tokens = self.estimate_tokens(file_diff)
            
            if file_tokens > self.max_tokens_per_chunk:
                # File too large - split by hunks if possible, or just push it as its own chunk and let the model handle it (truncation risk)
                # Ideally split by hunks.
                # Simplification: just add as separate chunk for now
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                chunks.append(file_diff)
            else:
                # File fits
                if current_tokens + file_tokens > self.max_tokens_per_chunk:
                    # Current chunk full - start new one
                    chunks.append("\n".join(current_chunk))
                    current_chunk = [file_diff]
                    current_tokens = file_tokens
                else:
                    current_chunk.append(file_diff)
                    current_tokens += file_tokens
        
        if current_chunk:
             chunks.append("\n".join(current_chunk))
             
        return chunks

    def summarize_changes(self, files: List[Tuple[str, str]]) -> str:
        """
        Create high-level summary of changes
        
        Example output:
        - Added: 3 files (authentication.py, user_model.py, test_auth.py)
        - Modified: 2 files (config.py, README.md)
        - Deleted: 1 file (legacy_auth.py)
        """
        added = []
        modified = []
        deleted = []
        renamed = []
        
        for f, change_type in files:
            if change_type == 'A': added.append(f)
            elif change_type == 'M': modified.append(f)
            elif change_type == 'D': deleted.append(f)
            elif change_type == 'R': renamed.append(f)
            
        summary = []
        if added:
            summary.append(f"- Added: {len(added)} files ({', '.join(added[:3])}{'...' if len(added)>3 else ''})")
        if modified:
             summary.append(f"- Modified: {len(modified)} files ({', '.join(modified[:3])}{'...' if len(modified)>3 else ''})")
        if deleted:
             summary.append(f"- Deleted: {len(deleted)} files ({', '.join(deleted[:3])}{'...' if len(deleted)>3 else ''})")
        if renamed:
             summary.append(f"- Renamed: {len(renamed)} files ({', '.join(renamed[:3])}{'...' if len(renamed)>3 else ''})")
             
        return "\n".join(summary)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        return len(self.encoding.encode(text))
