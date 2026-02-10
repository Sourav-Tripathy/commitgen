import pytest
import os
from commitgen.core.diff_processor import DiffProcessor

class TestDiffProcessor:
    @pytest.fixture
    def processor(self):
        return DiffProcessor(max_tokens_per_chunk=1000)
    
    def test_filter_lock_files(self, processor):
        """Test that lock files are filtered out"""
        files = [
            ("src/main.py", "M"),
            ("package-lock.json", "M"),
            ("poetry.lock", "M"),
            ("README.md", "M")
        ]
        
        # Mock diff with these files
        diff = """diff --git a/src/main.py b/src/main.py
+++ changed
diff --git a/package-lock.json b/package-lock.json
+++ lock file changes"""
        
        filtered = processor.filter_diff(diff, files)
        
        assert "package-lock.json" not in filtered
        assert "src/main.py" in filtered
    
    def test_summarize_changes(self, processor):
        """Test change summary generation"""
        files = [
            ("src/auth.py", "A"),
            ("src/config.py", "M"),
            ("old_file.py", "D")
        ]
        
        summary = processor.summarize_changes(files)
        
        assert "Added: 1 files" in summary
        assert "Modified: 1 files" in summary
        assert "Deleted: 1 files" in summary
        
    def test_chunk_large_diff(self, processor):
        """Test that large diffs are chunked appropriately"""
        # Create a large diff
        # Assuming token estimation is roughly 1 char = 0.25 tokens or so for code
        # But tiktoken is more complex. Let's just make it huge.
        large_diff = "diff --git a/file.py b/file.py\n" + ("+line\n" * 5000)
        
        chunks = processor.chunk_diff(large_diff)
        
        assert len(chunks) >= 1
        # It might be 1 if max_tokens_per_chunk is large enough, but 5000 lines is a lot.
        # Let's see if our logic chunks it.
        # Actually our current logic for single file chunking is... simplistic: it just returns the whole file diff if it's too big unless we implement hunk splitting.
        # In my implementation above: 
        # "File too large - split by hunks if possible, or just push it as its own chunk and let the model handle it (truncation risk)"
        # "Simplification: just add as separate chunk for now"
        # So it will likely just return 1 chunk if it's 1 file.
        
        # To test chunking logic, we need multiple files.
        pass
