from typing import List, Tuple, Optional
from pathlib import Path
import git
from git import Repo

class GitHandler:
    def __init__(self, repo_path: Optional[str] = None):
        """Initialize with repository path (defaults to current directory)"""
        self.repo_path = repo_path or "."
        try:
            self.repo = Repo(self.repo_path, search_parent_directories=True)
        except git.exc.InvalidGitRepositoryError:
            print(f"Error: {self.repo_path} is not a valid git repository.")
            # Handle nicely later or let it crash if critical
            self.repo = None

    def has_staged_changes(self) -> bool:
        """Check if there are staged changes"""
        if not self.repo: return False
        try:
            # Check for staged changes against HEAD
            # diff(HEAD) checks differences between index and HEAD (staged changes)
            return len(self.repo.index.diff("HEAD")) > 0
        except git.exc.BadName:
            # If HEAD doesn't exist (e.g. initial commit)
            # Check if there are any files in the index
            # git ls-files --cached
            try:
                # If ls-files has output, we have staged files
                return bool(self.repo.git.ls_files(cached=True))
            except:
                return False
        except Exception:
            # Fallback to checking dirty status of index specifically
            try:
                return self.repo.is_dirty(index=True, working_tree=False)
            except:
                return False

    def get_staged_diff(self) -> str:
        """
        Get diff of staged changes
        Returns: Full diff string
        """
        if not self.repo: return ""
        try:
            # Diff index against HEAD (staged vs HEAD)
            diff = self.repo.git.diff("--cached", "--unified=3")
            return diff
        except git.exc.GitCommandError:
            # Maybe initial commit
            try:
                # Diff against empty tree for initial commit
                diff = self.repo.git.diff("--cached", "--unified=3", git.NULL_TREE) 
                return diff
            except:
                return ""

    def get_staged_files(self) -> List[Tuple[str, str]]:
        """
        Get list of staged files with their change type
        Returns: List of (filename, change_type) tuples
        change_type: A=Added, M=Modified, D=Deleted, R=Renamed
        """
        if not self.repo: return []
        files = []
        try:
            # Use diff against HEAD
            diffs = self.repo.index.diff("HEAD")
            if not diffs: # Maybe initial commit
                 diffs = self.repo.index.diff(git.NULL_TREE)
            
            for d in diffs:
                change_type = d.change_type
                filename = d.a_path if change_type == 'D' else d.b_path
                files.append((filename, change_type))
        except Exception:
             # Fallback if HEAD doesn't exist
             try:
                 diffs = self.repo.index.diff(git.NULL_TREE)
                 for d in diffs:
                    change_type = d.change_type
                    filename = d.a_path if change_type == 'D' else d.b_path
                    files.append((filename, change_type))
             except:
                 pass
        return files

    def get_branch_name(self) -> str:
        """Get current branch name"""
        if not self.repo: return ""
        try:
            return self.repo.active_branch.name
        except TypeError:
            return "DETACHED_HEAD"

    def get_last_commit_message(self) -> str:
        """Get the last commit message"""
        if not self.repo: return ""
        try:
            return self.repo.head.commit.message.strip()
        except:
            return ""

    def commit_with_message(self, message: str) -> bool:
        """Execute git commit with message"""
        if not self.repo: return False
        try:
            self.repo.index.commit(message)
            return True
        except Exception as e:
            print(f"Commit failed: {e}")
            return False

    def amend_commit(self, message: str) -> bool:
        """Amend the last commit with new message"""
        if not self.repo: return False
        try:
            self.repo.git.commit("--amend", "-m", message)
            return True
        except Exception as e:
            print(f"Amend failed: {e}")
            return False
