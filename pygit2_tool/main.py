"""
pygit2 Tool
===========
Interact with Git repositories using pygit2 (libgit2 Python bindings).

Supports the following actions:
  status          - Show working directory status (staged, unstaged, untracked files).
  log             - Return the commit log (history) for a branch or ref.
  diff            - Show diff between two commits/refs, or working tree vs HEAD.
  branches        - List local and/or remote branches.
  tags            - List all tags.
  stage           - Stage file paths (git add).
  unstage         - Unstage file paths (git reset HEAD <file>).
  commit          - Create a commit with a message.
  checkout        - Checkout a branch or commit.
  create_branch   - Create a new branch.
  delete_branch   - Delete a branch.
  remotes         - List configured remotes.
  blame           - Show per-line blame for a file.

Parameters:
  action        - The operation to perform (required).
  repo_path     - Absolute path to the repository (defaults to OPENCHAD_PROJECT_DIR).
  paths         - List of file paths for stage/unstage/blame operations.
  message       - Commit message (required for commit).
  ref           - Branch name, tag, or commit SHA used as a target for log/diff/checkout.
  ref_from      - Start ref for diff (defaults to HEAD).
  ref_to        - End ref for diff (defaults to working tree).
  max_commits   - Maximum commits to return for log (default 20).
  author_name   - Author display name for commit (falls back to repo config).
  author_email  - Author email for commit (falls back to repo config).
"""

import asyncio
import os
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pygit2
from pygit2.enums import SortMode
from openchadpy.tool_base import ToolBase

logger = logging.getLogger(__name__)

# Per-repo async locks to prevent concurrent write operations
_repo_locks: Dict[str, asyncio.Lock] = {}
_registry_lock = threading.Lock()

def _get_repo_lock(repo_path: str) -> asyncio.Lock:
    canonical = os.path.normcase(os.path.realpath(repo_path))
    with _registry_lock:
        if canonical not in _repo_locks:
            _repo_locks[canonical] = asyncio.Lock()
        return _repo_locks[canonical]

def _open_repo(repo_path: str) -> pygit2.Repository:
    discovered = pygit2.discover_repository(repo_path)
    return pygit2.Repository(discovered)

def _sig_to_dict(sig: Optional[pygit2.Signature]) -> Optional[Dict[str, Any]]:
    if sig is None:
        return None
    return {
        "name": sig.name,
        "email": sig.email,
        "time": datetime.fromtimestamp(sig.time, tz=timezone.utc).isoformat(),
    }

def _commit_to_dict(commit: pygit2.Commit) -> Dict[str, Any]:
    return {
        "sha": str(commit.id),
        "message": commit.message.strip(),
        "author": _sig_to_dict(commit.author),
        "committer": _sig_to_dict(commit.committer),
        "parents": [str(p) for p in commit.parent_ids],
    }


class Tool(ToolBase):
    name = "pygit2"
    description = (
        "Interact with Git repositories using pygit2: inspect status, history, diffs, "
        "branches, tags, remotes; stage/unstage files; commit; checkout; blame."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "status", "log", "diff", "branches", "tags",
                    "stage", "unstage", "commit", "checkout",
                    "create_branch", "delete_branch", "remotes", "blame",
                ],
                "description": "Git operation to perform.",
            },
            "repo_path": {
                "type": "string",
                "description": "Absolute path to the repository. Defaults to OPENCHAD_PROJECT_DIR.",
            },
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "File paths for stage / unstage / blame.",
            },
            "message": {
                "type": "string",
                "description": "Commit message (required for 'commit').",
            },
            "ref": {
                "type": "string",
                "description": "Branch name, tag, or commit SHA for log / checkout / create_branch / delete_branch.",
            },
            "ref_from": {
                "type": "string",
                "description": "Start ref for diff (default: HEAD).",
            },
            "ref_to": {
                "type": "string",
                "description": "End ref for diff (default: working tree).",
            },
            "max_commits": {
                "type": "integer",
                "description": "Maximum number of commits returned by log (default 20).",
            },
            "author_name": {
                "type": "string",
                "description": "Author display name for commit.",
            },
            "author_email": {
                "type": "string",
                "description": "Author email for commit.",
            },
        },
        "required": ["action"],
    }
    allowed_callers = ["direct", "code_execution"]

    async def execute(self, **kwargs) -> Dict[str, Any]:
        action: str = kwargs.get("action", "").strip()
        repo_path: str = (
            kwargs.get("repo_path")
            or os.environ.get("OPENCHAD_PROJECT_DIR")
            or os.getcwd()
        ).strip()

        if not action:
            return {"error": "action is required."}

        lock = _get_repo_lock(repo_path)
        await lock.acquire()
        try:
            return await asyncio.to_thread(self._run_sync, action, repo_path, kwargs)
        except Exception as e:
            logger.error(f"[pygit2] Action '{action}' failed: {e}", exc_info=True)
            return {"error": str(e)}
        finally:
            lock.release()

    # ------------------------------------------------------------------
    # Synchronous dispatch (runs in threadpool)
    # ------------------------------------------------------------------
    def _run_sync(self, action: str, repo_path: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            repo = _open_repo(repo_path)
        except Exception as e:
            return {"error": f"Could not open repository at '{repo_path}': {e}"}

        dispatch = {
            "status":        self._action_status,
            "log":           self._action_log,
            "diff":          self._action_diff,
            "branches":      self._action_branches,
            "tags":          self._action_tags,
            "stage":         self._action_stage,
            "unstage":       self._action_unstage,
            "commit":        self._action_commit,
            "checkout":      self._action_checkout,
            "create_branch": self._action_create_branch,
            "delete_branch": self._action_delete_branch,
            "remotes":       self._action_remotes,
            "blame":         self._action_blame,
        }
        handler = dispatch.get(action)
        if handler is None:
            return {"error": f"Unknown action '{action}'."}
        return handler(repo, kwargs)

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------
    def _action_status(self, repo: pygit2.Repository, kwargs: Dict) -> Dict:
        staged, unstaged, untracked = [], [], []
        for path, flags in repo.status().items():
            if flags & (pygit2.GIT_STATUS_INDEX_NEW | pygit2.GIT_STATUS_INDEX_MODIFIED | pygit2.GIT_STATUS_INDEX_DELETED):
                staged.append(path)
            if flags & (pygit2.GIT_STATUS_WT_MODIFIED | pygit2.GIT_STATUS_WT_DELETED):
                unstaged.append(path)
            if flags & pygit2.GIT_STATUS_WT_NEW:
                untracked.append(path)
        return {
            "staged": staged,
            "unstaged": unstaged,
            "untracked": untracked,
            "clean": not staged and not unstaged and not untracked,
        }

    def _action_log(self, repo: pygit2.Repository, kwargs: Dict) -> Dict:
        ref = kwargs.get("ref") or "HEAD"
        max_commits = int(kwargs.get("max_commits") or 20)
        try:
            head_commit = repo.revparse_single(ref).peel(pygit2.Commit)
        except Exception as e:
            return {"error": f"Could not resolve ref '{ref}': {e}"}
        commits = []
        for commit in repo.walk(head_commit.id, SortMode.TIME):
            commits.append(_commit_to_dict(commit))
            if len(commits) >= max_commits:
                break
        return {"commits": commits, "count": len(commits)}

    def _action_diff(self, repo: pygit2.Repository, kwargs: Dict) -> Dict:
        ref_from = kwargs.get("ref_from") or "HEAD"
        ref_to = kwargs.get("ref_to")

        try:
            commit_from = repo.revparse_single(ref_from).peel(pygit2.Commit)
        except Exception as e:
            return {"error": f"Could not resolve ref_from '{ref_from}': {e}"}

        if ref_to:
            try:
                commit_to = repo.revparse_single(ref_to).peel(pygit2.Commit)
                diff = commit_from.tree.diff_to_tree(commit_to.tree)
            except Exception as e:
                return {"error": f"Could not resolve ref_to '{ref_to}': {e}"}
        else:
            # Diff HEAD tree vs working directory
            diff = commit_from.tree.diff_to_workdir()

        diff.find_similar()
        patches = []
        for patch, delta in zip(diff, diff.deltas):
            if patch is not None:
                patches.append({
                    "old_file": patch.delta.old_file.path,
                    "new_file": patch.delta.new_file.path,
                    "status": patch.delta.status_char(),
                    "additions": patch.line_stats[1],
                    "deletions": patch.line_stats[2],
                    "patch": patch.text,
                })
            else:
                patches.append({
                    "old_file": delta.old_file.path,
                    "new_file": delta.new_file.path,
                    "status": delta.status_char(),
                    "additions": 0,
                    "deletions": 0,
                    "patch": None,
                })
        return {
            "files_changed": len(patches),
            "patches": patches,
        }

    def _action_branches(self, repo: pygit2.Repository, kwargs: Dict) -> Dict:
        local = [b for b in repo.branches.local]
        remote = [b for b in repo.branches.remote]
        head = None
        try:
            head = repo.head.shorthand
        except Exception:
            pass
        return {"local": local, "remote": remote, "head": head}

    def _action_tags(self, repo: pygit2.Repository, kwargs: Dict) -> Dict:
        prefix = "refs/tags/"
        tags = []
        for ref_name in repo.references:
            if ref_name.startswith(prefix):
                tag_name = ref_name[len(prefix):]
                ref = repo.references[ref_name]
                target = str(ref.target)
                tags.append({"name": tag_name, "target": target})
        return {"tags": tags, "count": len(tags)}

    def _action_stage(self, repo: pygit2.Repository, kwargs: Dict) -> Dict:
        paths: Optional[List[str]] = kwargs.get("paths")
        if not paths:
            return {"error": "'paths' is required for 'stage' action."}
        index = repo.index
        index.read()
        for p in paths:
            index.add(p)
        index.write()
        return {"success": True, "staged": paths}

    def _action_unstage(self, repo: pygit2.Repository, kwargs: Dict) -> Dict:
        paths: Optional[List[str]] = kwargs.get("paths")
        if not paths:
            return {"error": "'paths' is required for 'unstage' action."}
        try:
            head_commit = repo.head.peel(pygit2.Commit)
        except Exception:
            # No commits yet — just remove from index
            index = repo.index
            index.read()
            for p in paths:
                try:
                    index.remove(p)
                except Exception:
                    pass
            index.write()
            return {"success": True, "unstaged": paths}

        repo.index.read()
        for p in paths:
            # Reset each path to HEAD tree entry
            try:
                entry = head_commit.tree[p]
                repo.index.add(pygit2.IndexEntry(p, entry.id, entry.filemode))
            except KeyError:
                # File not in HEAD — remove from index
                try:
                    repo.index.remove(p)
                except Exception:
                    pass
        repo.index.write()
        return {"success": True, "unstaged": paths}

    def _action_commit(self, repo: pygit2.Repository, kwargs: Dict) -> Dict:
        message: Optional[str] = kwargs.get("message")
        if not message:
            return {"error": "'message' is required for 'commit' action."}

        index = repo.index
        index.read()
        tree_id = index.write_tree()

        # Resolve author
        author_name = kwargs.get("author_name")
        author_email = kwargs.get("author_email")
        try:
            cfg = repo.config
            if not author_name:
                author_name = cfg["user.name"]
            if not author_email:
                author_email = cfg["user.email"]
        except Exception:
            pass
        author_name = author_name or "Unknown"
        author_email = author_email or "unknown@example.com"

        sig = pygit2.Signature(author_name, author_email)

        parents = []
        try:
            parents = [repo.head.target]
        except pygit2.GitError:
            pass  # initial commit — no parents

        commit_id = repo.create_commit(
            "HEAD",
            sig,
            sig,
            message,
            tree_id,
            parents,
        )
        return {"success": True, "sha": str(commit_id), "message": message}

    def _action_checkout(self, repo: pygit2.Repository, kwargs: Dict) -> Dict:
        ref = kwargs.get("ref")
        if not ref:
            return {"error": "'ref' is required for 'checkout' action."}
        try:
            branch = repo.branches.get(ref)
            if branch:
                repo.checkout(branch)
                repo.set_head(branch.name)
                return {"success": True, "ref": ref, "type": "branch"}
            # Try as a commit/tag
            obj = repo.revparse_single(ref)
            repo.checkout_tree(obj)
            repo.set_head(str(obj.id))
            return {"success": True, "ref": ref, "type": "detached"}
        except Exception as e:
            return {"error": f"Checkout failed: {e}"}

    def _action_create_branch(self, repo: pygit2.Repository, kwargs: Dict) -> Dict:
        ref = kwargs.get("ref")
        if not ref:
            return {"error": "'ref' is required as the new branch name for 'create_branch'."}
        try:
            head_commit = repo.head.peel(pygit2.Commit)
            new_branch = repo.create_branch(ref, head_commit, False)
            return {"success": True, "branch": new_branch.branch_name}
        except Exception as e:
            return {"error": f"Failed to create branch '{ref}': {e}"}

    def _action_delete_branch(self, repo: pygit2.Repository, kwargs: Dict) -> Dict:
        ref = kwargs.get("ref")
        if not ref:
            return {"error": "'ref' is required as the branch name for 'delete_branch'."}
        try:
            branch = repo.branches.get(ref)
            if branch is None:
                return {"error": f"Branch '{ref}' not found."}
            branch.delete()
            return {"success": True, "deleted": ref}
        except Exception as e:
            return {"error": f"Failed to delete branch '{ref}': {e}"}

    def _action_remotes(self, repo: pygit2.Repository, kwargs: Dict) -> Dict:
        remotes = [{"name": r.name, "url": r.url} for r in repo.remotes]
        return {"remotes": remotes, "count": len(remotes)}

    def _action_blame(self, repo: pygit2.Repository, kwargs: Dict) -> Dict:
        paths: Optional[List[str]] = kwargs.get("paths")
        if not paths or len(paths) == 0:
            return {"error": "'paths' must contain exactly one file path for 'blame'."}
        file_path = paths[0]
        try:
            blame = repo.blame(file_path)
        except Exception as e:
            return {"error": f"Blame failed for '{file_path}': {e}"}
        hunks = []
        for hunk in blame:
            hunks.append({
                "lines_in_hunk": hunk.lines_in_hunk,
                "final_commit_sha": str(hunk.final_commit_id),
                "final_start_line": hunk.final_start_line_number,
                "orig_commit_sha": str(hunk.orig_commit_id),
                "orig_start_line": hunk.orig_start_line_number,
                "boundary": hunk.boundary,
            })
        return {"file": file_path, "hunks": hunks, "hunk_count": len(hunks)}
