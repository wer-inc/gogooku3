"""
Git metadata utilities for dataset reproducibility.

Provides robust Git SHA and branch retrieval with environment variable fallbacks.
"""
import os
import subprocess
from typing import Dict


def get_git_metadata() -> Dict[str, str]:
    """
    Git metadata 取得 (例外安全)

    優先順位:
        1. 環境変数 DATASET_GIT_SHA / DATASET_GIT_BRANCH
        2. git コマンド実行
        3. フォールバック: "unknown"

    Returns:
        dict: {"git_sha": str, "git_branch": str}

    Examples:
        >>> # With environment variables
        >>> os.environ["DATASET_GIT_SHA"] = "abc123"
        >>> os.environ["DATASET_GIT_BRANCH"] = "main"
        >>> get_git_metadata()
        {"git_sha": "abc123", "git_branch": "main"}

        >>> # Without environment variables (git command)
        >>> get_git_metadata()
        {"git_sha": "0a89b26...", "git_branch": "feature/..."}

        >>> # Fallback when git is unavailable
        >>> get_git_metadata()
        {"git_sha": "unknown", "git_branch": "unknown"}
    """
    sha = os.getenv("DATASET_GIT_SHA")
    branch = os.getenv("DATASET_GIT_BRANCH")

    # Try git commands if environment variables are not set
    if not sha or not branch:
        try:
            if not sha:
                sha = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    text=True
                ).strip()

            if not branch:
                branch = subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    text=True
                ).strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback to "unknown" if git is unavailable or fails
            sha = sha or "unknown"
            branch = branch or "unknown"

    return {
        "git_sha": sha,
        "git_branch": branch
    }
