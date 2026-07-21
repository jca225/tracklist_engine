"""GitHub deploy-key helpers for ephemeral workers (private repo clone).

One-time operator setup (Mac + pi-storage/pi-worker if they git-pull):

    mkdir -p ~/.config/tracklist_engine
    ssh-keygen -t ed25519 -f ~/.config/tracklist_engine/github_deploy_key -N '' -C tracklist-engine-deploy
    # Add ~/.config/tracklist_engine/github_deploy_key.pub as a read-only Deploy key
    # on github.com/jca225/tracklist_engine → Settings → Deploy keys.

Private key path (first match wins):
  - ``TRACKLIST_GITHUB_DEPLOY_KEY`` env var
  - ``~/.config/tracklist_engine/github_deploy_key``
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_SSH_URL = "git@github.com:jca225/tracklist_engine.git"
REPO_HTTPS_URL = "https://github.com/jca225/tracklist_engine.git"

DEFAULT_DEPLOY_KEY_PATH = Path.home() / ".config/tracklist_engine/github_deploy_key"

DEPLOY_KEY_SETUP_HINT = (
    "Private repo clone needs a read-only deploy key:\n"
    "  mkdir -p ~/.config/tracklist_engine\n"
    "  ssh-keygen -t ed25519 -f ~/.config/tracklist_engine/github_deploy_key -N ''\n"
    "  Add github_deploy_key.pub → GitHub repo Settings → Deploy keys (read-only).\n"
    "  Or set TRACKLIST_GITHUB_DEPLOY_KEY to the private key path."
)


def resolve_deploy_key_path() -> Path | None:
    override = os.environ.get("TRACKLIST_GITHUB_DEPLOY_KEY", "").strip()
    if override:
        path = Path(override).expanduser()
        return path if path.is_file() else None
    return DEFAULT_DEPLOY_KEY_PATH if DEFAULT_DEPLOY_KEY_PATH.is_file() else None


def read_deploy_key_pem() -> str | None:
    path = resolve_deploy_key_path()
    if path is None:
        return None
    pem = path.read_text()
    return pem if pem.strip() else None


def remote_git_auth_shell(deploy_key_pem: str) -> str:
    """Remote shell: write deploy key + github.com SSH config."""
    if "DEPLOYKEY" in deploy_key_pem:
        raise ValueError("deploy key PEM must not contain DEPLOYKEY heredoc delimiter")
    return f"""mkdir -p ~/.ssh && chmod 700 ~/.ssh
cat > ~/.ssh/tracklist_engine_deploy <<'DEPLOYKEY'
{deploy_key_pem.rstrip()}
DEPLOYKEY
chmod 600 ~/.ssh/tracklist_engine_deploy
cat > ~/.ssh/config <<'SSHCFG'
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/tracklist_engine_deploy
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
SSHCFG
chmod 600 ~/.ssh/config"""


def clone_repo_url(*, deploy_key_pem: str | None) -> str:
    return REPO_SSH_URL if deploy_key_pem else REPO_HTTPS_URL
