"""Collection-time policies that keep canonical trajectories trainable."""
from __future__ import annotations

import re

from cuamaster.benchmarks import NON_ENVIRONMENT_TOOL_NAMES as NON_ENVIRONMENT_TOOL_NAMES

MAX_TRAINING_ACTION_CHARS = 3000
HARD_CODED_SUDO_PASSWORD_RE = re.compile(
    r"(?:echo|printf(?:\s+%s)?)\s+['\"]?password['\"]?\s*\|\s*sudo|"
    r"sudo\s+-S[^\n;&]*<<<\s*['\"]?password['\"]?",
    re.I,
)
HIDDEN_APPLICATION_STATE_RE = re.compile(
    r"(?:\.config/(?:google-chrome|chromium|libreoffice)|\.thunderbird|"
    r"/Mail/Local Folders/|(?:openpyxl|python-docx|python-pptx|odfpy|zipfile)"
    r"[\s\S]{0,1500}\.(?:xlsx|xlsm|docx|pptx))",
    re.I,
)
COLLECTOR_TAMPERING_RE = re.compile(
    r"claudecode_shell_screenshot_hook|/tmp_workspace/_trajectory",
    re.I,
)
EVALUATION_ARTIFACT_ACCESS_RE = re.compile(
    r"(?:^|[\s/'\"])(?:_grade|score|quality|episode_manifest)\.json"
    r"(?:$|[\s'\"])",
    re.I,
)
NETWORK_WORKAROUND_RE = re.compile(
    r"(?:--proxy-server\b|\b(?:mitmproxy|tinyproxy|squid)\b|"
    r"\b(?:iptables|nft|nmcli)\b|/etc/hosts\b|"
    r"socket\.(?:socket|create_server)[\s\S]{0,1200}\.(?:bind|listen)\s*\(|"
    r"\bpython(?:3)?\s+-m\s+http\.server\b)",
    re.I,
)

COLLECTION_POLICY_MARKER = "=== TRAJECTORY DATA POLICY ==="
COLLECTION_POLICY = f"""

{COLLECTION_POLICY_MARKER}
- The collection action space is GUI plus CLI only. Use the computer tool and, when
  available, the shell tool. Do not use Read, Write, Edit, Glob, Grep, NotebookEdit,
  planning, sub-agent, user-question, task-management, or skill-loading tools.
- Never include passwords, API keys, tokens, or other credentials in a tool call.
  Do not run sudo; request no interactive confirmation.
- Do not create local proxy servers, modify network configuration, or alter the
  trajectory collector and its screenshot hooks. Install software only when the task
  explicitly requires it; never install helper packages just to bypass the UI.
- For GUI tasks, complete and verify the task through the visible application. Do not
  inspect hidden browser profiles, mail stores, application databases, or Office archive
  internals as a shortcut around the UI.
- Do not emit an empty typing action. Keep each tool-call payload below
  {MAX_TRAINING_ACTION_CHARS} characters; split legitimate large file writes into bounded
  append operations.
- A GUI tool call already returns a fresh screenshot. Do not add a routine wait after
  every action. Wait only while a visible asynchronous transition is still in progress.
=== END TRAJECTORY DATA POLICY ===
"""


def append_collection_policy(prompt: str) -> str:
    if COLLECTION_POLICY_MARKER in prompt:
        return prompt
    return prompt.rstrip() + COLLECTION_POLICY
