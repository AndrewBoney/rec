# GitHub Copilot / Assistant Instructions for this Repository

Purpose
- Help maintain and extend a lightweight, end-to-end recommendation example focused on dual-encoder architectures.
- Prioritize clarity, minimalism, and reproducibility over adding complex features.

High-level goals
- Produce small, well-tested, and well-documented changes.
- Keep the codebase readable (< ~5k lines) and consistent with existing style.
- Preserve model checkpoints and training artifacts unless the user asks to modify them.

Repository entry points
- Data preparation: `rec/data_prep/dummy.py`
- Training orchestration: `rec/train_all.py`
- Retrieval training: `rec/retrieval/train.py`
- Ranking training: `rec/ranking/train.py`
- Shared utilities and configs: `rec/common/*`

Environment & running
- Use the Python environment from `requirements.txt`.
- Typical commands (adjust to user's environment):

```
python rec/train_all.py
python rec/retrieval/train.py
python rec/ranking/train.py
```

Coding conventions
- Keep changes minimal and focused to solve the task.
- Prefer explicit, clear code over clever abstractions.
- Add small, focused docstrings and type hints for public functions.
- Follow existing import and naming styles used in `rec/common`.

Testing & verification
- When adding code, include a small runnable example or a script that demonstrates the change.
- If you modify logic, prefer adding a unit test or a short smoke script that the user can run locally.
- Avoid running training-heavy tasks automatically in PRs; use quick smoke tests instead.

Commit and PR guidance
- Make small commits with focused messages.
- Explain rationale in the PR description; include a short checklist of manual verification steps.

What to avoid
- Do not modify or delete files under `lightning_logs/` or model checkpoint files in `version_*` unless explicitly requested.
- Avoid large refactors unless the user asks for them; propose them first.
- Don’t add heavy third-party dependencies without user approval.

When asked to generate code
- Ask for scope (file, function, or repo-wide) only if ambiguous.
- Prefer providing a complete, runnable change: source file(s), a small runner/test, and updated `requirements.txt` only when needed.
- Keep the change surface area small; explain any tradeoffs succinctly.

Developer notes for the assistant
- If the user asks for multi-step work, create a TODO plan and update it as changes are made.
- Prefer `apply_patch` for edits and `create_file` for new files; validate edits by running quick syntax checks when possible.
- When referencing files in messages, link to them relative to the workspace root (e.g., `instructions.md`).

Contact the user for:
- Adding new dependencies.
- Large refactors or dataset changes.
- Permission to modify or remove existing model checkpoints.


---
This file is intended to help GitHub Copilot / the assistant act as a careful, conservative collaborator for the `rec` repository.