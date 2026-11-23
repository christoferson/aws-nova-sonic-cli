# aws-nova-sonic-cli

Console speech-to-speech assistant on Amazon Nova 2 Sonic, written by hand against the raw
bidirectional event protocol (no framework). The Strands Agents version lives in the sister repo
`aws-nova-sonic-cli-strands`. Setup, usage and every feature are documented in README.md.

## Layout
- `nova_sonic.py`: main program (barge-in + date/time and calculator tools). Standalone on purpose;
  don't make it import from gallery/.
- `gallery/NN_*.py`: numbered feature showcases. Each builds on an earlier step and loads it with
  `importlib.import_module("02_audio_with_barge_in")`, because names starting with a digit can't
  be imported normally. Extension points in step 02: `tool_configuration()`, `on_event()`,
  `run_conversation(use_microphone=, until=)`; step 03: `execute_tool()`.
- `sonic/`: shared package installed into the venv by `uv sync` (credentials, CLI options,
  voices, default prompt, tools).
- `docs/nova-2-sonic/`: offline copies of the Nova 2 Sonic docs (saved 2026-11-22). Check API
  details there before guessing.

## Environment
- Windows 10, Git Bash, Python 3.12 via uv. Run with `uv run ...`; never pip.
- Credentials come from the AWS profile through boto3 (`Boto3CredentialsResolver` in
  sonic/common.py). Region us-east-1. Model `amazon.nova-2-sonic-v1:0`, default voice tiffany.
- Live runs need a mic and AWS access, so only the user can test them. Verify offline with fake
  streams (record sent events and feed in fake response events), and say what was not tested live.

## Lessons learned (from live testing)
- With tools configured, barge-in feels less responsive than in step 02.
- Keep the tool hint in the system prompt ("You have tools for... Use them instead of guessing");
  without it the model guessed the time instead of calling the tool.
- "(interrupted)" after most replies usually means speaker echo: use headphones. Muting the mic
  while the assistant speaks was tried and reverted (the assistant could no longer be stopped).
- Every toolUse must get a toolResult, even on error, or the conversation stalls.
- The calculator parses with `ast` and a whitelist; never use eval().
- Windows consoles can crash on some characters; sonic/common.py sets stdout errors="replace".

## Open questions
- The AWS price list shows a "Nova 2.5 Sonic" product; its model ID is unknown, so don't guess it.
- Unverified: whether typed (cross-modal) text is echoed back as a USER transcript.

## Working style
- User-facing console output: short, pre-written sentences about what the assistant can do.
  No counts, file paths or internals; developer detail goes behind --debug. Don't add libraries
  or code just for cosmetic output.
- After creating a feature, explain in plain language what it shows and how to try it, then ask
  what to build next.
- Stage changes, but only commit or push when asked. Never commit chat history or recordings
  (chat.json, *.wav).
