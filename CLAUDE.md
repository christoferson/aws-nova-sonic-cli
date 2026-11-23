# aws-nova-sonic-cli

Console speech-to-speech assistant on Amazon Nova 2 Sonic, written by hand against the raw
bidirectional event protocol (no agent framework). README.md covers setup and usage.

## Layout
```
nova_sonic.py      Main program: barge-in + date/time and calculator tools
gallery/           Numbered feature showcases, simplest first
  01_audio_plain.py                      Whole event protocol in one file
  02_audio_with_barge_in.py              Core classes every later step builds on
  03_audio_with_tools.py                 Tool use (builds on 02)
  04_audio_with_text_dual_mode_input.py  Typed input + chat history (builds on 03)
  05..12_*.py                            One feature each: turn-taking, session continuation,
                                         assistant speaks first, usage/cost, knowledge base,
                                         forced tool use, 8 kHz audio, multi-step tool workflow
sonic/             Shared package, installed into the venv by uv sync
  common.py        Credentials, Bedrock client, audio format, voices, default prompt, CLI options
  tools.py         Tool registry: handlers, descriptions, JSON schemas
docs/nova-2-sonic/ Offline copies of the Nova 2 Sonic docs; check API details here first
examples/          Sample chat history for step 04
```

## Design
- One long-lived bidirectional stream per conversation. Everything is a JSON event; every piece
  of content is a contentStart / content / contentEnd block with its own contentName, all under
  one promptName. Step 01's module docstring has the full event sequence.
- Concurrency is asyncio on one event loop: a receive task, a mic sender, and a playback task.
  Blocking PyAudio calls run in executor threads; the mic callback hands chunks to the loop with
  call_soon_threadsafe into a FIFO queue, so order is kept.
- Step 02 splits the work into `BedrockStreamManager` (protocol: send events, dispatch responses,
  barge-in) and `AudioStreamer` (devices). `run_conversation()` wires them together and ends on
  the stop condition, Ctrl+C, or the server closing the stream.
- Barge-in: on an interruption signal, drop all queued audio and bump `interrupt_count`; playback
  writes small slices and stops mid-chunk when the counter changes.
- Tools: every toolUse is run in its own task (asynchronous tool calling) and always answered
  with a toolResult, even on error, or the conversation stalls. Results of concurrent tools are
  sent under a lock so their three events don't interleave.

## Extending
- Gallery steps subclass earlier steps instead of copying them, and load them with
  `importlib.import_module("02_audio_with_barge_in")`, since names starting with a digit can't
  be imported normally. Extension points:
  - step 02: `tool_configuration()`, `on_event(event_type, event)`, overriding `send_event` /
    `send_audio_chunk`, `run_conversation(manager, use_microphone=, until=)`
  - step 03: `execute_tool(name, arguments)`
- New gallery step: next number, `NN_audio_with_<feature>.py`, a module docstring explaining the
  feature and the events involved, then add it to the README layout and run list.
- New tool: an async handler returning a dict plus an entry in `TOOLS` in sonic/tools.py.
- `nova_sonic.py` is standalone on purpose: it imports only from sonic/, never from gallery/, so
  the showcases can change without breaking it.

## Constraints
- Python 3.12 via uv (`uv run ...`). Credentials come from the AWS profile through boto3
  (`Boto3CredentialsResolver`), never from hard-coded keys. Region us-east-1, model
  `amazon.nova-2-5-sonic` (Nova 2.5 Sonic; `amazon.nova-2-sonic-v1:0` via --model-id).
- With tools, keep a tool hint in the system prompt ("Use them instead of guessing"); without it
  the model answers from memory instead of calling them.
- The calculator parses with `ast` and a whitelist; never use eval().
- stdout is reconfigured with errors="replace" (sonic/common.py) because Windows consoles can't
  encode every character the model outputs.
- Live runs need a mic and AWS access. Verify changes offline with fake streams (record sent
  events, feed in fake response events).
