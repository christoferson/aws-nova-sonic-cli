# aws-nova-sonic-cli

Console speech-to-speech AI agent using [Amazon Nova 2 Sonic](https://aws.amazon.com/blogs/aws/introducing-amazon-nova-2-sonic-next-generation-speech-to-speech-model-for-conversational-ai/) on Amazon Bedrock. You speak into your microphone, the model answers out loud, and both transcripts are printed in the terminal.

Based on the AWS [console-python samples](https://github.com/aws-samples/amazon-ai-samples/tree/main/speech-to-speech/amazon-nova-2-sonic/sample-codes/console-python).

## Project layout

```
nova_sonic.py                              Main program: voice assistant with barge-in and tools
gallery/                                   Feature showcases, in order of difficulty
  01_audio_plain.py                        Mic in, voice out, transcripts; the whole event protocol in one file
  02_audio_with_barge_in.py                Interrupt the assistant mid-answer; --debug event logging
  03_audio_with_tools.py                   Builds on 02: the model calls Python tools and speaks the results
  04_audio_with_text_dual_mode_input.py    Builds on 03: speak or type, chat history; tools off unless --tools
  05_audio_with_turn_taking.py             Builds on 02: set how long a pause ends your turn, and time each reply
  06_audio_with_session_continuation.py    Builds on 02: keep talking past the 8-minute limit by switching sessions
  07_audio_with_assistant_speaks_first.py  Builds on 02: the assistant greets you as soon as it connects
  08_audio_with_usage_tracking.py          Builds on 02: tokens and estimated cost per reply and per session
sonic/                                     Shared package (installed into the venv by uv sync)
  common.py                                AWS profile credentials, CLI options, voices, default prompt
  tools.py                                 Tool implementations and JSON schemas
examples/chat_history.json                 Sample history for 04
docs/                                      Offline copies of the Nova 2 Sonic docs
```

`nova_sonic.py` is standalone, so it can evolve without breaking the gallery. Each gallery step builds on the previous one, so reading them in order is a guided tour of the API.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (installs Python 3.12 for you if it's missing)
- A microphone and speakers. **Use headphones**, otherwise the model can hear itself and interrupt its own answer.
- An AWS account with access to the `amazon.nova-2-sonic-v1:0` model in Amazon Bedrock, in a region where it's available (default `us-east-1`).
- IAM permission `bedrock:InvokeModelWithBidirectionalStream`.

## Setup

```bash
uv sync
```

This creates `.venv` with Python 3.12 (pinned in `.python-version`) and installs the dependencies from `uv.lock`.

Without uv, use the pinned `requirements.txt` (Python 3.12+):

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt   # also installs the local `sonic` package
python nova_sonic.py
```

After changing dependencies with `uv add`, regenerate it:

```bash
uv export --format requirements-txt --no-hashes -o requirements.txt
```

## AWS credentials and region

Credentials are resolved by boto3 from your AWS profile, so no keys need to be pasted into the shell. SSO and assume-role profiles work and are refreshed automatically.

Profile selection: `--profile <name>`, else `$AWS_PROFILE`, else `default`.

For an SSO profile, log in first:

```bash
aws sso login --profile <your-profile>
```

Region selection: `--region <region>`, else `$AWS_REGION`, else `us-east-1`. The profile's configured region is **not** used, because Nova 2 Sonic is only available in some regions.

Git Bash:

```bash
export AWS_PROFILE=<your-profile>
export AWS_REGION=us-east-1
```

PowerShell:

```powershell
$env:AWS_PROFILE="<your-profile>"
$env:AWS_REGION="us-east-1"
```

## Run

Main program:

```bash
uv run nova_sonic.py           # barge-in + tools (date/time, calculator)
uv run nova_sonic.py --debug   # with event-level logging
```

Gallery:

```bash
uv run gallery/01_audio_plain.py
uv run gallery/02_audio_with_barge_in.py
uv run gallery/03_audio_with_tools.py
uv run gallery/04_audio_with_text_dual_mode_input.py
uv run gallery/05_audio_with_turn_taking.py --sensitivity HIGH
uv run gallery/06_audio_with_session_continuation.py --switch-after 30
uv run gallery/07_audio_with_assistant_speaks_first.py
uv run gallery/08_audio_with_usage_tracking.py
```

Then start talking. Press **Enter** (or Ctrl+C) to end the session.

From step 02 on, and in `nova_sonic.py`, you can start speaking while the assistant is talking. It stops immediately, prints `(interrupted)`, and answers your new input. Use headphones so it doesn't interrupt itself.

Options (all scripts):

| Flag | Default | Description |
| --- | --- | --- |
| `--profile` | `$AWS_PROFILE` or `default` | AWS profile for credentials |
| `--region` | `$AWS_REGION` or `us-east-1` | Bedrock region |
| `--model-id` | `amazon.nova-2-sonic-v1:0` | Model ID |
| `--voice` | `tiffany` | Output voice (see below) |
| `--list-voices` | | Print available voices and exit |
| `--system-prompt` | voice assistant prompt | Replace the default prompt. The default tells the model it is speaking out loud, matches its gender to the voice, and has it reply in the language you speak |
| `--debug` | off | All except step 01: log events, stop reasons, token usage, tool results |

Example:

```bash
uv run nova_sonic.py --voice matthew --system-prompt "You are a friendly travel guide. Keep answers short."
```

## Tools

Tools are defined in `sonic/tools.py`. `nova_sonic.py` enables `get_date_and_time` and `calculate` (see `ENABLED_TOOLS`), and gallery steps 03 and 04 enable all three. Each call is printed as `[tool] name(args)` along with how long it took.

| Tool | What it does | Try saying |
| --- | --- | --- |
| `get_date_and_time` | Current date, time and weekday, locally or in any IANA timezone | "What time is it in New York?" |
| `get_weather` | Current weather and 3-day forecast for a city, from [Open-Meteo](https://open-meteo.com/) (free, no API key, needs internet) | "Do I need an umbrella in Osaka tomorrow?" |
| `calculate` | Exact arithmetic (`+ - * / // % **`, `sqrt`, `log`, trig, `pi`, `e`), parsed safely without `eval` | "What's 15 percent of 2,400?" |

Tools run in the background, so the model keeps talking ("let me check the weather...") while it waits. This is Nova 2 Sonic's asynchronous tool calling. Every tool call gets a result, even on error, because the model waits for one.

To add a tool, write an async function and add an entry to `TOOLS` in `sonic/tools.py` with a description and JSON schema. To use it in `nova_sonic.py`, add its name to `ENABLED_TOOLS`.

## Text input and chat history

`gallery/04_audio_with_text_dual_mode_input.py` adds Nova 2 Sonic's cross-modal input. The tools above are off by default; pass `--tools` to enable them. You can type messages as well as speak, and the assistant always answers out loud.

| Flag | Default | Description |
| --- | --- | --- |
| `--mode` | `mixed` | `mixed`: speak or type. `text`: type only, no microphone. `audio`: speak only |
| `--history FILE` | | Chat history file. Loaded at start if it exists (so the model remembers it), and this conversation is appended to it on exit |
| `--no-save` | off | Load `--history` but leave the file unchanged |
| `--tools` | off | Enable the date/time, weather and calculator tools from step 03 |

In `mixed` and `text` modes, type a message and press Enter to send it. Type `/quit` or press Ctrl+C to end. In `text` mode the script streams silence instead of microphone audio, because the session needs continuous audio input.

Keep one conversation going across sessions by passing the same file every time:

```bash
uv run gallery/04_audio_with_text_dual_mode_input.py --history chat.json   # first run creates chat.json on exit
uv run gallery/04_audio_with_text_dual_mode_input.py --history chat.json   # later runs remember it, then append to it
```

Try the bundled example without modifying it:

```bash
uv run gallery/04_audio_with_text_dual_mode_input.py --history examples/chat_history.json --no-save   # ask "what's my name?"
```

The history file is a JSON list of `{"role": "USER" | "ASSISTANT", "content": "..."}` messages. Consecutive messages from the same role are merged, because history must alternate. The file keeps everything. If it grows past Nova Sonic's 200KB limit, only the newest messages are sent to the model. The saved transcript uses the model's final assistant text plus your recognized speech or typed messages.

If you see `(interrupted)` after most answers when you didn't interrupt, the microphone is probably picking up the assistant's voice from your speakers. Use headphones.

## Turn-taking sensitivity

`gallery/05_audio_with_turn_taking.py` sets `endpointingSensitivity`, which controls how long a pause Nova Sonic waits for before it decides you've finished speaking and answers.

| `--sensitivity` | Pause before answering | Good for |
| --- | --- | --- |
| `HIGH` | ~1.5 s | Quick Q&A and commands; may cut in if you pause mid-sentence |
| `MEDIUM` (default) | ~1.75 s | General conversation |
| `LOW` | ~2.0 s | People who pause to think |

After each reply starts, it prints `[turn] reply started 1.9s after you stopped speaking`. That time is the pause plus model and network latency. It's estimated from mic loudness, so use headphones in a quiet room. If background noise skews it, raise `SPEECH_RMS_THRESHOLD` in the script.

## Long conversations (session continuation)

Nova Sonic cuts every connection after about 8 minutes. `gallery/06_audio_with_session_continuation.py` keeps one conversation going by moving it to a fresh session before then:

1. After `--switch-after` seconds (default 360), it waits for a quiet moment: the reply has finished and played, and nobody is speaking. If there's no quiet moment before 7:30, it switches anyway.
2. It opens a new session and replays the conversation so far as chat history.
3. It hands the microphone over. Audio from the last 1.5 s and from during the switch is replayed to the new session, so nothing you say is lost.
4. It closes the old session in the background.

You'll see `[session] session 2 active: 6 history messages, 1.5s of audio replayed`. Use `--switch-after 30` to watch it switch every half minute. Then ask "what did I say earlier?" after a switch to check the context carried over.

## Assistant speaks first

`gallery/07_audio_with_assistant_speaks_first.py` makes the assistant open the conversation, the way a phone agent answers a call. Right after the audio stream starts, it sends a hidden typed instruction ("the user just joined, greet them"). The model answers it out loud. The greeting follows the system prompt, so a persona changes how it greets:

```bash
uv run gallery/07_audio_with_assistant_speaks_first.py --system-prompt "You are Sunny, the receptionist at Kyoto Ramen. Greet callers, take reservations, keep answers short."
```

`--greeting-instruction` replaces the hidden instruction itself. It prints `[greeting] assistant started speaking 0.9s after connecting` (time to first audio).

## Usage and cost

`gallery/08_audio_with_usage_tracking.py` reads the `usageEvent` token counts Nova Sonic sends during a session. After each spoken reply, it prints that reply's cost and the running total:

```
[usage] reply 2: in 310 speech / 12 text, out 455 speech / 61 text = $0.0065 | session $0.0158
```

On exit, it prints a breakdown by token type, the total, and the cost per minute. Estimates use the us-east-1 on-demand prices from the AWS Price List ("Nova Sonic 2.0", retrieved 2026-11-23), stored in `PRICE_PER_1K_TOKENS` in the script. Prices change, so check the [Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/).

| Tokens | USD per 1K |
| --- | --- |
| Input speech (your mic) | 0.003 |
| Output speech (assistant voice) | 0.012 |
| Input text (prompt, history, typed text, tool results) | 0.00033 |
| Output text | 0.00275 |

## Voices

| Voice | Locale | Notes |
| --- | --- | --- |
| `matthew` | en-US | masculine, polyglot (speaks all supported languages) |
| `tiffany` | en-US | feminine, polyglot (speaks all supported languages) |
| `amy` | en-GB | feminine |
| `olivia` | en-AU | feminine |
| `kiara` | en-IN / hi-IN | feminine |
| `arjun` | en-IN / hi-IN | masculine |
| `ambre` / `florian` | fr-FR | feminine / masculine |
| `beatrice` / `lorenzo` | it-IT | feminine / masculine |
| `tina` / `lennart` | de-DE | feminine / masculine |
| `lupe` / `carlos` | es-US | feminine / masculine |
| `carolina` / `leo` | pt-BR | feminine / masculine |


## Audio format

- Input: 16 kHz, 16-bit, mono LPCM (microphone)
- Output: 24 kHz, 16-bit, mono LPCM (speakers)

## Reference docs

Offline copies of the Nova 2 Sonic documentation are in [docs/](docs/README.md).

## Troubleshooting

- **`No AWS credentials found for profile`**: the profile name is wrong, or (for SSO) run `aws sso login --profile <name>`.
- **`AccessDeniedException`**: model access isn't enabled for your account in that region, or the IAM policy lacks `bedrock:InvokeModelWithBidirectionalStream`.
- **Ctrl+C** shuts down cleanly, same as Enter.
- **No sound or wrong mic**: the default Windows input/output devices are used; change them in Windows Sound settings.
- **Model interrupts itself**: use headphones to avoid echo.
