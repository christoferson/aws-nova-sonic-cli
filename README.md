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
  09_audio_with_knowledge_base.py          Builds on 03: answers from local documents via a search tool (RAG)
  10_audio_with_forced_tool_use.py         Builds on 09: choose whether the model may or must call a tool first
  11_audio_with_phone_quality.py           Builds on 02: 8 kHz audio both ways, like a phone call
  12_audio_with_multi_step_tools.py        Builds on 03: a returns agent that chains lookup, check and action tools
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
- An AWS account with access to the `amazon.nova-2-5-sonic` model (Nova 2.5 Sonic) in Amazon Bedrock, in a region where it's available (default `us-east-1`).
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
uv run gallery/09_audio_with_knowledge_base.py
uv run gallery/10_audio_with_forced_tool_use.py --tool-choice auto
uv run gallery/11_audio_with_phone_quality.py
uv run gallery/12_audio_with_multi_step_tools.py
```

Then start talking. Press **Enter** (or Ctrl+C) to end the session.

From step 02 on, and in `nova_sonic.py`, you can start speaking while the assistant is talking. It stops immediately, prints `(interrupted)`, and answers your new input. Use headphones so it doesn't interrupt itself.

Options (all scripts):

| Flag | Default | Description |
| --- | --- | --- |
| `--profile` | `$AWS_PROFILE` or `default` | AWS profile for credentials |
| `--region` | `$AWS_REGION` or `us-east-1` | Bedrock region |
| `--model-id` | `amazon.nova-2-5-sonic` | Model ID. Use `amazon.nova-2-sonic-v1:0` to switch back to Nova 2 Sonic |
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

## Knowledge base (RAG)

`gallery/09_audio_with_knowledge_base.py` gives the model one tool, `search_knowledge_base`. At startup it indexes every `.md`/`.txt` file in a folder, by default the saved Nova 2 Sonic docs in `docs/nova-2-sonic`, so you can ask the assistant about Nova Sonic itself:

- "How long can a Nova Sonic connection stay open?"
- "What does endpointing sensitivity do?"
- "How should a client handle barge-in?"

Each search prints the passages it found, for example `[kb] 'connection time limit' -> sonic-getting-started.md (Getting started ...)`. The model answers from them and says so when they don't contain the answer.

Point it at your own documents:

```bash
uv run gallery/09_audio_with_knowledge_base.py --kb-dir path/to/notes --kb-topic "my team's runbooks"
```

How the index works:
- **At startup:** the files are split into passages of a few paragraphs, and word statistics are kept in memory. Nothing is written to disk, and the index is rebuilt on every run.
- **When you ask:** the model calls the search tool with keywords. Every passage is scored and the best 3 are returned as the tool result.
- **What the model sees:** only those 3 passages. The documents are never pasted into its prompt as a whole, which would cost tokens on every reply and stop fitting as they grow.

The search is BM25 keyword ranking, implemented in the script with no extra services. It matches words rather than meaning. For semantic search over large document sets, swap `KnowledgeBase.search` for Amazon Bedrock Knowledge Bases (the `bedrock-agent-runtime` Retrieve API).

## Forced tool use

`gallery/10_audio_with_forced_tool_use.py` adds the date/time tool to step 09's docs search and sets `toolChoice`, which decides whether the model may or must call a tool before answering:

| `--tool-choice` | Behavior |
| --- | --- |
| `search_knowledge_base` (default) | Always searches the docs first, so answers can't come from memory |
| `get_date_and_time` | Always checks the time first |
| `any` | Always calls one of the tools first; the model picks which |
| `auto` | The model decides, as in steps 03 and 09 |

Forcing the search makes the model *look* in the docs, but it can still add general knowledge or answer off-topic questions. Add `--strict` to allow answers only from the documentation. The assistant then says when something isn't covered and declines unrelated questions; the date and time still work. Test it with something off-topic ("who won the World Cup?") and something the docs don't cover.

After each reply it prints which tools were used, for example `[choice] reply 3 (auto): no tool`. Ask the same questions in different modes to compare them. Forcing a tool makes every reply wait for it, even "hello" (which triggers a search for "hello").

## Phone-quality audio

`gallery/11_audio_with_phone_quality.py` runs both your microphone and the assistant's voice at 8 kHz, the sample rate phone networks use. You hear what a caller would hear (muffled, since only frequencies up to 4 kHz survive), and the model hears you the way it would over a phone line. Nova 2 Sonic's speech recognition was improved specifically for 8 kHz telephony audio. Compare with `--rate 16000` (or `24000`).

A real phone agent gets its audio from a provider such as Amazon Connect, Twilio or Vonage, usually as 8 kHz mu-law that is converted to 16-bit PCM first. This step skips the phone network and just records and plays at 8 kHz.

## Multi-step tool workflow

`gallery/12_audio_with_multi_step_tools.py` is a returns assistant for a made-up store, Nova Outfitters. A return needs three tools in order: `lookup_order`, then `check_return_eligibility`, then `create_return`. The model chains them across the conversation and asks you to confirm before it creates the return. This is the kind of agentic workflow Nova 2.5 Sonic's improved reasoning and tool calling target.

| Order | Try saying | What happens |
| --- | --- | --- |
| 1001 | "I'd like to return the socks from order 1001." | Eligible: it states the refund, asks you to confirm, then creates the return |
| 1002 | "Can I return my rain jacket? Order 1002." | Declined: past the 30-day window |
| 1003 | "Return the lantern from order 1003." | Declined: final-sale item (the water bottle in the same order is returnable) |
| 1004 | "Where is order 1004?" | Shipped but not delivered, so not returnable yet |

Each step is printed as `[tool] name(args)`. The data is in memory, so returns you create last until the program exits. `create_return` checks eligibility again itself, so it can't create a return that shouldn't exist even if the model skips a step.

Compare with Nova 2 Sonic using `--model-id amazon.nova-2-sonic-v1:0`. In testing, both completed the workflow, but 2.0 gave longer answers and once said it would process the return without calling `create_return`.

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

## References

- [aws-nova-sonic-cli](https://github.com/christoferson/aws-nova-sonic-cli): this project, written against the raw bidirectional event protocol
- [aws-nova-sonic-cli-strands](https://github.com/christoferson/aws-nova-sonic-cli-strands): the same assistant built with the Strands Agents `BidiAgent`
- [Nova 2.5 Sonic model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-2-5-sonic.html): model ID, regions, capabilities and quotas
- Offline copies of the Nova 2 Sonic documentation are in [docs/](docs/README.md).

## Troubleshooting

- **`No AWS credentials found for profile`**: the profile name is wrong, or (for SSO) run `aws sso login --profile <name>`.
- **`AccessDeniedException`**: model access isn't enabled for your account in that region, or the IAM policy lacks `bedrock:InvokeModelWithBidirectionalStream`.
- **Ctrl+C** shuts down cleanly, same as Enter.
- **No sound or wrong mic**: the default Windows input/output devices are used; change them in Windows Sound settings.
- **Model interrupts itself**: use headphones to avoid echo.
