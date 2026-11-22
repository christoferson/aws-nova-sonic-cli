# aws-nova-sonic-cli

Console speech-to-speech AI agent using [Amazon Nova 2 Sonic](https://aws.amazon.com/blogs/aws/introducing-amazon-nova-2-sonic-next-generation-speech-to-speech-model-for-conversational-ai/) on Amazon Bedrock. You speak into your microphone, the model answers out loud, and both transcripts are printed in the terminal.

Based on the AWS [console-python samples](https://github.com/aws-samples/amazon-ai-samples/tree/main/speech-to-speech/amazon-nova-2-sonic/sample-codes/console-python):

| Script | What it shows |
| --- | --- |
| `nova_sonic_simple.py` | Minimal speech-to-speech loop: mic in, voice out, transcripts printed |
| `nova_sonic.py` | Full version with **barge-in** (interrupt the assistant mid-answer), callback-based mic capture, auto-exit when the server ends the session, and `--debug` event logging |
| `nova_sonic_tool_use.py` | Barge-in version plus **tool use**: the model calls Python functions (date/time, weather, calculator) and speaks the results |
| `tools.py` | The tool implementations and their JSON schemas |
| `sonic_common.py` | Shared code: AWS profile credentials, CLI options, voices, Enter/Ctrl+C handling |

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
pip install -r requirements.txt
python nova_sonic_simple.py
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

```bash
uv run nova_sonic_simple.py   # simple
uv run nova_sonic.py          # with barge-in
uv run nova_sonic.py --debug  # with event-level logging
uv run nova_sonic_tool_use.py # with barge-in and tools
```

Then start talking. Press **Enter** (or Ctrl+C) to end the session.

With `nova_sonic.py` you can start speaking while the assistant is talking: it stops immediately, prints `(interrupted)`, and answers your new input. Use headphones so it doesn't interrupt itself.

Options (all scripts):

| Flag | Default | Description |
| --- | --- | --- |
| `--profile` | `$AWS_PROFILE` or `default` | AWS profile for credentials |
| `--region` | `$AWS_REGION` or `us-east-1` | Bedrock region |
| `--model-id` | `amazon.nova-2-sonic-v1:0` | Model ID |
| `--voice` | `tiffany` | Output voice (see below) |
| `--list-voices` | | Print available voices and exit |
| `--system-prompt` | built-in assistant prompt | System prompt text |
| `--debug` | off | `nova_sonic.py` / `nova_sonic_tool_use.py`: log events, stop reasons, token usage, tool results |

Example:

```bash
uv run nova_sonic.py --voice matthew --system-prompt "You are a friendly travel guide. Keep answers short."
```

## Tools

`nova_sonic_tool_use.py` gives the model three tools, defined in `tools.py`. Each call is printed as `[tool] name(args)` along with how long it took.

| Tool | What it does | Try saying |
| --- | --- | --- |
| `get_date_and_time` | Current date, time and weekday, locally or in any IANA timezone | "What time is it in New York?" |
| `get_weather` | Current weather and 3-day forecast for a city, from [Open-Meteo](https://open-meteo.com/) (free, no API key, needs internet) | "Do I need an umbrella in Osaka tomorrow?" |
| `calculate` | Exact arithmetic (`+ - * / // % **`, `sqrt`, `log`, trig, `pi`, `e`), parsed safely without `eval` | "What's 15 percent of 2,400?" |

Tools run in the background, so the model keeps talking ("let me check the weather...") while it waits. This is Nova 2 Sonic's asynchronous tool calling. Every tool call gets a result, even on error, because the model waits for one.

To add a tool, write an async function and add an entry to `TOOLS` in `tools.py` with a description and JSON schema.

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
