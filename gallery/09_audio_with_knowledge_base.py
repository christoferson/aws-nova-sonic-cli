"""Step 9: knowledge-base answers (RAG). The assistant searches local documents and answers from them.

Run:  uv run gallery/09_audio_with_knowledge_base.py
Try:  "How long can a Nova Sonic connection stay open?", "Which voices speak Hindi?",
      "What does endpointing sensitivity do?"
Other documents:
      uv run gallery/09_audio_with_knowledge_base.py --kb-dir path/to/notes --kb-topic "my team's runbooks"

Retrieval-augmented generation (RAG)
------------------------------------
The model only knows what it learned in training. To answer from *your* documents, give it a
search tool and tell it to use it (this is tool use from step 03 with a single tool):

    1. At startup, split every .md/.txt file in --kb-dir into passages of a few paragraphs and index
       them. The default is docs/nova-2-sonic, the Nova Sonic docs saved in this repo.
    2. The model calls search_knowledge_base(query) when a question is about the topic.
    3. We return the 3 best passages with their file and section.
    4. The model answers from those passages, out loud.

Where the index lives
---------------------
Not in the model's context: the documents are never put in the prompt as a whole. The index is a
plain Python object in this process's memory, rebuilt at every start (milliseconds for a few
hundred passages) and never written to disk. For each passage it stores word counts and length,
plus how many passages contain each word. The model only ever sees the 3 passages a search
returns (each cut to RESULT_MAX_CHARS), as the tool result.

Why not paste all the docs into the system prompt instead? Every reply would pay input tokens for
all of it, it stops fitting as the documents grow, and the model answers more reliably from a
few relevant passages than from everything at once.

    startup:  files -> split_into_passages -> tokenize -> KnowledgeBase (word statistics)
    question: model calls search_knowledge_base("keywords") -> KnowledgeBase.search (BM25 score
              every passage) -> top 3 passages -> tool result -> model answers from them

The search is BM25, the classic keyword-ranking formula behind most search engines: a passage
scores higher when it contains the query's words, especially rare words, and is not too long.
It needs no extra services or packages. It matches words, not meaning, so the tool description
asks the model for short English keyword queries.

Going further: for large or frequently changing document sets, replace KnowledgeBase.search with
Amazon Bedrock Knowledge Bases (the bedrock-agent-runtime Retrieve API via boto3), which adds
semantic (embedding) search, managed ingestion and access control.

Builds on step 03 (tool use) by overriding tool_configuration() and execute_tool().
"""
import asyncio
import collections
import importlib
import json
import math
import pathlib
import re

import boto3

from sonic.common import default_system_prompt, print_voices

# Builds on steps 2 and 3; importlib because module names can't start with a digit in an import statement.
barge_in = importlib.import_module("02_audio_with_barge_in")
tool_use = importlib.import_module("03_audio_with_tools")
build_parser = barge_in.build_parser
debug_print = barge_in.debug_print
run_conversation = barge_in.run_conversation
ToolUseStreamManager = tool_use.ToolUseStreamManager

DEFAULT_KB_DIR = pathlib.Path(__file__).resolve().parent.parent / "docs" / "nova-2-sonic"
DEFAULT_KB_TOPIC = "Amazon Nova 2 Sonic (the speech-to-speech model this app uses)"
# Shown to the user at startup, so they know what to ask.
DEFAULT_KB_SUMMARY = (
    "I can answer questions about Amazon Nova 2 Sonic: how it works, voices and languages, "
    "interrupting, turn-taking, tool use, chat history, and integrations."
)
TOOL_NAME = "search_knowledge_base"

PASSAGE_MAX_CHARS = 900  # paragraphs are grouped into passages up to about this size
RESULT_MAX_CHARS = 800  # each returned passage is cut to this; tool results cost input tokens
TOP_K = 3

# BM25 parameters (standard defaults): K1 limits how much repeating a word keeps adding,
# B controls how strongly long passages are penalised.
BM25_K1 = 1.5
BM25_B = 0.75

STOPWORDS = set(
    "a an and are as at be by can do does for from has have how i in is it its of on or that the "
    "their this to was what when where which who why will with you your".split()
)


def tokenize(text):
    """Lowercase words without stopwords, with a crude plural strip so 'voices' matches 'voice'."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in STOPWORDS]


def split_into_passages(text, max_chars=PASSAGE_MAX_CHARS):
    """Split markdown into (section heading, passage) pairs of whole paragraphs."""
    passages, section, paragraphs = [], "", []

    def flush():
        if paragraphs:
            passages.append((section, "\n\n".join(paragraphs)))
            paragraphs.clear()

    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block or block.startswith("Source: "):
            continue
        if block.startswith("#"):
            # A heading starts a new section; the passage text keeps the heading out.
            flush()
            lines = block.splitlines()
            section = lines[0].lstrip("#").strip()
            block = "\n".join(lines[1:]).strip()
            if not block:
                continue
        if paragraphs and sum(len(p) for p in paragraphs) + len(block) > max_chars:
            flush()
        paragraphs.append(block)
    flush()
    return passages


class KnowledgeBase:
    """In-memory BM25 index over the passages of every .md/.txt file in a folder."""

    def __init__(self, folder):
        self.folder = pathlib.Path(folder)
        self.passages = []
        for path in sorted(self.folder.rglob("*")):
            if path.suffix.lower() not in (".md", ".txt") or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            url = re.match(r"Source: (\S+)", text)
            for section, passage in split_into_passages(text):
                tokens = tokenize(f"{section} {passage}")
                if tokens:
                    self.passages.append({
                        "source": path.relative_to(self.folder).as_posix(),
                        "url": url.group(1) if url else None,
                        "section": section,
                        "text": passage,
                        "term_counts": collections.Counter(tokens),
                        "length": len(tokens),
                    })
        # How many passages contain each word; rare words are worth more.
        self.passages_with_term = collections.Counter()
        for passage in self.passages:
            self.passages_with_term.update(passage["term_counts"].keys())
        self.average_length = sum(p["length"] for p in self.passages) / max(len(self.passages), 1)

    def search(self, query, k=TOP_K):
        """The k best passages for a query, best first, using BM25 scoring."""
        n = len(self.passages)
        scored = []
        for passage in self.passages:
            score = 0.0
            for term in set(tokenize(query)):
                tf = passage["term_counts"].get(term, 0)
                if not tf:
                    continue
                df = self.passages_with_term[term]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))  # rarity of the word
                length_norm = 1 - BM25_B + BM25_B * passage["length"] / self.average_length
                score += idf * tf * (BM25_K1 + 1) / (tf + BM25_K1 * length_norm)
            if score > 0:
                scored.append((score, passage))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [passage for _score, passage in scored[:k]]


class KnowledgeBaseStreamManager(ToolUseStreamManager):
    """Offers a single tool, search_knowledge_base, backed by a local KnowledgeBase."""

    def __init__(self, *args, knowledge_base, topic=DEFAULT_KB_TOPIC, **kwargs):
        super().__init__(*args, **kwargs)
        self.knowledge_base = knowledge_base
        self.topic = topic
        if not kwargs.get("system_prompt"):
            # Replace step 03's hint (date/weather/calculator) with knowledge-base instructions.
            self.system_prompt = default_system_prompt(self.voice_id) + (
                f" You can search a knowledge base about {topic} with the {TOOL_NAME} tool. "
                f"For any question about {topic}, search it first and answer from the results. "
                "If the results don't contain the answer, say you couldn't find it. "
                "Summarize in your own words; don't read out URLs, code, or markdown."
            )

    def tool_configuration(self):
        schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A few English keywords, e.g. 'connection time limit' or 'Hindi voices'",
                }
            },
            "required": ["query"],
        }
        return {"tools": [{
            "toolSpec": {
                "name": TOOL_NAME,
                "description": (
                    f"Search the knowledge base about {self.topic}. Returns the most relevant passages "
                    "with their source. Use it for any question about this topic."
                ),
                # As in sonic/tools.py: inputSchema.json is a JSON string, not an object.
                "inputSchema": {"json": json.dumps(schema)},
            }
        }]}

    async def execute_tool(self, name, arguments):
        if name != TOOL_NAME:
            return {"error": f"Unknown tool '{name}'"}
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return {"error": "query must be a non-empty string"}
        passages = self.knowledge_base.search(query)
        if not passages:
            print(f"[kb] no passages match '{query}'")
            return {"query": query, "results": [], "note": "No matching passages found."}
        print(f"[kb] '{query}' -> " + ", ".join(f"{p['source']} ({p['section']})" for p in passages))
        return {
            "query": query,
            "results": [
                {
                    "source": p["source"],
                    "section": p["section"],
                    "text": p["text"][:RESULT_MAX_CHARS],
                    **({"url": p["url"]} if p["url"] else {}),
                }
                for p in passages
            ],
        }


def print_knowledge_base(kb_dir, topic):
    """Tell the user what the assistant can answer questions about."""
    if pathlib.Path(kb_dir).resolve() == DEFAULT_KB_DIR:
        print(DEFAULT_KB_SUMMARY)
    else:
        print(f"I can answer questions about {topic}.")


async def main():
    parser = build_parser("Amazon Nova 2 Sonic console client that answers from local documents")
    parser.add_argument(
        "--kb-dir",
        default=str(DEFAULT_KB_DIR),
        metavar="DIR",
        help="Folder of .md/.txt files to search (default: the saved Nova 2 Sonic docs)",
    )
    parser.add_argument(
        "--kb-topic",
        default=DEFAULT_KB_TOPIC,
        metavar="TEXT",
        help="What the documents are about; tells the model when to search",
    )
    args = parser.parse_args()
    barge_in.DEBUG = args.debug

    if args.list_voices:
        print_voices()
        return

    knowledge_base = KnowledgeBase(args.kb_dir)
    if not knowledge_base.passages:
        parser.error(f"no .md or .txt content found in {args.kb_dir}")
    print_knowledge_base(args.kb_dir, args.kb_topic)

    session = boto3.Session(profile_name=args.profile)
    print(f"Using AWS profile '{session.profile_name}' in region {args.region}, voice '{args.voice}'")

    await run_conversation(KnowledgeBaseStreamManager(
        session=session,
        model_id=args.model_id,
        region=args.region,
        voice_id=args.voice,
        system_prompt=args.system_prompt,
        knowledge_base=knowledge_base,
        topic=args.kb_topic,
    ))


if __name__ == "__main__":
    asyncio.run(main())
