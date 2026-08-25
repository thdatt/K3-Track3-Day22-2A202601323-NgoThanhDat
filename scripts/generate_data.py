from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

import typer
try:
    from dotenv import load_dotenv
except ImportError:  # optional for unit-test/import-only environments
    def load_dotenv() -> bool:
        return False

from rich import print

load_dotenv()

app = typer.Typer(help="Synthetic Data Generation for Preference Alignment")

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o"
DEFAULT_OPENAI_MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are an AI data engineer specializing in preference alignment (DPO/ORPO).
Your task is to generate high-quality preference pairs in JSONL format.
Each pair must have:
1. 'prompt': A clear instruction or question.
2. 'chosen': A high-quality, accurate, and helpful response.
3. 'rejected': A plausible but lower-quality response (e.g., contains a subtle error, hallucination, or poor formatting).
4. 'metadata': A dictionary with 'domain' and 'rubric'.

Output ONLY the JSONL lines, one per line. Do not include markdown formatting or extra text."""

USER_PROMPT_TEMPLATE = """Generate {count} new preference pairs about {domain}.
Use the following examples as a style guide:
{examples}

Focus on: {focus}"""


def resolve_client_kwargs(env: Mapping[str, str | None]) -> tuple[str, str | None]:
    """Resolve ``(api_key, base_url)`` for the OpenAI client.

    Prefer OpenRouter credentials when present; else fall back to a plain
    OpenAI (or OpenAI-compatible) endpoint. This avoids pairing an OpenAI
    key with an OpenRouter base URL, which authenticates nothing.
    """
    openrouter_key = env.get("OPENROUTER_API_KEY")
    if openrouter_key:
        return (
            openrouter_key,
            env.get("OPENROUTER_BASE_URL") or OPENROUTER_DEFAULT_BASE_URL,
        )
    api_key = env.get("OPENAI_API_KEY")
    if not api_key:
        return ("", None)
    return (
        api_key,
        env.get("OPENAI_BASE_URL"),
    )


def resolve_model(env: Mapping[str, str | None], requested: str | None) -> tuple[str, str]:
    """Return ``(model, provider_label)`` for the given environment."""
    if env.get("OPENROUTER_API_KEY"):
        return (
            env.get("OPENROUTER_MODEL") or requested or DEFAULT_OPENROUTER_MODEL,
            "openrouter",
        )
    return (requested or DEFAULT_OPENAI_MODEL, "openai")


@app.command()
def generate(
    count: int = 5,
    domain: str = "machine learning",
    focus: str = "technical accuracy and safety",
    output_file: Path = Path("data/synthetic_preferences.jsonl"),
    seed_file: Path = Path("data/sample_preferences.jsonl"),
    model: str | None = None,
) -> None:
    """Generate synthetic preference pairs via an OpenAI-compatible API."""
    env: dict[str, str | None] = dict(os.environ)
    api_key, base_url = resolve_client_kwargs(env)
    resolved_model, provider = resolve_model(env, model)

    if not api_key:
        print(
            "[red]Error: neither OPENROUTER_API_KEY nor OPENAI_API_KEY "
            "environment variable is set.[/red]"
        )
        raise typer.Exit(1)

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Load some examples from seed file
    examples_str = ""
    if seed_file.exists():
        with seed_file.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()][:3]
            examples_str = "\n".join(lines)

    print(f"Generating [blue]{count}[/blue] pairs for domain: [green]{domain}[/green]...")
    print(f"[dim]provider={provider} model={resolved_model}[/dim]")

    response = client.chat.completions.create(
        model=resolved_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    count=count, domain=domain, examples=examples_str, focus=focus
                ),
            },
        ],
        temperature=0.7,
    )

    content = response.choices[0].message.content
    if not content:
        print("[red]Error: Received empty response from API.[/red]")
        raise typer.Exit(1)

    # Simple validation and write
    valid_lines: list[str] = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip markdown code blocks if the model included them
        if line.startswith("```"):
            continue
        try:
            json.loads(line)
            valid_lines.append(line)
        except json.JSONDecodeError:
            print(f"[yellow]Skipping invalid JSON line: {line[:50]}...[/yellow]")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as f:
        for line in valid_lines:
            f.write(line + "\n")

    print(f"[green]Successfully added {len(valid_lines)} pairs to {output_file}[/green]")


if __name__ == "__main__":
    app()
