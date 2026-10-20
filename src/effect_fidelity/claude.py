"""Claude simulators: Anthropic API (ANTHROPIC_API_KEY) or Claude Code CLI (`claude -p`, uses your signed-in account).

Claude does not expose next-token probabilities, so we ask it to state the probability that this person picks each
option and return JSON. Those stated probabilities replace the local models' token probabilities; everything else
(personas, sampling, scoring) is identical.
"""
from __future__ import annotations
import json, re, shutil, subprocess, time
import numpy as np

DEFAULT_MODEL = "claude-sonnet-5-5"  # API backend default; the CLI backend uses your account's default model
CLI_DEFAULT_LABEL = "claude-default"  # label when the CLI response doesn't name its model
# USD per million tokens (input, output), list prices as of 2026-10. Used only for the printed estimate.
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-sonnet-5": (2.0, 10.0), "claude-sonnet-4-6": (3.0, 15.0),
          "claude-sonnet-4-5": (3.0, 15.0), "claude-opus-5-5": (4.0, 20.0), "claude-haiku-5-5": (0.10, 0.50)}
EST_OUTPUT_TOKENS = 40
EST_CLI_OVERHEAD_TOKENS = 0  # Claude Code adds its own system prompt; billed to your plan, not shown here

INSTR = ("\n\nInstead of a single answer, estimate how likely this person is to give each answer. Reply with ONLY a JSON "
         "object mapping each label to a probability (they must sum to 1), e.g. {example}. No other text.")


def build_prompt(system: str, user: str, labels: list[str]) -> str:
    ex = json.dumps({l: round(1 / len(labels), 2) for l in labels[:3]})
    return f"{system}\n\n{user}{INSTR.format(example=ex)}"


def parse_probs(text: str, labels: list[str]) -> np.ndarray:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in reply: {text[:200]!r}")
    d = json.loads(m.group(0))
    p = np.array([max(float(d.get(l, d.get(str(l), 0.0)) or 0.0), 0.0) for l in labels])
    if p.sum() <= 0:
        raise ValueError(f"no usable probabilities: {d}")
    return p / p.sum()


def est_tokens(prompt: str) -> int:
    return max(1, round(len(prompt) / 3.5))


def estimate_cost(prompts: list[str], model: str) -> tuple[int, int, float | None]:
    tin = sum(est_tokens(p) for p in prompts)
    tout = EST_OUTPUT_TOKENS * len(prompts)
    pr = PRICES.get(model)
    usd = None if pr is None else tin / 1e6 * pr[0] + tout / 1e6 * pr[1]
    return tin, tout, usd


class FatalClaudeError(SystemExit):
    """A non-transient failure (auth, bad model, usage limit): retrying won't help, so stop the whole run."""


def _retry(fn, tries=5, base=2.0):
    for i in range(tries):
        try:
            return fn()
        except FatalClaudeError:
            raise
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            wait = base * 2 ** i
            print(f"  retry {i + 1}/{tries - 1} in {wait:.0f}s: {str(e)[:120]}", flush=True)
            time.sleep(wait)


class ClaudeAPI:
    """Anthropic Messages API. Reads ANTHROPIC_API_KEY from the environment."""
    def __init__(self, model: str = DEFAULT_MODEL, client=None, retry_base: float = 2.0):
        self.model, self.retry_base = model, retry_base
        if client is None:
            import os, anthropic
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise SystemExit("ANTHROPIC_API_KEY is not set. Use --simulator claude-cli to use your signed-in Claude Code instead.")
            client = anthropic.Anthropic(max_retries=3)
        self.client = client

    def option_probs(self, system: str, user: str, labels: list[str]) -> np.ndarray:
        prompt = build_prompt(system, user, labels)
        def call():
            r = self.client.messages.create(model=self.model, max_tokens=200, messages=[{"role": "user", "content": prompt}])
            return parse_probs("".join(getattr(b, "text", "") for b in r.content), labels)
        return _retry(call, base=self.retry_base)


# (substring in error text, plain-English explanation + fix). Matched case-insensitively.
FATAL_PATTERNS = [
    (("not logged in", "please run /login", "login required", "authentication", "unauthorized", "401", "invalid api key",
      "oauth", "credential"),
     "Claude Code isn't signed in (or the sign-in expired). Fix: run `claude` in a terminal, sign in, type /exit, then rerun."),
    (("model", ),  # refined below: only with not-found/invalid/access wording
     "The Claude model isn't available to your account. Fix: rerun without --model (uses your account default), "
     "or pass a model your plan has, e.g. --model sonnet."),
    (("usage limit", "rate limit reached", "limit reached", "out of credits", "credit balance", "quota", "billing"),
     "Your Claude plan's usage limit is used up (or billing isn't set up). Fix: wait for the limit to reset "
     "(the message above usually says when), or switch accounts, then rerun; cached answers are kept."),
]
_MODEL_WORDS = ("not found", "invalid", "not available", "not exist", "no access", "have access", "issue with the selected model", "not supported", "unknown", "404")


def classify_error(text: str) -> str | None:
    """Return a plain-English fix if the error is clearly non-transient, else None (retry)."""
    t = (text or "").lower()
    if any(w in t for w in ("overloaded", "529", "timeout", "timed out", "econnreset", "network", "503", "502", "500")):
        return None
    for keys, msg in FATAL_PATTERNS:
        if keys == ("model",):
            if "model" in t and any(w in t for w in _MODEL_WORDS):
                return msg
        elif any(k in t for k in keys):
            return msg
    return None


def describe_cli_failure(returncode: int, stdout: str, stderr: str) -> str:
    """Full (untruncated) description of a failed/errored `claude -p --output-format json` call."""
    parts = [f"claude exited {returncode}"]
    try:
        out = json.loads(stdout)
    except Exception:  # noqa: BLE001
        out = None
    if isinstance(out, dict):
        for k in ("subtype", "is_error", "result", "error", "errors", "api_error_status", "terminal_reason"):
            if out.get(k) not in (None, "", []):
                parts.append(f"{k}: {out[k]}")
    elif stdout.strip():
        parts.append(f"stdout: {stdout.strip()}")
    if stderr and stderr.strip():
        parts.append(f"stderr: {stderr.strip()}")
    return "\n  ".join(parts)


def model_from_response(out: dict) -> str | None:
    mu = out.get("modelUsage") or {}
    if isinstance(mu, dict) and mu:
        # several keys can appear (e.g. a small helper model); take the one with the most output tokens
        return max(mu, key=lambda k: (mu[k] or {}).get("outputTokens", 0) if isinstance(mu[k], dict) else 0)
    return out.get("model") or None


class ClaudeCLI:
    """Claude Code CLI: `claude -p <prompt> --output-format json --max-turns 1 --tools "" [--model M]`.

    No API key; uses your login. Without --model, Claude Code uses your account's default model; the model it
    actually used is read from the JSON response (`modelUsage`) into `self.model_used`.
    """
    def __init__(self, model: str | None = None, bin: str = "claude", timeout: int = 180, retry_base: float = 2.0, run=subprocess.run):
        self.model, self.bin, self.timeout, self.retry_base, self._run = model, bin, timeout, retry_base, run
        self.model_used: str | None = None

    def check(self):
        if shutil.which(self.bin) is None:
            raise SystemExit("`claude` not found. Install Claude Code: npm install -g @anthropic-ai/claude-code, then run `claude` once to sign in.")

    def cmd(self, prompt: str) -> list[str]:
        # Documented flags (code.claude.com/docs/en/cli-reference): --max-turns (print mode), --tools "" disables
        # all built-in tools, --no-session-persistence keeps thousands of calls out of your session history.
        return ([self.bin, "-p", prompt, "--output-format", "json", "--max-turns", "1", "--tools", "",
                 "--no-session-persistence"] + (["--model", self.model] if self.model else []))

    def raw(self, prompt: str) -> str:
        """One call; returns the reply text. Raises FatalClaudeError for non-transient failures."""
        r = self._run(self.cmd(prompt), capture_output=True, text=True, timeout=self.timeout)
        try:
            out = json.loads(r.stdout)
        except Exception:  # noqa: BLE001
            out = None
        if r.returncode != 0 or not isinstance(out, dict) or out.get("is_error"):
            desc = describe_cli_failure(r.returncode, r.stdout or "", r.stderr or "")
            fix = classify_error(desc)
            if fix:
                raise FatalClaudeError(f"\nClaude Code call failed and retrying won't help.\n  {desc}\n\n{fix}\n")
            raise RuntimeError(desc)
        m = model_from_response(out)
        if m:
            self.model_used = m
        return out.get("result", "") or ""

    def preflight(self) -> str:
        """One tiny call. Returns the model label; raises SystemExit with a clear message if it fails."""
        try:
            _retry(lambda: self.raw("Reply with OK"), tries=2, base=self.retry_base)
        except FatalClaudeError:
            raise
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"\nClaude Code test call ('Reply with OK') failed:\n  {e}\n\n"
                             "Fix: run `claude -p \"Reply with OK\" --output-format json` yourself to see the error; "
                             "if it asks you to sign in, run `claude`, sign in, /exit, and rerun.\n")
        return self.label()

    def label(self) -> str:
        return self.model or self.model_used or CLI_DEFAULT_LABEL

    def option_probs(self, system: str, user: str, labels: list[str]) -> np.ndarray:
        prompt = build_prompt(system, user, labels)
        return _retry(lambda: parse_probs(self.raw(prompt), labels), base=self.retry_base)
