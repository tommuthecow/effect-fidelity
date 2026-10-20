"""Run LLM persona simulators on both arms of every experiment, with an on-disk response cache."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .experiments import Experiment, Item

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"

SYSTEM = ("You are simulating one specific real person taking a survey. Answer exactly as this person would, "
          "not as an AI assistant.")


@dataclass
class SimulatorConfig:
    name: str
    model_file: str
    template: str
    conditioning: str  # "blind" or "persona"
    label: str


SIMULATORS = [
    SimulatorConfig("qwen15_blind", "qwen2.5-1.5b-instruct-q4_k_m.gguf", "chatml", "blind", "Qwen2.5-1.5B, blind prompt"),
    SimulatorConfig("qwen15_persona", "qwen2.5-1.5b-instruct-q4_k_m.gguf", "chatml", "persona", "Qwen2.5-1.5B, real-respondent personas"),
    SimulatorConfig("llama32_1b_persona", "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "llama3", "persona", "Llama-3.2-1B, real-respondent personas"),
    # Claude results (written by `python -m effect_fidelity.run_claude`); included in the report when present.
    SimulatorConfig("claude_blind", "claude", "n/a", "blind", "Claude, blind prompt"),
    SimulatorConfig("claude_persona", "claude", "n/a", "persona", "Claude, real-respondent personas"),
]


class Cache:
    def __init__(self, name: str):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = CACHE_DIR / f"{name}.jsonl"
        self.d = {}
        if self.path.exists():
            for line in self.path.open():
                r = json.loads(line)
                self.d[r["k"]] = r["p"]
        self.f = self.path.open("a")
        self.hits = self.misses = 0

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha1("\x1f".join(map(str, parts)).encode()).hexdigest()

    def get(self, k):
        v = self.d.get(k)
        if v is not None:
            self.hits += 1
        return v

    def put(self, k, p):
        self.misses += 1
        self.d[k] = p
        self.f.write(json.dumps({"k": k, "p": p}) + "\n")
        self.f.flush()


def render(item: Item, arm: str, persona: str, pre_answer: str | None = None) -> tuple[str, list[str], dict]:
    opts = item.arm_options.get(arm, item.options) if hasattr(item, "arm_options") else item.options
    vals = item.arm_values[arm] if hasattr(item, "arm_values") else item.values
    body = item.text(arm)
    if pre_answer is not None:
        body = f"Earlier in this survey you were asked: \"{item.pre.text(arm)}\" You answered: {pre_answer}.\n\nNext question: {body}"
    if opts:
        body += "\n\n" + "\n".join(f"{l}) {o}" for l, o in zip(item.labels, opts))
        instr = f"Answer with only the letter of your choice ({', '.join(item.labels)})."
    else:
        body += f"\n\nScale: {item.labels[0]} to {item.labels[-1]} ({item.scale_note})."
        instr = "Answer with only the number."
    return f"{body}\n\nThe person answering: {persona}\n\n{instr}", item.labels, vals


class Runner:
    def __init__(self, cfg: SimulatorConfig, n_per_arm: int = 150, seed: int = 0, lm=None):
        self.cfg, self.n, self.seed = cfg, n_per_arm, seed
        self.cache = Cache(cfg.model_file.replace(".gguf", ""))
        self._lm = lm
        self.calls = 0

    @property
    def lm(self):
        if self._lm is None:
            from .llm import LocalLM
            self._lm = LocalLM(str(ROOT / "models" / self.cfg.model_file), self.cfg.template)
        return self._lm

    def probs(self, system: str, user: str, labels: list[str]) -> np.ndarray:
        k = Cache.key(self.cfg.model_file, system, user, labels)
        p = self.cache.get(k)
        if p is None:
            p = self.lm.option_probs(system, user, labels).tolist()
            self.cache.put(k, p)
            self.calls += 1
        return np.asarray(p)

    def run(self, exp: Experiment) -> dict:
        """Simulate n respondents per arm. Loops item-outer / persona-inner so the stimulus prefix stays cached."""
        out = {}
        for arm in ("A", "B"):
            rng = np.random.default_rng(int(hashlib.md5(f"{self.seed}|{exp.id}|{arm}".encode()).hexdigest()[:8], 16))
            if self.cfg.conditioning == "persona":
                idx = rng.choice(len(exp.persona_pool), size=self.n, replace=True)
                personas = [exp.persona_text(exp.persona_pool.iloc[i]) for i in idx]
            else:
                personas = [exp.blind_text] * self.n
            scores = np.zeros((self.n, len(exp.items)))
            expected = np.zeros((self.n, len(exp.items)))
            for k, item in enumerate(exp.items):
                pre_ans = [None] * self.n
                if item.pre is not None and arm in item.pre_arms:
                    for i, per in enumerate(personas):
                        ptxt, plabs, _ = render(item.pre, arm, per)
                        pre_ans[i] = item.pre.options[rng.choice(len(plabs), p=self.probs(SYSTEM, ptxt, plabs))]
                order = sorted(range(self.n), key=lambda i: str(pre_ans[i]))
                for i in order:
                    txt, labs, vals = render(item, arm, personas[i], pre_ans[i])
                    p = self.probs(SYSTEM, txt, labs)
                    scores[i, k] = vals[labs[rng.choice(len(labs), p=p)]]
                    expected[i, k] = float(sum(p[j] * vals[l] for j, l in enumerate(labs)))
            out[arm] = {"draws": scores.mean(1).tolist(), "expected": expected.mean(1).tolist()}
        return out

