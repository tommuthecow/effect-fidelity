"""Accuracy per dollar, part 1: a cheap LoRA fine-tune of Qwen2.5-1.5B-Instruct on SocSci210 (Kolluri, Wu, Park & Bernstein 2025).

Training data is SocSci210 (individual responses from 210 social-science studies), with every study that could overlap our
17 test experiments removed by keyword screen (see EXCLUDE_KW), so all 17 experiments are unseen at training time.
We do NOT train on our 17 experiments at all, so no leave-one-out loop is needed.

CPU-only box (8 cores, 15 GB RAM, no GPU), so this is deliberately tiny: N_TRAIN examples, 1 epoch, LoRA r=8 on q/v,
loss on the single answer token. The full-scale version of this is Socrates-Qwen-14B, which we could not run here.

Stages (each checkpoints to disk; rerunning resumes):
  python -m effect_fidelity.finetune data    # -> data/ft/train.jsonl
  python -m effect_fidelity.finetune train   # -> models/qwen15_socsci_lora/
  python -m effect_fidelity.finetune eval    # -> results/sims_ft/{qwen15hf_raw,qwen15hf_ft}_persona.json
"""
from __future__ import annotations
import argparse, hashlib, json, time
import numpy as np
from .simulate import ROOT, SYSTEM, render
from .experiments import load_all

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
FT_DIR = ROOT / "data" / "ft"
ADAPTER = ROOT / "models" / "qwen15_socsci_lora"
OUT = ROOT / "results" / "sims_ft"
TIMING = ROOT / "results" / "ft_timing.json"
EXCLUDE_KW = ["600 people", "disease", "sunk", "North Korea", "Muslim", "Islam", "Bin Laden", "flag", "dollar bill", "television",
              "TV", "speeches", "too little", "spending", "email", "e-mail", "retailer", "Washington", "system justification",
              "welfare", "space exploration", "big cities", "environment", "crime", "law enforcement"]
N_TRAIN, MAX_TOK, N_EVAL, MAX_ITEMS = 480, 320, 60, 2


def _log_time(key, secs, extra=None):
    d = json.loads(TIMING.read_text()) if TIMING.exists() else {}
    d[key] = {"seconds": round(secs, 1), **(extra or {})}
    TIMING.write_text(json.dumps(d, indent=1))


def is_excluded(text: str) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in EXCLUDE_KW)


def build_data(seed=0):
    """Streams the 1.4 GB parquet in batches (the box has 15 GB RAM)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(ROOT / "data" / "raw" / "socsci210.parquet")
    bad, rows_meta = set(), []
    off = 0
    for bt in pf.iter_batches(columns=["study_id", "stimuli", "response"], batch_size=50000):
        d = bt.to_pandas()
        bad |= set(d.study_id[d.stimuli.map(is_excluded)])
        ok = d.response.between(0, 9).to_numpy()
        rows_meta += [(off + i, sid) for i, sid in zip(np.flatnonzero(ok), d.study_id.to_numpy()[ok])]
        off += len(d)
    rng = np.random.default_rng(seed)
    by = {}
    for i, sid in rows_meta:
        if sid not in bad:
            by.setdefault(sid, []).append(i)
    per = N_TRAIN // len(by) + 1
    pick = set()
    for sid, ix in by.items():
        pick |= set(rng.choice(ix, min(per, len(ix)), replace=False).tolist())
    pick = set(rng.choice(sorted(pick), min(N_TRAIN, len(pick)), replace=False).tolist())
    FT_DIR.mkdir(parents=True, exist_ok=True)
    out, off = [], 0
    for bt in pf.iter_batches(columns=["study_id", "prompt", "response"], batch_size=50000):
        d = bt.to_pandas()
        for j in [k for k in range(len(d)) if off + k in pick]:
            out.append({"prompt": d.prompt.iat[j], "answer": str(int(d.response.iat[j])), "study": d.study_id.iat[j]})
        off += len(d)
    rng.shuffle(out)
    with (FT_DIR / "train.jsonl").open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    (FT_DIR / "excluded_studies.json").write_text(json.dumps(sorted(bad)))
    print(f"{len(out)} examples from {len({r['study'] for r in out})} studies; excluded {len(bad)} possibly-overlapping studies")


def _tok_model(adapter=False, dtype=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(BASE)
    m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dtype or torch.bfloat16)
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, str(ADAPTER))
    return tok, m


def encode(tok, system, user, max_tok=None):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    ids = list(ids)
    if max_tok and len(ids) > max_tok:  # keep the head (persona) and the tail (question + instructions)
        ids = ids[:96] + ids[-(max_tok - 96):]
    return ids


def train(lr=1e-4, accum=8):
    """fp32 weights (a first bf16 attempt diverged to NaN after ~400 examples), grad-norm clip 1.0, non-finite losses skipped."""
    import torch
    from peft import LoraConfig, get_peft_model
    tok, m = _tok_model(dtype=torch.float32)
    ck = ROOT / "models" / "qwen15_socsci_lora_ckpt"
    start, prev = 0, 0.0
    if (ck / "state.json").exists():  # resume from the last checkpoint (optimizer state restarts; documented)
        from peft import PeftModel
        st = json.loads((ck / "state.json").read_text()); start, prev = st["done"], st["seconds"]
        m = PeftModel.from_pretrained(m, str(ck), is_trainable=True)
        print(f"resuming at example {start}", flush=True)
    else:
        m = get_peft_model(m, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    m.train()
    rows = [json.loads(l) for l in (FT_DIR / "train.jsonl").open()]
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
    t0, losses = time.time(), []
    for i, r in enumerate(rows):
        if i < start:
            continue
        ids = encode(tok, SYSTEM, r["prompt"], MAX_TOK)
        ans = tok.encode(r["answer"], add_special_tokens=False)[0]
        x = torch.tensor([ids])
        logits = m(input_ids=x).logits[0, -1].float()
        loss = torch.nn.functional.cross_entropy(logits[None], torch.tensor([ans])) / accum
        if not torch.isfinite(loss):
            print(f"skipping non-finite loss at {i}", flush=True); continue
        loss.backward(); losses.append(loss.item() * accum)
        if (i + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            if (i + 1) % 80 == 0:
                m.save_pretrained(str(ck)); (ck / "state.json").write_text(json.dumps({"done": i + 1, "seconds": prev + time.time() - t0}))
        if (i + 1) % 40 == 0:
            print(f"step {i+1}/{len(rows)} loss(last40)={np.mean(losses[-40:]):.3f} {time.time()-t0:.0f}s", flush=True)
    m.save_pretrained(str(ADAPTER))
    _log_time("train", prev + time.time() - t0, {"examples": len(rows), "resumed_at": start, "loss_first40": float(np.mean(losses[:40])), "loss_last40": float(np.mean(losses[-40:]))})


class HFLM:
    def __init__(self, tok, m):
        import torch
        self.tok, self.m, self.torch = tok, m.eval(), torch
        self.cache = {}

    def option_probs(self, system, user, labels):
        k = hashlib.sha1((system + user + "|".join(labels)).encode()).hexdigest()
        if k not in self.cache:
            with self.torch.no_grad():
                lg = self.m(input_ids=self.torch.tensor([encode(self.tok, system, user)])).logits[0, -1].float()
            ids = [self.tok.encode(l, add_special_tokens=False)[0] for l in labels]
            sel = lg[ids].numpy().astype(float)
            p = np.exp(sel - sel.max()); self.cache[k] = (p / p.sum()).tolist()
        return np.asarray(self.cache[k])


def simulate(lm, exp, n=N_EVAL, seed=0):
    """Same protocol as Runner.run (persona drawn from the study's pool, one sampled answer per item) on the first MAX_ITEMS items."""
    out = {}
    for arm in ("A", "B"):
        rng = np.random.default_rng(int(hashlib.md5(f"{seed}|{exp.id}|{arm}".encode()).hexdigest()[:8], 16))
        idx = rng.choice(len(exp.persona_pool), size=n, replace=True)
        draws, expv = [], []
        for i in idx:
            per = exp.persona_text(exp.persona_pool.iloc[i]); sc, ex = [], []
            for item in exp.items[:MAX_ITEMS]:
                pre = None
                if item.pre is not None and arm in item.pre_arms:
                    t, l, _ = render(item.pre, arm, per)
                    pre = item.pre.options[rng.choice(len(l), p=lm.option_probs(SYSTEM, t, l))]
                t, labs, vals = render(item, arm, per, pre)
                p = lm.option_probs(SYSTEM, t, labs)
                sc.append(vals[labs[rng.choice(len(labs), p=p)]]); ex.append(sum(p[j] * vals[l] for j, l in enumerate(labs)))
            draws.append(float(np.mean(sc))); expv.append(float(np.mean(ex)))
        out[arm] = {"draws": draws, "expected": expv}
    return out


def evaluate(which):
    OUT.mkdir(parents=True, exist_ok=True)
    tok, m = _tok_model(adapter=(which == "ft"))
    lm = HFLM(tok, m)
    name = f"qwen15hf_{which}_persona"
    path = OUT / f"{name}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    t0, calls0 = time.time(), 0
    for e in load_all():
        if e.id in res:
            continue
        res[e.id] = simulate(lm, e)
        res["_meta"] = {"label": f"Qwen2.5-1.5B {'+ SocSci210 LoRA' if which == 'ft' else 'base (HF bf16)'}, personas, n={N_EVAL}/arm",
                        "n_per_arm": N_EVAL, "seconds": round(time.time() - t0 + res.get("_meta", {}).get("seconds", 0), 1),
                        "calls": len(lm.cache) + res.get("_meta", {}).get("calls", 0) - calls0}
        calls0 = len(lm.cache)
        path.write_text(json.dumps(res))
        print(f"[{name}] {e.id} done {time.time()-t0:.0f}s calls={len(lm.cache)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["data", "train", "eval"]); ap.add_argument("--which", default="both")
    a = ap.parse_args()
    if a.stage == "data": build_data()
    elif a.stage == "train":
        if not ADAPTER.exists(): train()
    else:
        for w in (["raw", "ft"] if a.which == "both" else [a.which]): evaluate(w)
