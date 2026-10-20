import json
import numpy as np
import effect_fidelity.simulate as sm
import effect_fidelity.ft_eval as fe
import effect_fidelity.finetune as ft
from effect_fidelity.experiments import load_all


class Fake:
    def __init__(self): self.n = 0
    def option_probs(self, s, u, labels):
        self.n += 1
        p = np.arange(1, len(labels) + 1, dtype=float); return p / p.sum()


def _patch(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(fe, "SIMS", tmp_path / "sims"); monkeypatch.setattr(fe, "PPI", tmp_path / "ppi")
    monkeypatch.setattr(fe, "TIMING", tmp_path / "t.json"); monkeypatch.setattr(ft, "N_EVAL", 4)


def test_sims_and_ppi_resume(tmp_path, monkeypatch):
    _patch(tmp_path, monkeypatch)
    exps = [e for e in load_all() if e.id == "ml1_sunk_cost"]
    lm = Fake()
    fe.run_sims("ft", lm=lm, exps=exps); fe.run_ppi("ft", lm=lm, exps=exps, pool=6)
    n = lm.n; assert n > 0
    s = json.loads((tmp_path / "sims" / "qwen15hf_ft_persona.json").read_text())
    p = json.loads((tmp_path / "ppi" / "qwen15hf_ft_persona.json").read_text())
    assert len(s["ml1_sunk_cost"]["A"]["draws"]) == 4 and len(p["ml1_sunk_cost"]["B"]["f"]) == 6
    t = json.loads((tmp_path / "t.json").read_text()); assert t["sims_ft"]["calls"] > 0
    fe.run_sims("ft", lm=Fake(), exps=exps)  # already done -> no new work
    assert json.loads((tmp_path / "t.json").read_text())["sims_ft"] == t["sims_ft"]


def test_cost_table_labels_estimates():
    c = fe.cost_table({"sims_ft": {"seconds": 100.0, "calls": 200}, "train": {"seconds": 3600}}, {"ft": 2.0})
    r = c.iloc[0]
    assert abs(r.cpu_hours_per_1k - 0.5 * 2 * 1000 / 3600) < 1e-9 and abs(r.usd_per_1k_cpu_est - r.cpu_hours_per_1k * fe.CPU_USD_HR) < 1e-9
