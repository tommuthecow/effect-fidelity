import json, types
import numpy as np
import pytest
import effect_fidelity.simulate as sm
import effect_fidelity.run_claude as rc
from effect_fidelity import claude as cl


def test_parse_probs():
    p = cl.parse_probs('Sure: {"A": 0.3, "B": 0.1}', ["A", "B", "C"])
    assert np.allclose(p, [0.75, 0.25, 0.0])
    with pytest.raises(ValueError):
        cl.parse_probs("no json", ["A"])


def test_api_backend_with_mocked_client():
    calls = []
    class Msgs:
        def create(self, **kw):
            calls.append(kw)
            if len(calls) == 1:
                raise RuntimeError("overloaded")  # exercise the retry
            return types.SimpleNamespace(content=[types.SimpleNamespace(text='{"A": 0.2, "B": 0.8}')])
    api = cl.ClaudeAPI(model="claude-test", client=types.SimpleNamespace(messages=Msgs()), retry_base=0)
    assert np.allclose(api.option_probs("sys", "q", ["A", "B"]), [0.2, 0.8])
    assert calls[-1]["model"] == "claude-test" and len(calls) == 2


def test_cli_backend_with_mocked_subprocess():
    seen = []
    def run(cmd, **kw):
        seen.append(cmd)
        return types.SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({"type": "result", "is_error": False, "result": '{"1": 1, "2": 3}'}))
    cli = cl.ClaudeCLI(model="sonnet", run=run, retry_base=0)
    assert np.allclose(cli.option_probs("s", "u", ["1", "2"]), [0.25, 0.75])
    assert seen[0][:2] == ["claude", "-p"] and seen[0][3:] == ["--output-format", "json", "--max-turns", "1", "--tools", "",
                                                              "--no-session-persistence", "--model", "sonnet"]


def _resp(code=0, **out):
    return lambda cmd, **kw: types.SimpleNamespace(returncode=code, stderr="", stdout=json.dumps(out))


def test_cli_default_passes_no_model_and_records_model_used():
    seen = []
    def run(cmd, **kw):
        seen.append(cmd)
        return _resp(type="result", is_error=False, result="OK",
                     modelUsage={"claude-haiku-x": {"outputTokens": 1}, "claude-sonnet-x": {"outputTokens": 9}})(cmd)
    cli = cl.ClaudeCLI(run=run, retry_base=0)
    assert cli.preflight() == "claude-sonnet-x"
    assert "--model" not in seen[0] and seen[0][2] == "Reply with OK"


def test_cli_label_fallback_when_model_not_reported():
    cli = cl.ClaudeCLI(run=_resp(type="result", is_error=False, result="OK"), retry_base=0)
    assert cli.preflight() == cl.CLI_DEFAULT_LABEL


def test_cli_fatal_error_fails_fast_with_full_message():
    n = {"c": 0}
    long_tail = "x" * 500 + " END"
    def run(cmd, **kw):
        n["c"] += 1
        return _resp(1, type="result", subtype="success", is_error=True, total_cost_usd=0,
                     result="There's an issue with the selected model (claude-sonnet-5-5). It may not exist or you may not have access to it. " + long_tail)(cmd)
    cli = cl.ClaudeCLI(model="claude-sonnet-5-5", run=run, retry_base=0)
    with pytest.raises(SystemExit) as e:
        cli.option_probs("s", "u", ["1", "2"])
    msg = str(e.value)
    assert n["c"] == 1 and "END" in msg and "without --model" in msg


def test_cli_not_logged_in_is_fatal_in_preflight():
    cli = cl.ClaudeCLI(run=_resp(1, is_error=True, result="Not logged in · Please run /login"), retry_base=0)
    with pytest.raises(SystemExit) as e:
        cli.preflight()
    assert "signed in" in str(e.value)


def test_cli_transient_error_is_retried():
    n = {"c": 0}
    def run(cmd, **kw):
        n["c"] += 1
        if n["c"] == 1:
            return _resp(1, is_error=True, result="API Error: 529 overloaded")(cmd)
        return _resp(type="result", is_error=False, result='{"1": 1, "2": 1}')(cmd)
    cli = cl.ClaudeCLI(run=run, retry_base=0)
    assert np.allclose(cli.option_probs("s", "u", ["1", "2"]), [0.5, 0.5]) and n["c"] == 2


def test_preflight_runs_before_prompt(monkeypatch):
    calls = []
    class Boom(cl.ClaudeCLI):
        def check(self): pass
        def preflight(self):
            calls.append("pre"); raise SystemExit("bad login")
    monkeypatch.setattr(cl, "ClaudeCLI", Boom)
    with pytest.raises(SystemExit):
        rc.main(["--n", "3", "--only", "ml1_sunk_cost"], confirm=lambda _: calls.append("prompt") or "y")
    assert calls == ["pre"]


def test_end_to_end_run_with_mock_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(rc, "ROOT", tmp_path)
    n = {"calls": 0}
    class Fake:
        def option_probs(self, s, u, labels):
            n["calls"] += 1
            return np.full(len(labels), 1 / len(labels))
    args = ["--simulator", "claude-cli", "--n", "3", "--only", "ml1_reciprocity", "ml1_sunk_cost", "--concurrency", "2"]
    assert rc.main(args, lm=Fake(), confirm=lambda _: "n") == 1 and n["calls"] == 0  # declined: nothing called
    assert rc.main(args, lm=Fake(), confirm=lambda _: "y") == 0
    first = n["calls"]; assert first > 0
    out = json.loads((tmp_path / "results" / "sims" / "claude_persona.json").read_text())
    assert {"ml1_reciprocity", "ml1_sunk_cost"} <= set(out) and len(out["ml1_sunk_cost"]["A"]["draws"]) == 3
    assert rc.main(args + ["-y"], lm=Fake()) == 0 and n["calls"] == first  # rerun is fully cached


def test_ppi_preds_written_and_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(rc, "ROOT", tmp_path)
    n = {"calls": 0}
    class Fake:
        def option_probs(self, s, u, labels):
            n["calls"] += 1
            return np.full(len(labels), 1 / len(labels))
    args = ["--n", "3", "--conditioning", "persona", "--ppi-pool", "5", "--only", "ml1_sunk_cost", "-y"]
    assert rc.main(args, lm=Fake()) == 0
    first = n["calls"]
    out = json.loads((tmp_path / "results" / "ppi_preds" / "claude_persona.json").read_text())
    a = out["ml1_sunk_cost"]["A"]; assert len(a["y"]) == len(a["f"]) == 5
    assert rc.main(args, lm=Fake()) == 0 and n["calls"] == first  # resume: all cached
