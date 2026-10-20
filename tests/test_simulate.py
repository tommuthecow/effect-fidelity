from effect_fidelity.experiments import load_all
from effect_fidelity.simulate import render, Cache


def test_render_has_arm_specific_wording_and_persona_last():
    E = {e.id: e for e in load_all()}
    a, _, _ = render(E["ml1_allowed_forbidden"].items[0], "A", "PERSONA")
    b, _, _ = render(E["ml1_allowed_forbidden"].items[0], "B", "PERSONA")
    assert "forbid" in a and "allow" in b
    assert a.rstrip().endswith("(A, B).") and "PERSONA" in a.split("\n\n")[-2]
    lo, _, vlo = render(E["ml1_scales"].items[0], "A", "P")
    hi, _, vhi = render(E["ml1_scales"].items[0], "B", "P")
    assert "Up to a half hour" in lo and "Up to two and a half hours" in hi
    assert vlo["F"] == 1 and vlo["E"] == 0 and vhi["A"] == 0 and vhi["B"] == 1
    r, _, _ = render(E["ml1_reciprocity"].items[0], "B", "P", pre_answer="Yes")
    assert "North Korea should let American" in r and "You answered: Yes" in r


def test_reverse_scored_items():
    E = {e.id: e for e in load_all()}
    flag = E["ml1_flag"].items
    assert flag[0].values["7"] == 7 and flag[1].values["7"] == 1  # item 2 (abortion rights) is reverse-scored


def test_cache_roundtrip(tmp_path, monkeypatch):
    import effect_fidelity.simulate as sm
    monkeypatch.setattr(sm, "CACHE_DIR", tmp_path)
    c = Cache("t"); k = Cache.key("m", "s", "u", ["A"]); c.put(k, [1.0])
    assert Cache("t").get(k) == [1.0]
