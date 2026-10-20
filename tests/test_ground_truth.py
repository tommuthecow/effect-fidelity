"""Ground truth must come from the public microdata and match the well-known direction of each effect."""
import numpy as np
import pytest
from effect_fidelity.experiments import load_all


@pytest.fixture(scope="module")
def E():
    return {e.id: e for e in load_all()}


def eff(e):
    return (e.real["B"].mean() - e.real["A"].mean()) * e.scale


def test_seventeen_experiments_with_real_data(E):
    assert len(E) == 17
    for e in E.values():
        assert len(e.real["A"]) > 900 and len(e.real["B"]) > 900, e.id
        assert len(e.persona_pool) > 1000


def test_classic_directions(E):
    assert eff(E["ml1_gain_loss"]) > 20          # loss frame -> more risk seeking
    assert eff(E["ml1_allowed_forbidden"]) > 10  # 'not allow' > 'forbid'
    assert eff(E["ml1_scales"]) > 10             # high-category scale -> more '>2.5h'
    assert eff(E["ml1_reciprocity"]) > 10
    assert eff(E["gss_welfare"]) > 40            # 'assistance to the poor' >> 'welfare'
    assert eff(E["gss_cities"]) < -20
    assert abs(eff(E["ml1_money"])) < 0.1 and abs(eff(E["ml1_flag"])) < 0.1  # failed-to-replicate primes


def test_hillstrom_matches_known_lift(E):
    assert eff(E["hill_mens"]) == pytest.approx(7.66, abs=0.01)
    assert eff(E["hill_womens"]) == pytest.approx(4.52, abs=0.01)


def test_binary_outcomes_are_binary(E):
    for e in E.values():
        if e.unit == "pp":
            for a in "AB":
                assert set(np.unique(e.real[a])) <= {0.0, 1.0}
