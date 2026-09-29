import pytest

from cedge_core.regime.posterior import mixture_posterior


def test_mixture_posterior_at_center_is_symmetric():
    post = mixture_posterior(50.0)
    assert post["risk_on"] == pytest.approx(post["risk_off"])
    assert sum(post.values()) == pytest.approx(1.0)
    assert post["soft_patch"] > post["risk_on"]


def test_mixture_posterior_known_value_at_20():
    post = mixture_posterior(20.0)
    assert post["risk_off"] == pytest.approx(0.88054, abs=1e-4)
    assert post["soft_patch"] == pytest.approx(0.11917, abs=1e-4)
    assert post["risk_on"] == pytest.approx(0.00030, abs=1e-4)


def test_mixture_posterior_none_is_uniform():
    post = mixture_posterior(None)
    assert post == {"risk_off": pytest.approx(1 / 3),
                     "soft_patch": pytest.approx(1 / 3),
                     "risk_on": pytest.approx(1 / 3)}