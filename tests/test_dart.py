import random
import statistics

from yogurt import config, dart, lab


def test_perturbation_is_smooth_bounded_and_spares_the_gripper():
    cfg = config.load("laptop").dart
    p = dart.Perturbation(cfg, random.Random(1))
    first = p.offset(1 / 30)
    assert all(abs(v) < 0.1 for v in first.values())  # ramps in: no jump when recording starts
    offsets = [p.offset(1 / 30) for _ in range(30 * 300)]
    for joint, sigma in cfg.sigma_deg.items():
        values = [o[joint] for o in offsets[60:]]
        if not sigma:
            assert set(values) == {0.0}
            continue
        assert 0.7 * sigma < statistics.pstdev(values) < 1.3 * sigma
        assert max(map(abs, values)) <= cfg.max_sigmas * sigma + 1e-9
        assert max(abs(a - b) for a, b in zip(values, values[1:], strict=False)) < 0.5 * sigma


def test_dart_setting_defaults_and_is_checked():
    cfg = config.load("laptop")
    base = {"dataset": "x", "new": True, "num_episodes": 1, "episode_time_s": 5, "reset_time_s": 0}
    assert lab.settings(base, cfg)["dart_fraction"] == cfg.dart.fraction
    assert lab.settings({**base, "dart_percent": 0}, cfg)["dart_fraction"] == 0
    for bad in (-1, 101, 12.5, True):
        try:
            lab.settings({**base, "dart_percent": bad}, cfg)
        except ValueError:
            continue
        raise AssertionError(bad)
