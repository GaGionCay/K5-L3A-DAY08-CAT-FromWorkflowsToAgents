"""M5-M7 self-check (Advanced/Expert). Run: python -m pytest -q -m student -o addopts= -k advanced
Public vectors only; the instructor grades with hidden ones."""
import copy
import math
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.student


@pytest.fixture(scope="module")
def stats():
    from lab9 import stats

    return stats


@pytest.mark.parametrize("n,c,k,want", [(5, 5, 5, 1.0), (5, 0, 1, 0.0), (5, 2, 1, 0.4), (10, 3, 3, 1 - 35 / 120)])
def test_pass_at_k(stats, n, c, k, want):
    assert math.isclose(stats.pass_at_k(n, c, k), want, abs_tol=1e-9)


@pytest.mark.parametrize("n,c,k,want", [(5, 5, 5, 1.0), (5, 4, 5, 0.0), (5, 4, 3, 0.4), (10, 8, 2, 28 / 45)])
def test_pass_hat_k(stats, n, c, k, want):
    assert math.isclose(stats.pass_hat_k(n, c, k), want, abs_tol=1e-9)


def test_pass_k_rejects_bad_args(stats):
    for args in [(5, 6, 1), (5, 2, 0), (3, 1, 4)]:
        with pytest.raises(ValueError):
            stats.pass_at_k(*args)


def _report(rows, metrics, items=4, seeds=(1, 2)):
    return {"manifest": {"eval_set_sha": "x", "items": items, "seeds": list(seeds)},
            "rows": rows, "metrics": metrics, "evaluator_failures": []}


ROWS = [{"item_id": f"E-{i:03d}", "seed": s, "pass_rate": True, "slices": ["denial"] if i < 2 else []}
        for i in range(4) for s in (1, 2)]
SPEC = {"eval_set_version": "v2", "metrics": [{"id": "pass_rate", "kind": "capability", "blocking": True,
                                               "threshold": {"min": 0.8}}]}


def test_integrity_flags_nan_and_dropped_rows():
    from lab9 import gate

    base = _report(copy.deepcopy(ROWS), {"pass_rate": 1.0})
    assert gate.integrity_check(base, base, SPEC) == []
    for invalid in (float("nan"), float("inf"), float("-inf")):
        non_finite = _report(copy.deepcopy(ROWS), {"pass_rate": invalid})
        assert any("NaN or inf" in p for p in gate.integrity_check(non_finite, base, SPEC))
    dropped = _report(copy.deepcopy(ROWS[:-1]), {"pass_rate": 1.0})
    assert any("row count mismatch" in p for p in gate.integrity_check(dropped, base, SPEC))


def test_integrity_flags_cherry_picked_seeds_and_duplicate_rows():
    from lab9 import gate

    base = _report(copy.deepcopy(ROWS), {"pass_rate": 1.0})
    changed_seed = copy.deepcopy(ROWS)
    changed_seed[0]["seed"] = 3
    cherry_picked = _report(changed_seed, {"pass_rate": 1.0})
    assert any("set differs from baseline" in p for p in gate.integrity_check(cherry_picked, base, SPEC))

    duplicated = copy.deepcopy(ROWS)
    duplicated[-1] = copy.deepcopy(duplicated[0])
    duplicate_report = _report(duplicated, {"pass_rate": 1.0})
    problems = gate.integrity_check(duplicate_report, base, SPEC)
    assert any("set differs from baseline" in p for p in problems)


def test_integrity_flags_evaluator_crashes_and_row_non_finite_values():
    from lab9 import gate

    base = _report(copy.deepcopy(ROWS), {"pass_rate": 1.0})
    crashed_row = copy.deepcopy(ROWS)
    crashed_row[0]["pass_rate"] = None
    crashed = _report(crashed_row, {"pass_rate": 1.0})
    assert any("has None for metric pass_rate" in p for p in gate.integrity_check(crashed, base, SPEC))

    failure = _report(copy.deepcopy(ROWS), {"pass_rate": 1.0})
    failure["evaluator_failures"] = ["E-001@1:TrialPass:crashed"]
    assert any("evaluator_failures non-empty" in p for p in gate.integrity_check(failure, base, SPEC))

    invalid_row = copy.deepcopy(ROWS)
    invalid_row[0]["pass_rate"] = float("nan")
    invalid = _report(invalid_row, {"pass_rate": 1.0})
    assert any("row E-000 seed 1 metric pass_rate is NaN or inf" in p
               for p in gate.integrity_check(invalid, base, SPEC))


def test_integrity_flags_loosened_spec():
    from lab9 import gate

    base = _report(copy.deepcopy(ROWS), {"pass_rate": 1.0})
    loose = copy.deepcopy(SPEC)
    loose["metrics"][0]["threshold"]["min"] = 0.5
    assert gate.integrity_check(base, base, loose, SPEC)
    assert gate.integrity_check(base, base, SPEC, SPEC) == []


def test_integrity_flags_ratchet_disappearances_and_slice_relaxations():
    from lab9 import gate

    base = _report(copy.deepcopy(ROWS), {"pass_rate": 1.0})
    base_spec = copy.deepcopy(SPEC)
    base_spec["metrics"][0]["threshold"]["no_significant_drop"] = True
    base_spec["slices"] = [
        {"name": "denial", "metric": "pass_rate", "min": 0.9,
         "no_significant_drop": True, "blocking": True}
    ]

    relaxed = copy.deepcopy(base_spec)
    relaxed["metrics"][0]["blocking"] = False
    relaxed["metrics"][0]["threshold"]["no_significant_drop"] = False
    relaxed["slices"][0]["blocking"] = False
    relaxed["slices"][0]["min"] = 0.8
    relaxed["slices"][0]["no_significant_drop"] = False
    problems = gate.integrity_check(base, base, relaxed, base_spec)
    assert any("metric pass_rate relaxed" in p for p in problems)
    assert any("dropped no_significant_drop" in p for p in problems)
    assert any("slice denial relaxed" in p for p in problems)
    assert any("slice denial min threshold loosened" in p for p in problems)

    removed = copy.deepcopy(base_spec)
    removed["metrics"] = []
    removed["slices"] = []
    problems = gate.integrity_check(base, base, removed, base_spec)
    assert any("metric pass_rate disappeared" in p for p in problems)
    assert any("slice denial disappeared" in p for p in problems)


def test_integrity_ratchet_only_compares_the_same_eval_set_version():
    from lab9 import gate

    base = _report(copy.deepcopy(ROWS), {"pass_rate": 1.0})
    previous_spec = copy.deepcopy(SPEC)
    previous_spec["eval_set_version"] = "v1"
    loose_v2 = copy.deepcopy(SPEC)
    loose_v2["metrics"][0]["threshold"]["min"] = 0.1
    assert gate.integrity_check(base, base, loose_v2, previous_spec) == []


def test_integrity_flags_slice_tag_tampering():
    from lab9 import gate

    base = _report(copy.deepcopy(ROWS), {"pass_rate": 1.0})
    relabeled = copy.deepcopy(ROWS)
    relabeled[0]["slices"] = ["adversarial"]
    pr = _report(relabeled, {"pass_rate": 1.0})
    assert any("slice tags differ from baseline" in p for p in gate.integrity_check(pr, base, SPEC))


def test_slice_check_blocks_a_slice_below_floor():
    from lab9 import gate

    base = _report(copy.deepcopy(ROWS), {"pass_rate": 1.0})
    rows = copy.deepcopy(ROWS)
    for r in rows:
        if "denial" in r["slices"] and r["seed"] == 1:
            r["pass_rate"] = False
    spec = dict(SPEC, slices=[{"name": "denial", "metric": "pass_rate", "min": 0.9, "blocking": True}])
    blocked, _ = gate.slice_check(_report(rows, {"pass_rate": 0.75}), base, spec)
    assert blocked == ["slice:denial/pass_rate"]


def test_slice_check_blocks_paired_regression():
    from lab9 import gate

    base = _report(copy.deepcopy(ROWS), {"pass_rate": 1.0})
    rows = copy.deepcopy(ROWS)
    for row in rows:
        if "denial" in row["slices"]:
            row["pass_rate"] = False
    spec = dict(SPEC, bootstrap={"n_resamples": 100, "rng": 1}, slices=[
        {"name": "denial", "metric": "pass_rate", "no_significant_drop": True, "blocking": True}
    ])
    blocked, lines = gate.slice_check(_report(rows, {"pass_rate": 0.5}), base, spec)
    assert blocked == ["slice:denial/pass_rate"]
    assert "CI" in lines[0]


def test_slice_check_fails_closed_when_blocking_slice_has_no_data():
    from lab9 import gate

    base = _report(copy.deepcopy(ROWS), {"pass_rate": 1.0})
    spec = dict(SPEC, slices=[
        {"name": "adversarial", "metric": "pass_rate", "min": 0.75, "blocking": True}
    ])
    blocked, lines = gate.slice_check(base, base, spec)
    assert blocked == ["slice:adversarial/pass_rate"]
    assert "no PR items in slice" in lines[0]


def test_gate_check_includes_slice_blocks_and_derived_pass_hat():
    from lab9 import gate

    rows = [
        {"item_id": f"E-{item:03d}", "seed": seed, "pass_rate": True, "slices": ["denial"]}
        for item in range(4) for seed in range(1, 6)
    ]
    report = _report(rows, {"pass_rate": 1.0, "pass_hat_5": 1.0}, seeds=tuple(range(1, 6)))
    spec = {
        "eval_set_version": "v2",
        "metrics": [{
            "id": "pass_hat_5", "blocking": True,
            "threshold": {"min": 0.8, "no_significant_drop": True},
        }],
        "slices": [{
            "name": "denial", "metric": "pass_rate", "min": 0.9, "blocking": True,
        }],
    }
    blocked, lines = gate.check(report, report, spec)
    assert blocked == []
    assert any("pass_hat_5" in line and "CI [+0.000, +0.000]" in line for line in lines)
    assert any("slice:denial/pass_rate" in line for line in lines)


def test_v2_configuration_has_required_failure_modes_metrics_and_slices():
    root = Path(__file__).resolve().parents[1]
    fm_map = yaml.safe_load((root / "config" / "fm_map.yaml").read_text(encoding="utf-8"))
    spec = yaml.safe_load((root / "config" / "metric_spec_v2.yaml").read_text(encoding="utf-8"))

    assert fm_map["version"] == "1.0"
    assert fm_map["mappings"] == {
        "ticket_over_db": "FM-01",
        "stale_policy": "FM-02",
        "clause_sum": "FM-03",
        "double_deduct": "FM-04",
        "sibling_refund": "FM-05",
        "denied_ignored": "FM-06",
        "reply_ledger_mismatch": "FM-07",
        "cross_principal": "FM-08",
    }

    pass_hat_5 = next(metric for metric in spec["metrics"] if metric["id"] == "pass_hat_5")
    assert pass_hat_5["threshold"]["min"] == 0.80
    assert pass_hat_5["blocking"] is True
    assert {slice_spec["name"] for slice_spec in spec["slices"]} >= {
        "money_movement", "denial", "adversarial"
    }
