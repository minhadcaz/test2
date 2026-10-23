import pytest

from core.adaptive_sampler import decide


@pytest.mark.parametrize("sizes,used,elapsed,reason", [
    ([4], 4, 1, "consensus"), ([3, 1], 4, 1, "consensus"),
    ([2, 2], 4, 1, "continue"), ([3, 2], 5, 1, "continue"),
    ([], 4, 1, "continue"), ([], 32, 1, "max_candidates"),
    ([2], 2, 1, "continue"), ([4], 4, None, "time_budget"),
])
def test_stopping(sizes, used, elapsed, reason, config):
    groups = {str(i): [None] * size for i, size in enumerate(sizes)}
    if elapsed is None:  # just past whatever budget config.yaml sets
        elapsed = config["sampling"]["time_budget_seconds"] + 1
    decision = decide(groups, used, elapsed, config["sampling"])
    assert decision.reason == reason
    assert decision.should_stop == (reason != "continue")
