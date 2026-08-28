from __future__ import annotations

import pytest

from isaaclab_pruning.sim.pruning_env import ISAAC_IMPORT_ERROR, require_isaaclab


def test_pruning_env_is_gated_without_isaac_lab() -> None:
    try:
        import isaaclab  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="Gate 0"):
            require_isaaclab()
        assert "create_empty.py" in ISAAC_IMPORT_ERROR
    else:
        require_isaaclab()
