"""Robot description constants that do not import Isaac Lab."""

from .ur5e_pruner import JointSpec, Ur5ePrunerSpec, load_ur5e_pruner_config, load_ur5e_pruner_spec

__all__ = ["JointSpec", "Ur5ePrunerSpec", "load_ur5e_pruner_config", "load_ur5e_pruner_spec"]
