# isaaclab-pruning

Geometry, sensors, vision control, and task utilities for the
[robotic pruning workflow](https://github.com/joses2017smjh/isaac-sim-pruning-workflow).

Core modules import without launching Isaac Sim. Simulator adapters, the GPU
capture path, and the independent sequence grader live in the repository root.

```bash
python -m pip install -e '.[demo,dev]'
python -m pytest -q -m 'not isaacsim_ci'
```

Start at the [root README](../../README.md) for the recorded Isaac demo, CPU
quickstart, architecture, and measured evidence.
