# Provenance and licensing

This repository is a GitHub fork of
[`lukestroh/isaaclab-sensor-learning`](https://github.com/lukestroh/isaaclab-sensor-learning)
at `5701a774af7b8579269f924689aaf79b9574a53c`. Inherited files retain
their original history and per-file notices.

The upstream repository does not declare a repository-level license. Several
inherited files carry BSD-3-Clause headers, while the inherited `setup.py`
contains inconsistent Apache-2.0 metadata. This fork therefore does not add a
root license that could imply rights over all inherited content.

External sources with `license: NOASSERTION` in `third_party/sources.yaml` are
fetch-only. Their code, meshes, and URDFs must not be copied into this
repository until the owner documents a license or grants permission.

`OSUrobotics/pybullet-tree-sim` and
`UniversalRobots/Universal_Robots_ROS2_Description` declare BSD-3-Clause at
the pinned revisions. Their notices must travel with any files later vendored
from them.

The packed `bark_brown_02_diff.jpg` is a 512² downsample of the spur-depth
orchard albedo used as the Blender baseline (`bark_brown_02`). It is a
material sample for UsdPreviewSurface, not a relicense of blender_virtual_orchard.

For permission or licensing clarification, contact Jose Sanchez
<sanchej7@oregonstate.edu> and the relevant upstream owner.
