# Provenance and licensing

This repository preserves the history of
`lukestroh/isaaclab-sensor-learning` at
`5701a774af7b8579269f924689aaf79b9574a53c` (commit date 2026-06-30).
The original GitHub URL returned repository-not-found when checked on
2026-08-31. The commit remains available in
[`joses2017smjh/isaac-sim-pruning-workflow`](https://github.com/joses2017smjh/isaac-sim-pruning-workflow).
Inherited files retain their history and per-file notices. The similarly named
`lukestroh/sensor-learning` repository is a different project.

The upstream repository does not declare a repository-level license. Several
inherited files carry BSD-3-Clause headers, while the inherited `setup.py`
contains inconsistent Apache-2.0 metadata. This fork therefore does not add a
root license that could imply rights over all inherited content.

External source pins, integration status, and scoped license findings are in
`third_party/sources.yaml`. `NOASSERTION` at a repository root does not erase a
license found inside a package, and a package-local license does not license
unrelated sibling packages.

At the reviewed `branch_detection_system` revision, the repository root has no
license declaration. The selected `branch_detection_system_description`,
`final_approach_controller`, `vl53l8cx_bringup`, and `vl53l8cx_msgs` packages
contain BSD-3-Clause license files and package metadata. The SolidWorks archive
inside the description package has no CAD-specific provenance or notice; do
not assume that the package license resolves third-party CAD rights. Preserve
the package notice and obtain clarification before redistributing CAD or
derived meshes.

The `ag-robot` and `linear-slider` roots also have no root declaration, while
their selected description packages contain BSD-3-Clause files. The tracked
`ag-robot` temporary URDF is retained only as a composition reference, not as a
redistributable complete robot artifact.

Sources marked `reference_only` are research evidence and are not fetched by
default. Sources marked `fetch_only` remain outside this repository unless
their scoped license and required notices allow a deliberate vendoring step.
Do not copy material from a root-`NOASSERTION` source merely because its GitHub
repository is public.

The reviewed revisions of `OSUrobotics/pybullet-tree-sim` and
`UniversalRobots/Universal_Robots_ROS2_Description` declare BSD-3-Clause at
the pinned revisions. Their notices must travel with any files later vendored
from them. The local cutter failure proxy remains a derived experiment
contract; its presence does not establish that it is current mock-pruner CAD.

The packed `bark_brown_02_diff.jpg` is a 512² downsample of the spur-depth
orchard albedo used as the Blender baseline (`bark_brown_02`). It is a
material sample for UsdPreviewSurface, not a relicense of blender_virtual_orchard.

For permission or licensing clarification, contact Jose Sanchez
<sanchej7@oregonstate.edu> and the relevant upstream owner.
