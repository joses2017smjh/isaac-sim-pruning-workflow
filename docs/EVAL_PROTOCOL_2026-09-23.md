# Task success-rate protocol — September 23, 2026

This protocol is registered **before** the trials are submitted. It fixes the
target population, the selection rule, the conditions, the primary metric and
how rejections are counted, so the result cannot be shaped after the fact.

Today the repository's honest answer to "how often does it work?" is *once, on
one known spur* (job `21328323`, 17/17). The
[lighting pilot](RESEARCH_EXPERIMENTS_2026-09-19.md) added five more sequences on
that same spur. Neither estimates a success rate. This protocol does.

## What is being estimated, and what is not

The estimate is a **task success rate over spur geometries, axis orientations and
local occlusion contexts, at a canonicalized approach pose.**

That phrasing is exact, and the reason is structural.
`blender_demo_scene.spawn_blender_demo_scene` translates the whole orchard so the
selected spur's centroid lands at a fixed workspace pose, and refuses to continue
if the realized centroid misses that pose by more than 1e-5 m:

```python
translation = target - rotation @ candidate["center_m"]
...
if center_error > 1e-5:
    raise ValueError(f"Selected spur does not match requested world target ...")
```

So across trials:

| Held fixed | Varies between trials |
|---|---|
| Spur centroid world pose, hence approach distance and standoff | Spur radius (4.85–10.07 mm) and length (47.9–100.6 mm) |
| Robot base, home pose, camera mount, gates and thresholds | Spur **axis orientation**, which differs widely between components |
| Frames, capture rate, render quality, physics settings | Local occlusion: the rest of the tree translates rigidly with the spur |
| Daylight preset within a condition | Surface appearance and texture at that spur |

**This is therefore not a reachability study.** Reachability is constant by
construction. Anyone reading the resulting number must read this paragraph with
it. A field success rate would also vary approach pose, and this does not.

## Target selection rule

Applied to mesh metadata with a fixed seed. No target is chosen by hand.

1. **Population.** Every connected component of `tree0_SPUR` and `tree1_SPUR` in
   the hash-verified two-tree export. Measured: **310** components on tree0 and
   **399** on tree1, which matches the exporter's own recorded `component_count`
   for each object.
2. **Measurement.** Each component's centre, axis, length and maximum radius come
   from `blender_component.component_geometry`, the same helper the simulator
   uses. Nothing is re-derived or approximated for this protocol.
3. **Screen.** Keep components with `0 < max_radius_m <= 0.012`, which is the
   existing demo jaw-fit rule enforced by `blender_demo_scene.select_component`.
   This admits **185** spurs on tree0 and **259** on tree1, a pool of **444**.
   The 265 screened-out components are recorded by vertex ID in the evidence file,
   so the screen is auditable rather than asserted.
4. **Sample.** Draw **10 per tree** with `random.Random("20260923:tree<N>")` over
   the pool ordered by `component_first_vertex`, then record the draw ordered by
   vertex ID so the register is stable regardless of draw order.
5. **N = 20** targets, spanning both trees.

Regenerate the register with one command (USD is required to read the mesh, so
this runs under the pinned Isaac virtualenv):

```bash
/nfs/hpc/share/sanchej7/Humanoid_Lite/venv-isaac60/bin/python tools/enumerate_spur_targets.py \
  --export-dir artifacts/blender_scene/orchard_two_trees_v1 \
  --per-tree 10 --seed 20260923 \
  --output docs/evidence/eval_targets_2026-09-23.json
```

The registered set is [`docs/evidence/eval_targets_2026-09-23.json`](evidence/eval_targets_2026-09-23.json).
**The one-target success `21328323` used `tree0_SPUR_component_8235`, which is not
in this set**, so this estimate does not reuse the trial it is meant to generalize.

## Conditions

| Array | Light | Tracker | Trials | Purpose |
|---|---|---|---|---|
| A | source | raw | 20 | The primary success-rate estimate |
| B | morning | raw | 20 | Lighting stress, only if the budget allows |

Source light comes first, as the pilot's protocol requires. Morning is the chosen
stress condition rather than evening because the pilot measured it as the harder
one: the raw tracker fell to **4 surviving features against its own
`min_features` floor of 4**, and to 0.1735 confidence against a 0.15 floor, while
still completing. Evening was less severe.

The tracker is **raw**. CLAHE was dropped as a rejected diagnostic on
September 23 and must not be silently reintroduced. Every frozen plan records
`photometric_normalization` and the runner verifies it against the report, as it
already does for daylight.

No threshold, gate or grader check is changed for this sweep. If one looks wrong,
it is proposed as a separate, labelled experiment, never adjusted to raise a rate.

## Primary metric

**X / N completed sequences**, where a completion is a pass on all 17 checks of
the independent grader `tools/validate_vision_sequence.py`, reported with a
**Wilson 95% confidence interval**.

- `COMPLETED (0:0)` from Slurm is accounting, never a pass.
- A missing or unreadable report is `incomplete`, which is **not** a pass.
- Failed runs stay in the table and in the denominator.

Secondary, reported but not used to decide anything: per-condition rates, the
failure taxonomy, tracking coverage against position error, and the feature and
confidence margins that the lighting pilot showed to be informative.

## How pre-recording rejections count

Every registered target is an attempt. **N is fixed at 20 before submission and
does not shrink.**

| Outcome | Counts in N | Counts in X | Recorded as |
|---|---|---|---|
| Graded pass, 17/17 | yes | yes | `pass` |
| Graded fail, any check | yes | no | `fail`, with the first failed check |
| Rejected for visibility inside the trial | yes | no | `rejected_visibility` |
| Capture missing or unreadable | yes | no | `incomplete` |
| Scheduler failure before the renderer starts | yes | no | `infrastructure`, reported separately as well |

The visibility decision is made **inside** the GPU trial: the renderer validates
the known selection once and reports `selected_branch_occluded_or_wrong_surface`.
A rejected target therefore still consumes allocation time, and it still counts.
Excluding occluded targets would measure a hand-cleaned orchard, not this one.

Infrastructure failures are also reported separately, so a reader can see a rate
that excludes them, but the headline X/N is the inclusive one.

## Budget and resources

Measured from the pilot: mean **16.5 minutes** per 200-frame trial on one A40,
array concurrency `%1`. One run directory is **398 MB**.

| Array | Trials | GPU-minutes | Wall time at `%1` | Storage |
|---|---|---|---|---|
| A (source) | 20 | 330 | ~5.5 h | ~8.0 GB |
| B (morning) | 20 | 330 | ~5.5 h | ~8.0 GB |
| Both | 40 | 660 | ~11 h | ~16 GB |

`/nfs/hpc/share` had 162 GB free when this was written. Home is 92% full and must
not receive outputs. Each array is submitted separately and requires explicit
approval first. No existing job is modified, cancelled, held or requeued.

## Aggregation

Results are aggregated through a small DuckDB layer whose queries live in `sql/`,
run against the per-run evidence JSON. That layer is the real aggregation path
for the reported numbers, not a demonstration beside them, and it is covered by
tests on fixture JSON so the queries are checked without needing the captures.

## What this will still not establish

- Not a field success rate: approach pose is canonicalized, so reachability,
  base placement and approach-angle variation are untested.
- Not outdoor robustness: daylight presets are artistic, not radiometric, and
  rendered RGB lighting is not calibrated ToF sunlight behaviour.
- Not perception generality: branch identity, axis and radius come from mesh
  metadata, and depth is RTX optical-Z ground truth, not learned.
- Not cutting: the release is a visual jaw surrogate and a discrete rigid-piece
  detachment, not blade actuation or wood fracture.
- Not two-tree orchard coverage: both trees contribute targets, but each trial
  still presents one spur at one pose.
- Twenty trials give a wide interval. The interval is reported precisely so the
  width is visible rather than implied.

---

## Addendum, written during execution — September 23, 2026

**This section was written after the first trials ran, not before.** It is kept
separate from the registered protocol above so a reader can see exactly what was
decided in advance and what was learned afterwards. **It changes no threshold, no
target, no denominator and no counting rule.** N is still 20, and every attempted
target still counts.

### A rejection category the registration did not name

The registered table anticipated a target being "rejected for visibility inside
the trial". Execution surfaced a second, different pre-recording rejection that
the registration did not name:

```
RuntimeError: Orchard layout rejected: startup robot contact 226.126 N > 5 N
```

This is an existing guard in `hpc/inner/render_pruning_workflow.py:581`, not new
code. It fires before any motion. The cause is structural, and follows directly
from the canonicalized approach pose described above: the whole orchard is
translated so the selected spur's centroid lands at the fixed workspace pose, so
for some spurs that translation drives the rest of the tree into the robot. The
run is refused rather than started in contact.

The category is recorded as `rejected_layout_startup_contact`. Like every other
rejection, **it counts in the denominator**. Excluding these targets would report
a rate over "spurs whose surrounding tree happens to miss the robot", which is
not the population that was registered.

| Outcome | Counts in N | Counts in X | Recorded as |
|---|---|---|---|
| Layout refused before motion, startup contact above 5 N | yes | no | `rejected_layout_startup_contact` |

These trials are cheap: they abort in roughly one minute rather than sixteen, so
the array's real cost is below the registered budget rather than above it.

### What this says about the design, not just the run

The canonicalized approach pose buys a controlled comparison across spur
geometry, and this is the bill for it. A field evaluation would move the robot to
the branch; this moves the branch to the robot, and a fixed pose cannot suit
every spur in a real tree. That limitation was stated in the registration under
"what this will still not establish"; the startup-contact rejections are the
measurable form of it. A later protocol that varies base placement or approach
pose would remove this rejection class and replace it with a reachability
question, which is a different and larger study.

## Result — September 23, 2026

Arrays `21400715` (source) and `21400716` (morning) ran all 40 registered trials.
Every number below comes from [`docs/evidence/eval_2026-09-23.json`](evidence/eval_2026-09-23.json),
produced by `tools/aggregate_eval.py` through the queries in `sql/`.

| Rate | Successes / attempts | Wilson 95% |
|---|---|---|
| **Inclusive, both lights (headline)** | **0 / 40** | **0.000 – 0.088** |
| Inclusive, source light only | 0 / 20 | 0.000 – 0.161 |
| Inclusive, morning light only | 0 / 20 | 0.000 – 0.161 |
| Excluding infrastructure | 0 / 20 | 0.000 – 0.161 |
| Excluding infrastructure and layout rejection | 0 / 14 | 0.000 – 0.215 |

**No registered target was completed.** The one earlier success, `21328323`, does
not generalize to other spurs at the canonical approach pose.

| Outcome | Trials | Mean vision commands before stop |
|---|---|---|
| Infrastructure: target could not be presented (see below) | 20 | 0 |
| Layout refused, startup contact above 5 N | 6 | 0 |
| Stopped on hazard contact | 6 | 31.0 |
| Stopped on invalid vision | 6 | 17.3 |
| Stopped on time-of-flight minimum clearance | 2 | 28.5 |

Every one of the 20 targets ended the same way under source and morning light.
In this sweep, spur geometry and the surrounding tree decided the outcome;
lighting did not. That matches the lighting pilot, where presets moved tracker
margins but not outcomes.

Every stop is a gate refusing to continue: none approached through contact
unchecked, and none released without closure. The safety gates behaved as
designed; the controller did not complete the task.

### An error in the target register

**All ten tree1 targets could never be presented, and that is a mistake in this
protocol, not a property of the controller.** The renderer only accepts a tree1
spur that is among the export's ten listed candidates, and refuses any other
with `Unlisted tree1 components require a new geometry audit`. That constraint
was recorded during the Phase 0 audit, and then not re-applied when this
protocol widened sampling from the listed candidates to all 444 screened spurs.
All 20 tree1 trials aborted in about a minute each, before recording.

They are reported as `infrastructure` and **remain in the inclusive
denominator**, as registered. The consequence for interpretation is plain: this
sweep measured **tree0 only**. Nothing here says anything about tree1.

The launcher now refuses such a register on CPU before submission
(`queue_vision_robustness.unpresentable_targets`), and this exact register is
rejected by it. The register itself is left unchanged, because rewriting a
pre-registered target set after seeing results would defeat the registration.

A tree1 evaluation needs a new registration: either sample tree1 from the listed
candidates only, of which seven pass the jaw-fit screen, or first extend the
export's candidate list through the geometry audit the renderer asks for.
