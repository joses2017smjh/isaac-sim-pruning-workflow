# Ground audit — November 25, 2026

**In 29 of 233 recorded captures the robot is beneath the orchard's ground.**
The orchard's ground relief is drawn, and nothing collides with it. Every
selected spur is moved to one fixed point in the robot's workspace, and the
ground moves with the orchard. For a spur that grows near the ground, that puts
the ground surface above the robot. The simulated ToF sensors and the known-map
planner cast against the trees only, and the controller knows nothing about the
ground, so those runs proceeded like any other.

No registered verdict is edited by this audit. Each one stands as a record of
what the simulated system did in its scene. What it cannot be is evidence about
a physical approach to that spur: a real arm could not be where the simulated
one was.

## How it was checked

`tools/audit_ground.py`, run from a clean clone at `42194b2` (CPU job
`22151498`); [evidence](evidence/ground_audit_2026-11-25.json). It covers every capture that carries the
orchard's placement: 233 captures, 48,001 frames.

- **The test.** A point is beneath the ground when a vertical ray from it meets
  the ground above it and none below it. The relief is an open sheet, so a
  clearance test cannot see a robot that starts under it.
- **What was tested, in every frame.** The tool point, the jaw mouth, the wrist
  camera, and all eleven collision hulls of the robot, posed by forward
  kinematics from the recorded joint positions and sampled on their surfaces.
  Forward kinematics reproduced the recorded tool point within 0.008 mm in every
  capture.
- **Checked against the recordings.** At the first and last frame of each
  capture, a grid of 384 recorded depth pixels was compared with casts through
  the pixel centres. In the 29 affected captures (58 images) a median of 95% of
  the pixels agree within 1 mm once the ground is in the mesh set (lowest image
  89%), against a median of 7% with the trees alone (highest 25%). In the
  other 204 captures the median is 92% with the ground (lowest 80%); the rest are
  pixels on the robot, its jaws and the pedestal, which are in neither mesh set.

## What was found

| Target | Captures | What is beneath the ground | What the run did |
|---|---|---|---|
| tree0 530 | 15 | The robot's base link and the wrist camera in every frame of all 15. The tool point in every frame of 13; in the other two for 30 and 192 of 200 frames | 4 cut and detached, 4 reached closure and stopped, 7 stopped on the approach (6 hazard contact, 1 ToF clearance) |
| tree0 590 | 7 | Base link, camera, tool point and mouth, every frame | Stopped on the approach (5 vision invalid, 2 ToF clearance) |
| tree0 3721 | 3 | Base link, camera, tool point and mouth, every frame | Stopped at frame 0; the arm never moved |
| tree1 3433, 3673 | 1 each | Base link, camera, tool point and mouth, every frame | Stopped at frame 0 |
| tree1 3493 | 1 | Base link, camera, tool point and mouth, every frame | Stopped on the approach (ToF clearance) |
| tree0 4438 | 1 | The wrist camera and the tool body, every frame. The tool point, the mouth and the base link are above | Stopped at frame 0 |

In all 29 something is beneath the ground from the first frame to the last. In
28 of them that includes the robot's base link: the whole robot stands under
the surface.

In the other 204 captures nothing is beneath the ground in any frame, no arm
link included. That covers every capture on targets 14944, 15004, 8235 and
14884, and the four sweep targets that passed (8175, 21386, 28610, 36786).

## What was already known

An offline cast on October 5 included the ground
([evidence](evidence/home_visibility_check_2026-10-05.json)). It found the
ground 47 mm in front of target 3721's seed pixel, and the October 6 sweep
registered its prediction that 4438, 3433 and 3673 would not be visible from
home, which held. Nobody had checked whether the robot itself was beneath the
ground.

## Results that used an affected capture

Each document below carries a dated addendum.

| Result | Affected captures | What rested on them |
|---|---|---|
| [Population study, September 23](EVAL_PROTOCOL_2026-09-23.md) | 530 and 590 in both lights (4) | 4 of the 14 trials that recorded frames; the headline 0 of 40 does not change |
| [Approach strategies, September 24](EVAL_PROTOCOL_STRATEGIES_2026-09-23.md) | 530 and 590 under three strategies (6), and the baseline rows above | Two of seven rows of the tree0 table, and P1's reading of 530 |
| [Perception variants, September 28](EVAL_PROTOCOL_PERCEPTION_2026-09-26.md) | 530 and 590 under two variants (4); 3493 in the seeded draw (1) | P1's supporting observation is on 590 |
| [Planned approach, September 30](EVAL_PROTOCOL_PLANNED_APPROACH_2026-09-27.md) | 530, three repeats (3) | Everything the result says about 530 |
| [Jaw in view, October 2](EVAL_PROTOCOL_JAW_IN_VIEW_2026-10-01.md) | 530, three runs (3) | The 3 of 3 pass, and P4 to P8 |
| [Hold generalization, October 5](EVAL_PROTOCOL_HOLD_GENERALIZATION_2026-10-04.md) | 3721, three runs; control 530 (4) | 3721's frame-0 stops; one of the two controls |
| [Hold sweep, October 6](EVAL_PROTOCOL_HOLD_SWEEP_2026-10-06.md) | 4438, 3433, 3673; control 530 (4) | S1's three rejected targets; S7's failed control |
| [Depth-appearance replay register](EVAL_PROTOCOL_DEPTH_APPEARANCE_HELDOUT_2026-10-01.md) | 18 of its 129 regression runs | Inputs to offline replays |
| [Planner scope, October 2](SCOPE_CLOSURE_HOLD_GENERALIZATION_2026-10-02.md) | — | "Exactly 530 and 19444 have a clear path" came from a planner with no ground |

The November 25 [mouth study](ROS2_SIL.md#november-25-what-could-witness-the-target-in-the-mouth-exploratory)
and [jaw-presence replay](EVAL_PROTOCOL_JAW_PRESENCE_2026-11-25.md) set these
captures aside when they ran. The replay's criterion was the tool point alone,
which kept the 4438 capture; it stops at frame 0 and reaches no prediction about
closure.

## What this audit does not say

- **It does not explain any recorded stop.** No hazard contact, ToF stop or
  vision stop is re-attributed to the ground here. Whether the ground changed
  what the tracker or the sensors saw in these runs was not analysed.
- **It does not cover everything.** 81 captures recorded no frames and 8 early
  captures carry no orchard placement. The link test uses the collision hulls,
  not the drawn meshes.
- **It does not say what a physical run would do.** It says these were not
  physical scenes.
