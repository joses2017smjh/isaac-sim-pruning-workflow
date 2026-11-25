#!/usr/bin/env python3
"""The design/test split for the next jaw reading, fixed before any beam pattern or range was opened.

The one-beam jaw-presence rule was refuted on November 25, 2026: with the jaw 20 mm short of the spur, other wood
blocked a beam and the gate accepted 5 of 87 closures (docs/EVAL_PROTOCOL_JAW_PRESENCE_2026-11-25.md). A better
reading has to be designed on some recordings and tested on others, or the test only shows that a rule fits the
data it was tuned on.

The split is by target, so that a rule is tested on spurs it was not designed on. One target is divided: all five
refuted closures are on target 8235, so its captures are split by the batch that recorded them, which puts three of
the five on the design side and two on the test side. That is the limit of this split: no held-out *target* has a
known failure, only held-out captures of the same target.

Committed at the commit that introduced this file, when the only things seen were the counts in
docs/evidence/jaw_presence_verdicts_2026-11-25.json and docs/evidence/jaw_beams_summary_2026-11-25.json: which
beams were blocked, and where along the gap, had not been opened for any capture.
"""

from __future__ import annotations

#: Targets whose captures may be used to design a reading.
DESIGN_TARGETS = frozenset({15004, 14884, 12261, 21386, 36786})
#: Targets whose captures are opened only after the reading and its predictions are registered.
TEST_TARGETS = frozenset({14944, 8175, 12321, 28610})
#: The one divided target: its captures from this batch are design, all its others are test.
DIVIDED_TARGET = 8235
DIVIDED_DESIGN_BATCH = "/lighting-20260919/"
SIDES = ("design", "test")


def component(target_id):
    """The component number of a target id such as ``tree1_SPUR_component_14944``."""
    return int(str(target_id).rsplit("_", 1)[1])


def side(capture, target_id):
    """``"design"`` or ``"test"`` for one capture. Targets not named above: even component numbers are design."""
    number = component(target_id)
    if number == DIVIDED_TARGET:
        return "design" if DIVIDED_DESIGN_BATCH in f"/{capture}/" else "test"
    if number in DESIGN_TARGETS:
        return "design"
    if number in TEST_TARGETS:
        return "test"
    return "design" if number % 2 == 0 else "test"
