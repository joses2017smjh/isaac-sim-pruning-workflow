#!/usr/bin/env python3
"""A jaw reading that asks for one centred spur and nothing else in the gap. No simulator import.

The one-beam rule (any beam blocked) was refuted on November 25, 2026: it cannot tell the target from neighbouring
wood. This rule reads more from each beam. Every beam of the grid across the jaw gap
(isaaclab_pruning.sensors.jaw_beam_grid) reports how far from each jaw face its first obstruction is, so an
obstruction has a thickness along the closing axis and a centre relative to the middle of the gap.

A beam is *spur-like* when its obstruction is centred within CENTRE_MAX_MM of the middle of the gap and no thicker
than the claimed spur's diameter plus THICKNESS_SLACK_MM. The claimed radius comes with the target estimate, from
the scene's description of the selected spur, not from the tracker. The rule accepts when at least
MIN_SPUR_BEAMS beams are spur-like and no beam sees anything that is not: a spur where the controller says it is,
and nothing else between the jaws.

The three numbers were chosen on the design side of tools/jaw_reading_split.py and are fixed here before any
test-side reading is cast (docs/EVAL_PROTOCOL_JAW_READING_2026-11-25.md).
"""

from __future__ import annotations

#: An obstruction's centre may be this far from the middle of the gap, in mm.
CENTRE_MAX_MM = 6.0
#: An obstruction may be this much thicker than the claimed spur's diameter, in mm.
THICKNESS_SLACK_MM = 2.0
#: Spur-like beams needed to accept.
MIN_SPUR_BEAMS = 2
CLEAR, SPUR, FOREIGN = "clear", "spur", "foreign"


def beam_kind(near_mm, far_mm, gap_mm, radius_mm):
    """What one beam's two ranges say: clear, a spur-like obstruction, or something else.

    ``near_mm`` and ``far_mm`` are the distances from the first and the second jaw face to the first obstruction,
    -1 when that side sees none. A beam that one side sees blocked and the other does not is not explained by a
    spur and counts as foreign."""
    if near_mm < 0 and far_mm < 0:
        return CLEAR
    if near_mm < 0 or far_mm < 0:
        return FOREIGN
    thickness = gap_mm - near_mm - far_mm
    centre = near_mm + thickness / 2.0 - gap_mm / 2.0
    if abs(centre) <= CENTRE_MAX_MM and thickness <= 2.0 * radius_mm + THICKNESS_SLACK_MM:
        return SPUR
    return FOREIGN


def read(ranges, gap_mm, radius_mm):
    """``{"accept", "spur_beams", "foreign_beams"}`` for one reading of the grid."""
    kinds = [beam_kind(near, far, gap_mm, radius_mm) for near, far in ranges]
    spur, foreign = kinds.count(SPUR), kinds.count(FOREIGN)
    return {"accept": spur >= MIN_SPUR_BEAMS and foreign == 0, "spur_beams": spur, "foreign_beams": foreign}


def accepts(ranges, gap_mm, radius_mm):
    return read(ranges, gap_mm, radius_mm)["accept"]
