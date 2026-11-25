# A jaw reading that tells the target from other wood — protocol, November 25, 2026

The one-beam jaw-presence rule was refuted the same day
([result](EVAL_PROTOCOL_JAW_PRESENCE_2026-11-25.md#result--november-25-2026)):
with the jaw 20 mm short of the spur, other wood blocked a beam and the gate
accepted 5 of 87 closures. The user asked for the next reading to be tested
offline, registered first, on CPU.

A reading has to be designed on some recordings and tested on others. This
document is written in two steps, and each is committed before the data it
governs is opened.

## Step 1 — the split (committed before any beam pattern or range was opened)

`tools/jaw_reading_split.py` assigns every capture to **design** or **test**.

- **By target**, so a rule is tested on spurs it was not designed on. Design:
  15004, 14884, 12261, 21386, 36786. Test: 14944, 8175, 12321, 28610. Any other
  target: design when its component number is even, test when odd.
- **One target is divided.** All five refuted closures are on target 8235. Its
  captures from `lighting-20260919` are design and all its others are test,
  which puts three of the five on the design side and two on the test side.

Among captures with the tool point above the ground this gives 115 design
captures (39 reach closing, 44 have a ready cycle) and 90 test captures (48 and
51).

**The limit of this split.** No held-out *target* has a known failure. The test
side can show that a reading refuses the two held-out failures on 8235 and
keeps the recorded cuts on four other targets. It cannot show that the reading
would refuse neighbouring wood near a spur nobody has looked at.

**What had been seen when the split was fixed.** Counts only: how many beams
were blocked per capture and by what kind of surface
([verdicts](evidence/jaw_presence_verdicts_2026-11-25.json),
[beam summary](evidence/jaw_beams_summary_2026-11-25.json)), and the result's
unregistered note that a count threshold does not separate the cases. Which
beams were blocked, and where along the gap, had not been opened for any
capture.

## Step 2 — the reading and its predictions

Not written yet. It will be added here, and pushed, before any test-side
reading is cast or opened.
