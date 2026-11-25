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

## Step 2 — the reading and its predictions (committed before any test-side reading was cast)

### The reading

Each of the nine beams reports two ranges: how far from each jaw face its first
obstruction is. An obstruction then has a thickness along the closing axis and
a centre relative to the middle of the gap. `tools/cast_jaw_beams.py` casts the
ranges; the design side was cast at `95c56f5` (CPU job `22151802`).

### The rule (`tools/jaw_reading_rules.py`)

A beam is **spur-like** when its obstruction is centred within 6 mm of the
middle of the gap and no thicker than the claimed spur's diameter plus 2 mm.
The claimed radius comes with the target estimate, from the scene's description
of the selected spur, not from the tracker. The rule **accepts** when at least
two beams are spur-like and no beam sees anything that is not. In words: one
spur where the controller says it is, and nothing else between the jaws.

The three numbers (6 mm, 2 mm, two beams) were chosen on the design side.

### What the design side showed

Scored at `8d0f841` (CPU job `22151857`,
[evidence](evidence/jaw_reading_design_2026-11-25.json)): 115 design captures
with the tool point above the ground, 39 of which reach closing.

| | One-beam rule | This rule |
|---|---|---|
| Recorded closures kept, targets other than 8235 | 33 of 33 | 33 of 33 |
| Recorded closures kept, target 8235 | 6 of 6 | **0 of 6** |
| Closures refused with the tool 20 mm short | 36 of 39 | 39 of 39 |
| Displaced poses without the target, accepted | 73 of 633 | 21 of 633 |
| Displaced poses with the target, accepted | 951 of 951 | 487 of 951 |

Two things in that table are costs, and they are stated here so they are not
tuned away.

- **It refuses every recorded cut on target 8235.** Those cuts have other wood
  between the jaws beside the target. A rule strict enough to reject a
  neighbour cannot accept them. Whether a cut with other wood in the jaw should
  go ahead is a decision about the product, not about this rule.
- **It refuses about half of the displaced poses that still have the target
  between the jaws.** Those are poses where the spur is off centre.

The 21 accepted poses without the target are jaws displaced 30 or 50 mm along
the jaw onto other wood that is itself centred and spur-sized. This reading
cannot tell that from the target.

### Method

`tools/score_jaw_reading.py`, Python only. For each capture the rule turns every
reading into accept or refuse, that is fed to the reference gate as its
jaw-presence input, and the capture is replayed with the node's jaw rule
(`tools/gate_replay.py`), exactly as the one-beam rule was. No message, node or
C++ change is made unless the rule survives this test. Captures with the tool
point beneath the ground are left out, as before.

### Pre-registered predictions, test side

The test side is 90 captures with the tool point above the ground; 48 reach
closing: 40 on target 14944, one each on 8175, 12321 and 28610, and 5 on 8235.
Its readings are cast only after this section is pushed.

- **K1 — it keeps recorded cuts where the gap is clear.** Every test-side
  closure capture on a target other than 8235 (43) replays unchanged: same
  verdict, reason, latch and jaw output in every cycle. *Refuted* by one changed
  capture.
- **K2 — it refuses the jaw that stopped short.** With the tool-short reading,
  the jaw output stays at 0 in every test-side closure capture (48), including
  the two on 8235 that the one-beam rule accepted. *Refuted* by one accepted
  closure.
- **K3 — it refuses recorded cuts with other wood in the gap.** Every test-side
  closure capture on 8235 (5) is refused. This is the cost stated above,
  registered as an expected outcome. *Refuted* by one kept capture, which would
  mean the design side's picture of 8235 does not hold on these captures.
- **K4 — fewer false acceptances.** Of the test side's displaced poses without
  the target between the jaws, the rule accepts fewer than the one-beam rule.
  *Refuted* if it accepts as many or more.

Reported without a prediction: how many displaced poses that do have the target
the rule accepts.

### What a result would and would not show

- It is the same replay as before: recorded controller messages, exact
  geometry, beams as lines, jaws in no mesh set.
- The ranged reading assumes a sensor in each jaw face that measures distance
  across the gap. No such sensor is modelled in the simulator.
- K2 on target 8235 tests two held-out captures of a target the rule was
  designed on. No other target has a known failure to hold out.
- If K1, K2 and K4 hold, the rule is worth building into the gate. Whether to
  build it with the K3 cost is the user's decision.

## Result — November 25, 2026

The test side was cast and scored from a clean clone at the registration commit
`cb0952a` (CPU jobs `22151866` and `22151867`): 90 captures, 48 reaching
closing. Verdicts:
[jaw_reading_verdicts_2026-11-25.json](evidence/jaw_reading_verdicts_2026-11-25.json),
copied byte for byte.

| Prediction | Verdict | Measured on the test side |
|---|---|---|
| K1 — keeps recorded cuts on targets other than 8235 | **refuted** | 42 of 43 unchanged. The one changed capture is on target 8175: the rule stops its closure with `jaw_empty_during_closure` |
| K2 — refuses the jaw that stopped short | **supported** | 48 of 48 refused, including the 2 on 8235 that the one-beam rule accepted |
| K3 — refuses recorded cuts with other wood in the gap | **supported** | 5 of 5 on 8235 refused |
| K4 — fewer false acceptances | **supported** | 14 of 707 displaced poses without the target accepted, against 105 for the one-beam rule |

Without a prediction: of the 1,129 displaced poses that still have the target
between the jaws, the rule accepts 613.

### What this shows

- **The rule fixes what the one-beam rule got wrong.** On captures it was not
  designed on, it refuses every jaw that stopped short and accepts about one
  displaced pose in fifty without the target, where the one-beam rule accepted
  about one in seven.
- **It is too strict to build as it stands.** It stopped a recorded cut on a
  target other than 8235. In that capture, on 8175, three beams are blocked: one
  by another spur and two by the target, and the target sits 6.0 to 6.5 mm from
  the middle of the gap during closure. The rule's limit is 6 mm. K1's title
  assumed the gap is clear on every target other than 8235; on 8175 it is not,
  although the other spur's beam passed the rule and the target's did not. The
  limit was
  chosen on the design side, where no recorded spur was more than 5.9 mm off
  centre. The gate's own mouth tolerance is 8 mm, so a spur that far off centre
  is one the controller is allowed to cut.
- **Its cost held up as registered.** It refuses every recorded cut on 8235,
  where other wood is between the jaws.

### After the result, not registered

A centre limit at the gate's 8 mm mouth tolerance would keep the 8175 cut. On
the design side that limit accepted a few more poses without the target than
6 mm did. It cannot be tested as registered here: both sides of the split have
now been seen. A fair test of a revised limit needs recordings that do not
exist yet, on targets neither side used.

### What was not built

Nothing in the gate, the messages or the C++ node changed. The ranged reading
is still a calculation on recorded geometry.
