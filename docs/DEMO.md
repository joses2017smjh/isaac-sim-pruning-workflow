# Show the pruning demo

Run the four commands in the [README](../README.md), then open
`demo-output/pruning_demo.html` in a browser. Keep the generated HTML, GIF,
and JSON together when sharing the folder. No web server is required.

## Capture sequence

1. Set your browser to 100% zoom with a viewport at least 1140 pixels wide.
2. Record the top GIF for **18 seconds**, one complete loop. It shows the clear
   approach first, then sensor blackout, then the nearby-wood rejection.
3. Scroll to **Inspect each sensor frame**. Select **Sensor blackout** and drag
   the slider from start to end over **5 seconds**. The last four frames have
   zero returns and the same tool pose.
4. Select **Nearby wood** and scrub to the end over **5 seconds**. Point out the
   failed clearance check. Finish on the results table.

The full recording is **28 seconds**, excluding scrolling. For the GitHub
README, use the generated 18-second GIF directly; no screen capture is needed.
For a slide, use [the poster](demo/pruning_demo.png). For an interview, bring
the offline HTML folder so you can inspect individual measured sensor frames.

## What the recording proves

Two source-offset 8×8 ToF sensors cast rays against procedural finite cylinders.
The demo applies the repository noise model, estimates a branch axis, commands
the scripted tool controller, and evaluates mouth, failure-volume, and angle
conditions. A bounded insertion follows the controller's 80 mm standoff.

The JSON contains the scene, seed, sensor configuration, tool pose, ranges,
validity masks, and geometry checks for every step. Playback slows or repeats
frames to give each scenario six seconds; sensor time remains recorded at 15 Hz.
Results are calculated from the simulated steps, not the playback frames.

The tool has ideal kinematics. There is no rendered robot, arm collision model,
physics contact, learned depth inference, or physical cutting. Nearby-wood
checks use known scene geometry and versioned legacy cutter proxies. The
close insertion may continue after the two ToF views lose overlap; that is an
open-loop limitation, not validated real-robot behavior.

The fusion display uses a synthetic noisy metric estimate on the **same rays**
as each ToF sensor. It does not claim calibrated cross-camera fusion. Paired
RMSE compares only shared valid samples; all-available RMSE also includes
filled misses and can be worse.

## Secondary evidence demos

After the motion demo, show the
[successful robot-import animation](demo/import_gate_success.gif), then the
[failed import gate](demo/import_gate_failure.gif). These are animated evidence
summaries from real cluster jobs. They are not recordings of a moving robot.
The [RTX result card](demo/gate0_rtx.png) is likewise a measured summary, not a
camera screenshot.

Regenerate those summaries from the repository root:

```bash
python tools/render_demos.py
```

A live-ToF success animation requires a passing report:

```bash
python tools/render_demos.py --smoke-evidence docs/evidence/smoke_<jobid>.json
```

The renderer rejects failed or incomplete evidence. See [the job ledger](../SLURM_JOBS.md).
