# One reading of a beam grid across the jaw gap: a presence sensor the mock pruner
# does not have yet (isaaclab_pruning.sensors.jaw_beam_grid). The safety gate can
# require blocked beams before the jaw may close; that is evidence about the jaw
# mouth which does not come from the controller's tracker.

# stamp: the control cycle (sensor bundle) this reading belongs to.
# frame_id: the tool frame the grid is fixed in.
std_msgs/Header header

# Beams in the grid. blocked must have exactly this many entries.
uint8 beams
# One flag per beam: true when something crosses it. Heights along the tool axis
# major, positions along the jaw minor.
bool[] blocked
