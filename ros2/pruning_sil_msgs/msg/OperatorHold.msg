# An operator's request to stop the arm. The safety gate latches it for the run.

# stamp: when the operator asked (wall clock of the sender).
std_msgs/Header header
bool hold
# Who asked, e.g. operator_panel.
string source
