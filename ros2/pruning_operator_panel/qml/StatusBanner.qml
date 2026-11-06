// The gate's call for the cycle on screen. One state at a time, by priority:
// review > stale > waiting > hold > pass.
import QtQuick 2.15
import "."

Rectangle {
    id: banner
    property bool compact: false
    readonly property string state_: backend.reviewMode ? "review"
                                     : backend.stale ? "stale"
                                     : !backend.haveDecision ? "waiting"
                                     : backend.passed ? "pass" : "hold"
    objectName: "banner"
    radius: Theme.radiusM
    implicitHeight: compact ? 96 : 120
    color: state_ === "pass" ? Theme.pass
           : state_ === "hold" ? Theme.hold
           : state_ === "stale" ? Theme.stale
           : state_ === "review" ? (backend.passed ? Theme.pass : Theme.hold)
           : Theme.waiting

    Column {
        anchors.centerIn: parent
        width: parent.width - 2 * Theme.spaceM
        spacing: Theme.spaceXs
        Text {
            width: parent.width
            horizontalAlignment: Text.AlignHCenter
            text: banner.state_ === "stale" ? "STALE"
                  : banner.state_ === "waiting" ? "WAITING"
                  : backend.passed ? "PASS" : "HOLD"
            color: "white"
            font.pixelSize: banner.compact ? Theme.textDisplayCompact : Theme.textDisplay
            font.bold: true
        }
        Text {
            width: parent.width
            horizontalAlignment: Text.AlignHCenter
            elide: Text.ElideRight
            color: "white"
            font.pixelSize: Theme.textTitle
            text: banner.state_ === "stale"
                  ? "no gate decision for " + backend.decisionAgeS.toFixed(1) + " s, do not trust the last call"
                  : banner.state_ === "waiting" ? "no safety decision yet"
                  : backend.passed ? "command forwarded to the arm" : backend.reason
        }
    }
}
