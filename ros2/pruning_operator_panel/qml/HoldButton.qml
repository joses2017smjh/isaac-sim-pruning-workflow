// The operator's stop. Always enabled while live, never hidden by scrolling,
// and reachable from the keyboard (Space). A hold latches for the run.
import QtQuick 2.15
import QtQuick.Controls 2.15
import "."

Button {
    id: hold
    objectName: "holdButton"
    enabled: !backend.reviewMode
    implicitHeight: Theme.holdMinHeight + 8
    text: backend.reviewMode ? "REVIEW (no control)" : backend.holdRequested ? "HOLD SENT" : "HOLD   (Space)"
    onClicked: backend.requestHold("button")
    Accessible.name: "Hold the arm"
    contentItem: Text {
        text: hold.text
        color: "white"
        font.pixelSize: Theme.textHeadline + 4
        font.bold: true
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
    }
    background: Rectangle {
        radius: Theme.radiusM
        color: !hold.enabled ? Theme.waiting
               : hold.down ? Theme.holdActionPressed
               : backend.holdRequested ? "#5c1a15" : Theme.holdAction
        border.color: "#ffcdd2"
        border.width: hold.visualFocus ? 4 : 2
    }
}
