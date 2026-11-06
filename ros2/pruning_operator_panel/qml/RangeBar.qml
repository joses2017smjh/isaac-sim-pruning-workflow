// One time-of-flight sensor: its nearest usable return against the stop line.
import QtQuick 2.15
import "."

Item {
    id: bar
    property string label: ""
    property real value: NaN          // metres; NaN = no usable zone this cycle
    property real stopLine: 0.06      // the safety gate's clearance
    property real fullScale: 0.40     // metres shown across the bar
    readonly property bool known: !isNaN(value)
    readonly property bool tooClose: known && value < stopLine
    implicitHeight: 48

    Text {
        id: caption
        text: bar.label + ":  " + (bar.known ? (bar.value * 100).toFixed(1) + " cm" : "no usable zone")
        color: bar.tooClose ? "#ff8a80" : Theme.text
        font.pixelSize: Theme.textBody
    }
    Rectangle {
        id: track
        anchors.top: caption.bottom
        anchors.topMargin: Theme.spaceXs
        width: parent.width
        height: 18
        radius: 4
        color: Theme.surfaceRaised
        border.color: Theme.border
        Rectangle {
            width: bar.known ? Math.min(1.0, bar.value / bar.fullScale) * track.width : 0
            height: parent.height
            radius: 4
            color: bar.tooClose ? Theme.holdAction : Theme.pass
        }
        Rectangle {
            x: Math.min(1.0, bar.stopLine / bar.fullScale) * track.width - width / 2
            width: 3
            height: parent.height + 8
            y: -4
            color: Theme.limit
        }
    }
}
