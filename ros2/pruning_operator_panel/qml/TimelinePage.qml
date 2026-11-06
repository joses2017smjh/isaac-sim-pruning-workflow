// How the run unfolded: both ToF sensors against the stop line, the mouth distance
// against the closing tolerance, holds shaded, and every decision in a list.
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import Pruning.Panel 1.0
import "."

Item {
    id: page
    property string layoutMode: "wide"
    readonly property bool compact: layoutMode === "compact"

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: Theme.spaceL
        spacing: Theme.spaceM

        RowLayout {
            visible: backend.reviewMode
            Layout.fillWidth: true
            spacing: Theme.spaceS
            Button { text: "◀"; onClicked: backend.stepReview(-1); Accessible.name: "Previous decision" }
            Slider {
                objectName: "reviewSlider"
                Layout.fillWidth: true
                from: 0
                to: Math.max(0, backend.reviewCount - 1)
                stepSize: 1
                value: backend.reviewIndex
                onMoved: backend.reviewIndex = Math.round(value)
            }
            Button { text: "▶"; onClicked: backend.stepReview(1); Accessible.name: "Next decision" }
            Text {
                text: (backend.reviewIndex + 1) + " / " + backend.reviewCount + "   t = " + backend.timeS.toFixed(1) + " s"
                color: Theme.text
                font.pixelSize: Theme.textBody
            }
        }

        GridLayout {
            id: timeline
            Layout.fillWidth: true
            Layout.fillHeight: true
            columns: page.layoutMode === "wide" ? 2 : 1
            columnSpacing: Theme.spaceL
            rowSpacing: Theme.spaceM
            // From the page, never from this layout's own width (that would loop).
            readonly property real available: page.width - 2 * Theme.spaceL
            readonly property real chartsWidth: columns === 2 ? (available - columnSpacing) * 0.6 : available
            readonly property real listWidth: columns === 2 ? available - columnSpacing - chartsWidth : available

            ColumnLayout {
                Layout.preferredWidth: timeline.chartsWidth
                Layout.maximumWidth: timeline.chartsWidth
                Layout.fillHeight: true
                spacing: Theme.spaceM
                TimeSeries {
                    objectName: "tofChart"
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.minimumHeight: 150
                    store: backend.series
                    title: "Nearest ToF return"
                    channels: ["tof_left_cm", "tof_right_cm"]
                    labels: ["left", "right"]
                    colors: [Theme.seriesLeft, Theme.seriesRight]
                    holdChannel: "hold"
                    yMin: 0; yMax: 40
                    limit: backend.clearanceM * 100
                    limitLabel: "stop line 6 cm"
                    unit: "cm"
                    windowS: backend.reviewMode ? 0 : 30
                    cursorS: backend.cursorS
                }
                TimeSeries {
                    objectName: "mouthChart"
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.minimumHeight: 150
                    store: backend.series
                    title: "Mouth to target"
                    channels: ["mouth_mm"]
                    labels: ["mouth"]
                    colors: [Theme.seriesMouth]
                    holdChannel: "hold"
                    yMin: 0; yMax: 120
                    limit: backend.mouthToleranceM * 1000
                    limitLabel: "close within 8 mm"
                    unit: "mm"
                    windowS: backend.reviewMode ? 0 : 30
                    cursorS: backend.cursorS
                }
            }
            ColumnLayout {
                Layout.preferredWidth: timeline.listWidth
                Layout.maximumWidth: timeline.listWidth
                Layout.fillHeight: true
                Layout.minimumHeight: page.compact ? 220 : 0
                spacing: Theme.spaceS
                RowLayout {
                    Layout.fillWidth: true
                    Text {
                        text: "Decisions  " + backend.decisionCount + " (holds " + backend.holdCount + ")"
                        color: Theme.text
                        font.pixelSize: Theme.textBody
                        Layout.fillWidth: true
                    }
                    CheckBox {
                        objectName: "holdsOnly"
                        text: "holds only"
                        checked: backend.decisions.holdsOnly
                        onToggled: backend.decisions.holdsOnly = checked
                    }
                }
                DecisionList { Layout.fillWidth: true; Layout.fillHeight: true }
            }
        }
    }
}
