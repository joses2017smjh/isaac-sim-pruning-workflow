// Operator panel: what the safety gate decided and why, how the run is going,
// whether the panel itself is keeping up, and a way to stop the arm.
//
// Layout modes follow Theme breakpoints: compact (< 760 px), medium, wide (>= 1180 px).
// The Hold button lives in the footer, outside every scroll area, at every size.
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import "."

ApplicationWindow {
    id: root
    width: 1280
    height: 760
    minimumWidth: 360
    minimumHeight: 560
    visible: true
    color: Theme.background
    title: "Pruning operator panel (simulation)"
    // Dark palette for the stock Controls (tabs, check boxes, sliders).
    palette.window: Theme.surface
    palette.windowText: Theme.text
    palette.base: Theme.surfaceRaised
    palette.text: Theme.text
    palette.button: Theme.surfaceRaised
    palette.buttonText: Theme.text
    palette.highlight: Theme.accent
    palette.mid: Theme.border
    palette.dark: Theme.border
    palette.light: Theme.surfaceRaised
    palette.midlight: Theme.surfaceRaised

    readonly property string layoutMode: width < Theme.compactBelow ? "compact"
                                         : width >= Theme.wideFrom ? "wide" : "medium"
    property alias currentTab: tabs.currentIndex

    Shortcut { sequence: "Space"; enabled: !backend.reviewMode; onActivated: backend.requestHold("keyboard") }
    Shortcut { sequence: "Left"; enabled: backend.reviewMode; onActivated: backend.stepReview(-1) }
    Shortcut { sequence: "Right"; enabled: backend.reviewMode; onActivated: backend.stepReview(1) }
    Shortcut { sequence: "Ctrl+1"; onActivated: tabs.currentIndex = 0 }
    Shortcut { sequence: "Ctrl+2"; onActivated: tabs.currentIndex = 1 }
    Shortcut { sequence: "Ctrl+3"; onActivated: tabs.currentIndex = 2 }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // Header: identity, mode and health at a glance.
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: 48
            color: Theme.surface
            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: Theme.spaceL
                anchors.rightMargin: Theme.spaceL
                spacing: Theme.spaceS
                Text {
                    text: root.layoutMode === "compact" ? "Pruning panel" : "Pruning operator panel"
                    color: Theme.text
                    font.pixelSize: Theme.textTitle
                    font.bold: true
                    Layout.fillWidth: true
                    elide: Text.ElideRight
                }
                Repeater {
                    model: [
                        { label: backend.reviewMode ? "REVIEW " + backend.reviewRun : "LIVE",
                          tone: backend.reviewMode ? Theme.review : Theme.pass, show: true },
                        { label: "STALE", tone: Theme.stale, show: backend.stale },
                        { label: "HOLD LATCHED", tone: Theme.hold, show: backend.latchedReason !== "" },
                        { label: "SIMULATION", tone: Theme.waiting, show: root.layoutMode !== "compact" }
                    ]
                    Rectangle {
                        visible: modelData.show
                        radius: Theme.radiusS
                        color: modelData.tone
                        implicitHeight: 24
                        implicitWidth: Math.min(chip.implicitWidth + 2 * Theme.spaceS, root.width * 0.4)
                        Text {
                            id: chip
                            anchors.centerIn: parent
                            width: Math.min(implicitWidth, root.width * 0.4 - 2 * Theme.spaceS)
                            elide: Text.ElideMiddle
                            text: modelData.label
                            color: "white"
                            font.pixelSize: Theme.textCaption
                            font.bold: true
                        }
                    }
                }
            }
        }

        TabBar {
            id: tabs
            objectName: "tabs"
            Layout.fillWidth: true
            TabButton { text: "Live" }
            TabButton { text: "Timeline" }
            TabButton { text: "Diagnostics" }
        }

        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: tabs.currentIndex
            LivePage { objectName: "livePage"; layoutMode: root.layoutMode }
            TimelinePage { objectName: "timelinePage"; layoutMode: root.layoutMode }
            DiagnosticsPage { objectName: "diagnosticsPage"; layoutMode: root.layoutMode }
        }

        // Footer: the safety control, always visible.
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: holdButton.implicitHeight + 2 * Theme.spaceM
            color: Theme.surface
            RowLayout {
                anchors.fill: parent
                anchors.margins: Theme.spaceM
                spacing: Theme.spaceM
                Text {
                    visible: root.layoutMode !== "compact"
                    text: "Simulated UR5e in Isaac Sim. Not connected to hardware. A hold latches for the run."
                    color: Theme.textMuted
                    font.pixelSize: Theme.textCaption
                    wrapMode: Text.WordWrap
                    Layout.preferredWidth: 260
                }
                HoldButton { id: holdButton; Layout.fillWidth: true }
            }
        }
    }
}
