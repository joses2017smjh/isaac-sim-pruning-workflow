// What the gate decided now, what the camera and both ToF sensors see.
// Two columns from the medium breakpoint up, one scrolling column when compact.
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import Pruning.Panel 1.0
import "."

Flickable {
    id: page
    property string layoutMode: "wide"
    readonly property bool compact: layoutMode === "compact"
    contentWidth: width
    contentHeight: grid.implicitHeight + 2 * Theme.spaceL
    clip: true
    ScrollBar.vertical: ScrollBar {}

    GridLayout {
        id: grid
        x: Theme.spaceL
        y: Theme.spaceL
        width: page.width - 2 * Theme.spaceL
        columns: page.compact ? 1 : 2
        columnSpacing: Theme.spaceL
        rowSpacing: Theme.spaceL
        // Explicit column widths: the camera column takes 48 %, the status column the rest.
        readonly property real colA: page.compact ? width : (width - columnSpacing) * 0.48
        readonly property real colB: page.compact ? width : width - columnSpacing - colA

        // Compact puts the call first; wide puts the camera on the left.
        StatusBanner {
            Layout.preferredWidth: grid.colB
            Layout.maximumWidth: grid.colB
            Layout.columnSpan: 1
            Layout.row: 0
            Layout.column: page.compact ? 0 : 1
            compact: page.compact
        }
        SectionCard {
            objectName: "cameraCard"
            title: "Wrist camera"
            Layout.preferredWidth: grid.colA
            Layout.maximumWidth: grid.colA
            Layout.row: page.compact ? 1 : 0
            Layout.column: 0
            Layout.rowSpan: page.compact ? 1 : 2
            Rectangle {
                // Sized from the page's column width, not the laid-out card, so height never feeds back.
                objectName: "cameraView"
                width: grid.colA - 2 * Theme.spaceM
                height: width / 1.5
                color: "black"
                Image {
                    anchors.fill: parent
                    cache: false
                    fillMode: Image.PreserveAspectFit
                    source: backend.frameCounter > 0 ? "image://wrist/" + backend.frameCounter : ""
                }
                Text {
                    anchors.centerIn: parent
                    visible: backend.frameCounter === 0
                    text: "waiting for /pruning/wrist/image_raw"
                    color: Theme.textMuted
                }
            }
        }
        SectionCard {
            title: "Time-of-flight zones (cm), 6 cm stop line"
            Layout.preferredWidth: grid.colB
            Layout.maximumWidth: grid.colB
            Layout.row: page.compact ? 2 : 1
            Layout.column: page.compact ? 0 : 1
            Row {
                id: grids
                spacing: Theme.spaceM
                readonly property real side: Math.min(220, (grid.colB - 2 * Theme.spaceM - spacing) / 2)
                ZoneGrid {
                    objectName: "zoneGridLeft"
                    width: grids.side; height: grids.side + 18
                    title: "left"
                    zones: backend.tofLeftZones
                    clearance: backend.clearanceM
                }
                ZoneGrid {
                    objectName: "zoneGridRight"
                    width: grids.side; height: grids.side + 18
                    title: "right"
                    zones: backend.tofRightZones
                    clearance: backend.clearanceM
                }
            }
        }
        SectionCard {
            title: "Nearest returns"
            Layout.preferredWidth: grid.colB
            Layout.maximumWidth: grid.colB
            Layout.row: page.compact ? 3 : 2
            Layout.column: page.compact ? 0 : 1
            Column {
                width: parent.width
                spacing: Theme.spaceS
                RangeBar { width: parent.width; label: "ToF left"; value: backend.tofLeftM; stopLine: backend.clearanceM }
                RangeBar { width: parent.width; label: "ToF right"; value: backend.tofRightM; stopLine: backend.clearanceM }
            }
        }
        Grid {
            // Two fixed columns of tiles: a wrapping Flow would make height depend on width.
            id: tiles
            Layout.preferredWidth: grid.colA
            Layout.maximumWidth: grid.colA
            Layout.row: page.compact ? 4 : 2
            Layout.column: 0
            columns: 2
            spacing: Theme.spaceS
            readonly property real tileWidth: (grid.colA - spacing) / 2
            MetricTile { width: tiles.tileWidth; label: "cycle"; value: String(backend.cycle) }
            MetricTile {
                width: tiles.tileWidth
                label: "mouth to target"
                value: isNaN(backend.mouthDistanceM) ? "n/a" : (backend.mouthDistanceM * 1000).toFixed(1)
                unit: "mm"
                tone: backend.mouthDistanceM <= backend.mouthToleranceM ? "#56d364" : Theme.text
            }
            MetricTile { width: tiles.tileWidth; label: "message delay (modelled)"; value: backend.delayMs.toFixed(1); unit: "ms" }
            MetricTile {
                width: tiles.tileWidth
                label: "latched for this run"
                value: backend.latchedReason === "" ? "none" : backend.latchedReason
                tone: backend.latchedReason === "" ? Theme.text : "#ff7b72"
            }
        }
    }
}
