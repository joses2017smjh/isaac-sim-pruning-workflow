// A labelled number with a unit; tone highlights values that need attention.
import QtQuick 2.15
import "."

Rectangle {
    id: tile
    property string label: ""
    property string value: "n/a"
    property string unit: ""
    property color tone: Theme.text
    radius: Theme.radiusS
    color: Theme.surfaceRaised
    border.color: Theme.border
    implicitWidth: 150
    implicitHeight: 64
    Column {
        anchors.fill: parent
        anchors.margins: Theme.spaceS
        spacing: 2
        Text { text: tile.label; color: Theme.textMuted; font.pixelSize: Theme.textCaption; elide: Text.ElideRight; width: parent.width }
        Row {
            spacing: Theme.spaceXs
            Text { text: tile.value; color: tile.tone; font.pixelSize: Theme.textHeadline; font.bold: true }
            Text { text: tile.unit; color: Theme.textMuted; font.pixelSize: Theme.textBody; topPadding: 6 }
        }
    }
}
