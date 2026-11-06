// A titled surface that groups related content.
import QtQuick 2.15
import "."

Rectangle {
    id: card
    property string title: ""
    default property alias content: body.data
    color: Theme.surface
    radius: Theme.radiusM
    border.color: Theme.border
    implicitHeight: header.height + body.childrenRect.height + 3 * Theme.spaceM
    Text {
        id: header
        x: Theme.spaceM
        y: Theme.spaceM
        text: card.title
        color: Theme.textMuted
        font.pixelSize: Theme.textCaption
        font.capitalization: Font.AllUppercase
        font.letterSpacing: 1
    }
    Item {
        id: body
        x: Theme.spaceM
        y: header.y + header.height + Theme.spaceS
        width: card.width - 2 * Theme.spaceM
        height: childrenRect.height
    }
}
