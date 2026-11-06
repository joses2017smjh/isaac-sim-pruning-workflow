// Every gate decision, newest last, from the C++ DecisionLogModel through HoldFilterModel.
import QtQuick 2.15
import QtQuick.Controls 2.15
import "."

ListView {
    id: list
    objectName: "decisionList"
    clip: true
    model: backend.decisions
    spacing: 2
    ScrollBar.vertical: ScrollBar {}
    onCountChanged: if (!backend.reviewMode) positionViewAtEnd()
    delegate: Rectangle {
        width: list.width - 10
        height: 30
        radius: 4
        color: model.passed ? Theme.surfaceRaised : "#3b1714"
        border.color: backend.reviewMode && Math.abs(model.timeS - backend.timeS) < 1e-6 ? "white" : "transparent"
        Row {
            anchors.verticalCenter: parent.verticalCenter
            x: Theme.spaceS
            spacing: Theme.spaceM
            Text { text: "#" + model.cycle; color: Theme.textMuted; font.pixelSize: Theme.textCaption; width: 44 }
            Text { text: model.timeS.toFixed(1) + " s"; color: Theme.textMuted; font.pixelSize: Theme.textCaption; width: 52 }
            Text {
                text: model.passed ? "PASS" : "HOLD"
                color: model.passed ? "#56d364" : "#ff7b72"
                font.bold: true
                font.pixelSize: Theme.textBody
                width: 48
            }
            Text {
                text: model.passed ? (isNaN(model.mouthMm) ? "" : "mouth " + model.mouthMm.toFixed(1) + " mm") : model.reason
                color: Theme.text
                font.pixelSize: Theme.textCaption
                elide: Text.ElideRight
                width: list.width - 230
            }
        }
        MouseArea {
            anchors.fill: parent
            enabled: backend.reviewMode
            onClicked: backend.seekReviewTime(model.timeS)
        }
    }
}
