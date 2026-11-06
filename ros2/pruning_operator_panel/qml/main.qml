// Operator panel: what the safety gate decided, why, and a way to stop the arm.
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15

Window {
    id: root
    width: 1040
    height: 560
    visible: true
    color: "#14171c"
    title: "Pruning operator panel (simulation)"

    RowLayout {
        anchors.fill: parent
        anchors.margins: 18
        spacing: 18

        ColumnLayout {
            Layout.preferredWidth: 480
            Layout.alignment: Qt.AlignTop
            spacing: 8
            Text { text: "Wrist camera"; color: "#8b949e"; font.pixelSize: 14 }
            Rectangle {
                Layout.preferredWidth: 480
                Layout.preferredHeight: 320
                color: "black"
                Image {
                    anchors.fill: parent
                    cache: false
                    asynchronous: false
                    fillMode: Image.PreserveAspectFit
                    source: backend.frameCounter > 0 ? "image://wrist/" + backend.frameCounter : ""
                }
                Text {
                    anchors.centerIn: parent
                    visible: backend.frameCounter === 0
                    text: "waiting for /pruning/wrist/image_raw"
                    color: "#8b949e"
                }
            }
            Text {
                text: "cycle " + backend.cycle + "    message delay " + backend.delayMs.toFixed(1) + " ms (modelled)"
                color: "#c9d1d9"
                font.pixelSize: 15
            }
            Text {
                text: "Simulated UR5e in Isaac Sim. Not connected to hardware."
                color: "#8b949e"
                font.pixelSize: 12
            }
        }

        ColumnLayout {
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignTop
            spacing: 14

            Rectangle {
                objectName: "banner"
                Layout.fillWidth: true
                Layout.preferredHeight: 120
                radius: 10
                color: !backend.haveDecision ? "#3a3f47" : (backend.passed ? "#1f7a3a" : "#b3261e")
                Column {
                    anchors.centerIn: parent
                    spacing: 4
                    Text {
                        anchors.horizontalCenter: parent.horizontalCenter
                        text: !backend.haveDecision ? "WAITING" : (backend.passed ? "PASS" : "HOLD")
                        color: "white"
                        font.pixelSize: 48
                        font.bold: true
                    }
                    Text {
                        anchors.horizontalCenter: parent.horizontalCenter
                        text: !backend.haveDecision ? "no safety decision yet"
                              : (backend.passed ? "command forwarded to the arm" : backend.reason)
                        color: "white"
                        font.pixelSize: 17
                    }
                }
            }
            Text {
                visible: backend.latchedReason !== ""
                text: "latched for this run: " + backend.latchedReason
                color: "#ffb4a9"
                font.pixelSize: 14
            }
            RangeBar {
                Layout.fillWidth: true
                label: "ToF left"
                value: backend.tofLeftM
                stopLine: backend.clearanceM
            }
            RangeBar {
                Layout.fillWidth: true
                label: "ToF right"
                value: backend.tofRightM
                stopLine: backend.clearanceM
            }
            Text {
                text: "stop line " + (backend.clearanceM * 100).toFixed(0) + " cm    mouth to target "
                      + (isNaN(backend.mouthDistanceM) ? "n/a" : (backend.mouthDistanceM * 1000).toFixed(1) + " mm")
                color: "#8b949e"
                font.pixelSize: 13
            }
            Button {
                id: holdButton
                objectName: "holdButton"
                Layout.fillWidth: true
                Layout.preferredHeight: 76
                text: backend.holdRequested ? "HOLD SENT" : "HOLD"
                onClicked: backend.requestHold("button")
                contentItem: Text {
                    text: holdButton.text
                    color: "white"
                    font.pixelSize: 30
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                background: Rectangle {
                    radius: 10
                    color: holdButton.down ? "#8e1c14" : (backend.holdRequested ? "#5c1a15" : "#d32f2f")
                    border.color: "#ffcdd2"
                    border.width: 2
                }
            }
        }
    }
}
