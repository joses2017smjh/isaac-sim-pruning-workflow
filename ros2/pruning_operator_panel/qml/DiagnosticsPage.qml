// Is the panel itself trustworthy right now? Message rates, ages and latency.
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "."

Flickable {
    id: page
    property string layoutMode: "wide"
    contentWidth: width
    contentHeight: column.implicitHeight + 2 * Theme.spaceL
    clip: true
    ScrollBar.vertical: ScrollBar {}

    ColumnLayout {
        id: column
        x: Theme.spaceL
        y: Theme.spaceL
        width: page.width - 2 * Theme.spaceL
        spacing: Theme.spaceL

        Flow {
            Layout.fillWidth: true
            spacing: Theme.spaceS
            MetricTile {
                label: "last gate decision"
                value: isNaN(backend.decisionAgeS) ? "n/a" : backend.decisionAgeS.toFixed(1)
                unit: "s ago"
                tone: backend.stale ? "#e3b341" : Theme.text
            }
            MetricTile { label: "stale after"; value: backend.staleAfterS.toFixed(0); unit: "s" }
            MetricTile {
                label: "gate to panel, p50"
                value: isNaN(backend.latencyP50Ms) ? "n/a" : backend.latencyP50Ms.toFixed(1)
                unit: "ms"
            }
            MetricTile {
                label: "gate to panel, p95"
                value: isNaN(backend.latencyP95Ms) ? "n/a" : backend.latencyP95Ms.toFixed(1)
                unit: "ms"
            }
            MetricTile {
                label: "missed decisions"
                value: String(backend.missedDecisions)
                tone: backend.missedDecisions > 0 ? "#ff7b72" : Theme.text
            }
            MetricTile { label: "repaints"; value: backend.renderFps.toFixed(0); unit: "/s" }
        }

        SectionCard {
            title: "Topics"
            Layout.fillWidth: true
            Column {
                width: parent.width
                spacing: 2
                Row {
                    spacing: Theme.spaceM
                    Repeater {
                        model: ["topic", "msgs", "Hz", "last (s)", "pubs"]
                        Text {
                            text: modelData
                            color: Theme.textMuted
                            font.pixelSize: Theme.textCaption
                            width: index === 0 ? Math.max(120, page.width - 420) : 60
                        }
                    }
                }
                Repeater {
                    objectName: "topicRows"
                    model: backend.topicStats
                    Row {
                        spacing: Theme.spaceM
                        Text { text: modelData.topic; color: Theme.text; font.pixelSize: Theme.textCaption; elide: Text.ElideMiddle; width: Math.max(120, page.width - 420) }
                        Text { text: modelData.count; color: Theme.text; font.pixelSize: Theme.textCaption; width: 60 }
                        Text { text: modelData.hz.toFixed(1); color: Theme.text; font.pixelSize: Theme.textCaption; width: 60 }
                        Text { text: isNaN(modelData.ageS) ? "n/a" : modelData.ageS.toFixed(1); color: Theme.text; font.pixelSize: Theme.textCaption; width: 60 }
                        Text { text: modelData.publishers === undefined ? "n/a" : modelData.publishers; color: modelData.publishers === 0 ? "#ff7b72" : Theme.text; font.pixelSize: Theme.textCaption; width: 60 }
                    }
                }
                Text {
                    visible: backend.topicStats.length === 0
                    text: backend.reviewMode ? "review mode: no live topics" : "no messages yet"
                    color: Theme.textMuted
                    font.pixelSize: Theme.textCaption
                }
            }
        }
    }
}
