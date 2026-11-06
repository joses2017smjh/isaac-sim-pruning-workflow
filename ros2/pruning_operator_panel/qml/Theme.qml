// Design tokens: the one place colours, spacing, type and breakpoints are defined.
// Components read these instead of literals, so a spec change is a one-line edit.
pragma Singleton
import QtQuick 2.15

QtObject {
    // Colour roles
    readonly property color background: "#14171c"
    readonly property color surface: "#1b2027"
    readonly property color surfaceRaised: "#232932"
    readonly property color border: "#30363d"
    readonly property color text: "#e6edf3"
    readonly property color textMuted: "#8b949e"
    readonly property color pass: "#1f7a3a"
    readonly property color hold: "#b3261e"
    readonly property color holdAction: "#d32f2f"
    readonly property color holdActionPressed: "#8e1c14"
    readonly property color stale: "#6e5b00"
    readonly property color waiting: "#3a3f47"
    readonly property color review: "#1f4b8a"
    readonly property color limit: "#ffcc00"
    readonly property color accent: "#58a6ff"
    readonly property color seriesLeft: "#58a6ff"
    readonly property color seriesRight: "#d2a8ff"
    readonly property color seriesMouth: "#3fb950"

    // Spacing scale (px)
    readonly property int spaceXs: 4
    readonly property int spaceS: 8
    readonly property int spaceM: 12
    readonly property int spaceL: 18
    readonly property int spaceXl: 24

    // Shape
    readonly property int radiusS: 6
    readonly property int radiusM: 10

    // Type scale (px)
    readonly property int textCaption: 12
    readonly property int textBody: 14
    readonly property int textTitle: 17
    readonly property int textHeadline: 22
    readonly property int textDisplay: 44
    readonly property int textDisplayCompact: 32

    // Breakpoints (window width, px)
    readonly property int compactBelow: 760
    readonly property int wideFrom: 1180

    // Touch target for the safety control, never smaller than this
    readonly property int holdMinHeight: 64
}
