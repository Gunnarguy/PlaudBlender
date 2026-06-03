import SwiftUI

extension Color {
    // MARK: - Category Colors (from CSS --cat-* variables)
    static let catWork = Color(hex: "0969da")
    static let catPersonal = Color(hex: "8250df")
    static let catMeeting = Color(hex: "1a7f37")
    static let catReflection = Color(hex: "9a6700")
    static let catIdea = Color(hex: "bf3989")
    static let catBreak = Color(hex: "6e7781")
    static let catDeepWork = Color(hex: "116329")
    static let catUnknown = Color(hex: "8b949e")

    // MARK: - Accent Colors
    static let accentPrimary = Color(hex: "0969da")
    static let accentGreen = Color(hex: "1a7f37")
    static let accentYellow = Color(hex: "9a6700")
    static let accentRed = Color(hex: "cf222e")
    static let accentPurple = Color(hex: "8250df")
    static let accentOrange = Color(hex: "bc4c00")
    static let accentPink = Color(hex: "bf3989")
    static let accentCyan = Color(hex: "0891b2")

    /// Get category color by name string.
    static func forCategory(_ category: String) -> Color {
        switch category.lowercased().replacingOccurrences(of: "_", with: "") {
        case "work": return .catWork
        case "personal": return .catPersonal
        case "meeting": return .catMeeting
        case "reflection": return .catReflection
        case "idea": return .catIdea
        case "break", "breaktime": return .catBreak
        case "deepwork": return .catDeepWork
        default: return .catUnknown
        }
    }

    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet(charactersIn: "#"))
        guard hex.count == 6 || hex.count == 8,
              hex.allSatisfy({ $0.isHexDigit }) else {
            self.init(white: 0.5) // safe fallback for invalid hex
            return
        }
        let scanner = Scanner(string: hex)
        var rgb: UInt64 = 0
        scanner.scanHexInt64(&rgb)
        self.init(
            red: Double((rgb >> 16) & 0xFF) / 255.0,
            green: Double((rgb >> 8) & 0xFF) / 255.0,
            blue: Double(rgb & 0xFF) / 255.0
        )
    }
}
