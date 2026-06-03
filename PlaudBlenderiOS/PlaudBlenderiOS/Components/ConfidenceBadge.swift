import SwiftUI

/// Badge showing AI classification confidence level.
struct ConfidenceBadge: View {
    let confidence: Double?

    private var level: String {
        guard let confidence else { return "?" }
        switch confidence {
        case 0.8...: return "high"
        case 0.5..<0.8: return "med"
        default: return "low"
        }
    }

    private var color: Color {
        switch level {
        case "high": return .green
        case "med": return .yellow
        default: return .red
        }
    }

    var body: some View {
        Text(level)
            .font(.caption2)
            .fontWeight(.medium)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.2))
            .foregroundStyle(color)
            .clipShape(Capsule())
            .accessibilityLabel("Confidence: \(level)")
    }
}

#Preview {
    HStack {
        ConfidenceBadge(confidence: 0.95)
        ConfidenceBadge(confidence: 0.6)
        ConfidenceBadge(confidence: 0.3)
        ConfidenceBadge(confidence: nil)
    }
    .padding()
}
