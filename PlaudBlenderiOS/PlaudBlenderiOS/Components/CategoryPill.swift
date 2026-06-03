import SwiftUI

/// Colored pill showing event category.
struct CategoryPill: View {
    let category: String

    var body: some View {
        Text(category.capitalized.replacingOccurrences(of: "_", with: " "))
            .font(.caption2)
            .fontWeight(.medium)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(Color.forCategory(category).opacity(0.2))
            .foregroundStyle(Color.forCategory(category))
            .clipShape(Capsule())
            .accessibilityLabel("Category: \(category.capitalized)")
            .accessibilityLabel("Category: \(category.capitalized)")
    }
}

#Preview {
    HStack {
        CategoryPill(category: "work")
        CategoryPill(category: "meeting")
        CategoryPill(category: "personal")
        CategoryPill(category: "deep_work")
    }
    .padding()
}
