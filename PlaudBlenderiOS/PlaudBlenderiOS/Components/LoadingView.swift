import SwiftUI

/// Loading spinner with optional message.
struct LoadingView: View {
    var message: String = "Loading..."

    var body: some View {
        VStack(spacing: 12) {
            ProgressView()
                .controlSize(.large)
            Text(message)
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// Empty state placeholder with icon and message.
struct EmptyStateView: View {
    let icon: String
    let title: String
    var message: String?
    var actionTitle: String?
    var action: (() -> Void)?

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: icon)
                .font(.system(size: 48))
                .foregroundStyle(.tertiary)
            Text(title)
                .font(.title3)
                .fontWeight(.semibold)
            if let message {
                Text(message)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 32)
            }
            if let actionTitle, let action {
                Button(actionTitle, action: action)
                    .buttonStyle(.borderedProminent)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// Error banner shown at the top of a view.
struct ErrorBanner: View {
    let message: String
    var onDismiss: (() -> Void)?

    var body: some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.yellow)
            Text(message)
                .font(.caption)
                .lineLimit(2)
            Spacer()
            if let onDismiss {
                Button(action: onDismiss) {
                    Image(systemName: "xmark")
                        .font(.caption2)
                }
            }
        }
        .padding(10)
        .background(.red.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .padding(.horizontal)
    }
}

#Preview {
    VStack(spacing: 20) {
        LoadingView()
        EmptyStateView(
            icon: "waveform",
            title: "No Recordings",
            message: "Connect to your Plaud device to start capturing recordings."
        )
    }
}
