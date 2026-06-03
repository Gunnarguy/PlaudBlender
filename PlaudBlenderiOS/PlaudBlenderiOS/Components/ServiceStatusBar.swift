import SwiftUI

/// Compact horizontal bar showing integration statuses at a glance.
/// Tappable to expand into a detail view with per-service info.
struct ServiceStatusBar: View {
    let systemStatus: SystemStatus?
    let isLoading: Bool
    var onRefresh: (() async -> Void)?

    @State private var isExpanded = false

    var body: some View {
        VStack(spacing: 0) {
            // Compact bar — always visible
            Button {
                withAnimation(.easeInOut(duration: 0.2)) { isExpanded.toggle() }
            } label: {
                compactBar
            }
            .buttonStyle(.plain)

            // Expanded detail
            if isExpanded, let sys = systemStatus {
                expandedDetail(sys)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
    }

    private var compactBar: some View {
        HStack(spacing: 8) {
            if isLoading {
                ProgressView()
                    .controlSize(.mini)
                Text("Checking services…")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } else if let sys = systemStatus {
                // Service dots
                HStack(spacing: 5) {
                    serviceDot("DB", ok: sys.database?.ok == true)
                    serviceDot("Qdrant", ok: sys.qdrant?.ok == true)
                    serviceDot(geminiTitle(sys.gemini), ok: sys.gemini?.isSafeOrOperational == true)
                    serviceDot(openAITitle(sys.openai), ok: sys.openai?.isSafeOrOperational == true)
                    serviceDot("Plaud", ok: sys.plaud?.isUp == true)
                    serviceDot("Notion", ok: sys.notion?.isUp == true)
                    serviceDot("Local", ok: sys.localLLM?.isUp == true || sys.localLLM?.enabled != true)
                }

                Spacer()

                // Summary
                let healthy = healthyCount(sys)
                let total = serviceTotal
                Text("\(healthy)/\(total)")
                    .font(.caption2.weight(.bold).monospacedDigit())
                    .foregroundStyle(healthy == total ? .green : (healthy >= max(3, total / 2) ? .yellow : .red))

                Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            } else {
                Image(systemName: "questionmark.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text("No status")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Spacer()
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    private func expandedDetail(_ sys: SystemStatus) -> some View {
        VStack(spacing: 6) {
            Divider().padding(.horizontal, 12)

            serviceRow("Database", icon: "cylinder", ok: sys.database?.ok == true,
                       detail: sys.database?.error ?? sys.database?.url)
            serviceRow("Qdrant", icon: "circle.hexagongrid", ok: sys.qdrant?.ok == true,
                       detail: sys.qdrant?.error ?? qdrantDetail(sys.qdrant))
            serviceRow(geminiTitle(sys.gemini), icon: "sparkle", ok: sys.gemini?.isSafeOrOperational == true,
                       detail: geminiDetail(sys.gemini))
            serviceRow(openAITitle(sys.openai), icon: "brain.head.profile", ok: sys.openai?.isSafeOrOperational == true,
                       detail: sys.openai?.error ?? sys.openai?.detail)
            serviceRow("Plaud", icon: "mic.fill", ok: sys.plaud?.isUp == true,
                       detail: sys.plaud?.error ?? (sys.plaud?.isUp == true ? "Authenticated" : "Not connected"))
            serviceRow("Notion", icon: "doc.text", ok: sys.notion?.isUp == true,
                       detail: notionDetail(sys.notion))
            serviceRow("Local LLM", icon: "cpu", ok: sys.localLLM?.isUp == true || sys.localLLM?.enabled != true,
                       detail: localLLMDetail(sys.localLLM))

            if let onRefresh {
                Button {
                    Task { await onRefresh() }
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                        .font(.caption2)
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
                .padding(.top, 4)
            }
        }
        .padding(.bottom, 8)
    }

    private func serviceDot(_ name: String, ok: Bool) -> some View {
        VStack(spacing: 2) {
            Circle()
                .fill(ok ? .green : .red)
                .frame(width: 8, height: 8)
            Text(name)
                .font(.system(size: 8, weight: .medium))
                .foregroundStyle(.secondary)
        }
    }

    private func serviceRow(_ name: String, icon: String, ok: Bool, detail: String?) -> some View {
        HStack(spacing: 8) {
            Image(systemName: ok ? "checkmark.circle.fill" : "xmark.circle.fill")
                .font(.caption)
                .foregroundStyle(ok ? .green : .red)

            Image(systemName: icon)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 16)

            Text(name)
                .font(.caption.weight(.medium))

            Spacer()

            if let detail, !detail.isEmpty {
                Text(detail)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            } else {
                Text(ok ? "OK" : "Down")
                    .font(.caption2)
                    .foregroundStyle(ok ? .green : .red)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 3)
    }

    private func qdrantDetail(_ q: SystemStatus.QdrantCheck?) -> String? {
        guard let q else { return nil }
        if let c = q.collections {
            return "\(c) collection\(c == 1 ? "" : "s")"
        }
        return q.url
    }

    private func healthyCount(_ sys: SystemStatus) -> Int {
        var count = 0
        if sys.database?.ok == true { count += 1 }
        if sys.qdrant?.ok == true { count += 1 }
        if sys.gemini?.isSafeOrOperational == true { count += 1 }
        if sys.openai?.isSafeOrOperational == true { count += 1 }
        if sys.plaud?.isUp == true { count += 1 }
        if sys.notion?.isUp == true { count += 1 }
        if sys.localLLM?.isUp == true || sys.localLLM?.enabled != true { count += 1 }
        return count
    }

    private var serviceTotal: Int {
        // Local LLM is always shown as an explicit disabled/ready/unavailable row.
        7
    }

    private func notionDetail(_ notion: SystemStatus.ServiceFlag?) -> String? {
        guard let notion else { return nil }
        if let error = notion.error, !error.isEmpty {
            return error
        }
        if let workspaceName = notion.workspaceName, !workspaceName.isEmpty {
            return workspaceName
        }
        if notion.isAuthenticated == true {
            return "Connected"
        }
        if notion.hasCredentials == true {
            return "Integration token configured"
        }
        if notion.hasAccessToken == true {
            return "OAuth token present"
        }
        return notion.isUp ? "Configured" : "Not connected"
    }

    private func localLLMDetail(_ local: SystemStatus.LocalLLMCheck?) -> String? {
        guard let local else { return "Not reported" }
        if local.enabled != true {
            return "Disabled"
        }
        if let error = local.error, !error.isEmpty {
            return error
        }
        if let detail = local.detail, !detail.isEmpty {
            return detail
        }
        let pieces: [String] = [local.provider, local.model].compactMap { value -> String? in
            guard let value, !value.isEmpty else { return nil }
            return value
        }
        return pieces.isEmpty ? (local.ok == true ? "Ready" : "Unavailable") : pieces.joined(separator: " · ")
    }

    private func openAITitle(_ openAI: SystemStatus.ServiceCheck?) -> String {
        openAI?.enabled == false ? "OpenAI Off" : "OpenAI"
    }

    private func geminiTitle(_ gemini: SystemStatus.ServiceFlag?) -> String {
        gemini?.enabled == false ? "Gemini Off" : "Gemini"
    }

    private func geminiDetail(_ gemini: SystemStatus.ServiceFlag?) -> String? {
        guard let gemini else { return "Not reported" }
        if let error = gemini.error, !error.isEmpty {
            return error
        }
        if let detail = gemini.detail, !detail.isEmpty {
            return detail
        }
        if gemini.enabled == false {
            return "Disabled by local/Ollama routing"
        }
        return gemini.isUp ? "Configured" : "Not configured"
    }
}
