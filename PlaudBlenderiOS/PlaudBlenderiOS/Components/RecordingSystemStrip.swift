import SwiftUI

struct RecordingSystemStrip: View {
    let summary: RecordingSummary
    var workflowStatus: [String: JSONValue]? = nil
    var detail: Bool = false

    private struct Badge: Identifiable {
        let id: String
        let label: String
        let tone: Tone
        let title: String?
    }

    private enum Tone {
        case success
        case info
        case warning
        case danger
        case muted
        case notion

        var foreground: Color {
            switch self {
            case .success:
                return .accentGreen
            case .info:
                return .accentPrimary
            case .warning:
                return .accentOrange
            case .danger:
                return .accentRed
            case .muted:
                return .secondary
            case .notion:
                return .accentCyan
            }
        }

        var background: Color {
            switch self {
            case .success:
                return .accentGreen.opacity(0.14)
            case .info:
                return .accentPrimary.opacity(0.14)
            case .warning:
                return .accentOrange.opacity(0.14)
            case .danger:
                return .accentRed.opacity(0.14)
            case .muted:
                return Color.secondary.opacity(0.12)
            case .notion:
                return .accentCyan.opacity(0.14)
            }
        }
    }

    private var badges: [Badge] {
        var result: [Badge] = [processingBadge]

        if let plaudBadge {
            result.append(plaudBadge)
        }

        if let notionBadge {
            result.append(notionBadge)
        }

        return result
    }

    var body: some View {
        if !badges.isEmpty || (detail && notionURL != nil) {
            VStack(alignment: .leading, spacing: 8) {
                if !badges.isEmpty {
                    ViewThatFits(in: .horizontal) {
                        HStack(spacing: 6) {
                            badgeRow
                        }

                        VStack(alignment: .leading, spacing: 6) {
                            badgeRow
                        }
                    }
                }

                if detail, let notionURL, let url = URL(string: notionURL) {
                    Link(destination: url) {
                        Label("Open Notion", systemImage: "arrow.up.right.square")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(Color.accentCyan)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var badgeRow: some View {
        ForEach(badges) { badge in
            Text(badge.label)
                .font(.caption2.weight(.semibold))
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .foregroundStyle(badge.tone.foreground)
                .background(badge.tone.background)
                .clipShape(Capsule())
                .help(badge.title ?? badge.label)
                .accessibilityLabel(badge.title ?? badge.label)
        }
    }

    private var processingBadge: Badge {
        let status = normalizedProcessingStatus
        switch status {
        case "completed":
            return Badge(
                id: "processing",
                label: "Sync ready",
                tone: .success,
                title: "Chronos extraction finished and this recording is available in the app."
            )
        case "processing":
            return Badge(
                id: "processing",
                label: "Sync processing",
                tone: .info,
                title: "Chronos is currently extracting and indexing this recording."
            )
        case "pending":
            return Badge(
                id: "processing",
                label: "Sync pending",
                tone: .warning,
                title: "Chronos has this recording queued but has not extracted moments yet."
            )
        case "failed":
            return Badge(
                id: "processing",
                label: "Sync failed",
                tone: .danger,
                title: "Chronos hit an error while processing this recording."
            )
        default:
            return Badge(
                id: "processing",
                label: "Sync unknown",
                tone: .muted,
                title: "Chronos could not determine the processing state for this recording."
            )
        }
    }

    private var plaudBadge: Badge? {
        let source = (summary.source ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let workflow = normalizedWorkflowStatus
        let hasPlaudAI = summary.hasPlaudAI == true
            || !(summary.plaudAiSummary?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)
            || summary.cloudStatus?.lowercased() == "ai"
        let expectsPlaudAI = source == "plaud_cloud" || !workflow.isEmpty

        if ["PENDING", "PROCESSING", "RUNNING"].contains(workflow) {
            let templateID = workflowStatus?["template_id"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines)
            let detail = (templateID?.isEmpty == false)
                ? "Plaud cloud workflow is still running (\(templateID!))."
                : "Plaud cloud workflow is still running for this recording."
            return Badge(id: "plaud", label: "Plaud AI running", tone: .info, title: detail)
        }

        if ["FAILED", "ERROR"].contains(workflow) {
            let errorDetail = workflowStatus?["error"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines)
            return Badge(
                id: "plaud",
                label: "Plaud AI failed",
                tone: .danger,
                title: (errorDetail?.isEmpty == false)
                    ? errorDetail
                    : "Plaud workflow failed for this recording."
            )
        }

        if ["SUCCESS", "COMPLETED"].contains(workflow) {
            if hasPlaudAI {
                return Badge(
                    id: "plaud",
                    label: "Plaud AI ready",
                    tone: .success,
                    title: "Plaud cloud summary or transcript data is available for this recording."
                )
            }

            return Badge(
                id: "plaud",
                label: "Plaud workflow done",
                tone: .success,
                title: "Plaud cloud finished processing this recording."
            )
        }

        if hasPlaudAI {
            return Badge(
                id: "plaud",
                label: "Plaud AI ready",
                tone: .success,
                title: "Plaud cloud summary or transcript data is available for this recording."
            )
        }

        if expectsPlaudAI {
            return Badge(
                id: "plaud",
                label: "Plaud AI missing",
                tone: .muted,
                title: "This Plaud cloud recording does not yet have Plaud AI output attached."
            )
        }

        return nil
    }

    private var notionBadge: Badge? {
        let state = (summary.notionState ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !state.isEmpty else {
            return nil
        }

        let title = summary.notionPageTitle?.trimmingCharacters(in: .whitespacesAndNewlines)
        let matchCount = summary.notionMatchCount ?? 0
        let countSuffix = matchCount > 1 ? " x\(matchCount)" : ""

        switch state {
        case "imported":
            return Badge(
                id: "notion",
                label: "Notion imported",
                tone: .notion,
                title: (title?.isEmpty == false) ? title : "This recording originated from a Notion page."
            )
        case "imported-stale":
            return Badge(
                id: "notion",
                label: "Notion newer",
                tone: .warning,
                title: (title?.isEmpty == false) ? title : "The source Notion page changed after this recording was imported."
            )
        case "linked":
            return Badge(
                id: "notion",
                label: "Notion linked\(countSuffix)",
                tone: .notion,
                title: (title?.isEmpty == false) ? title : "This recording is linked to a Notion page through the Chronos bridge."
            )
        case "stale":
            return Badge(
                id: "notion",
                label: "Notion newer\(countSuffix)",
                tone: .warning,
                title: (title?.isEmpty == false) ? title : "A linked Notion page has been edited since Chronos last synced it."
            )
        case "chronos-only":
            return Badge(
                id: "notion",
                label: "Ready for Notion",
                tone: .info,
                title: "This Chronos recording is not linked to Notion yet and can be pushed there."
            )
        default:
            return nil
        }
    }

    private var normalizedProcessingStatus: String {
        let raw = summary.processingStatus?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if let raw, !raw.isEmpty {
            return raw
        }
        return "completed"
    }

    private var normalizedWorkflowStatus: String {
        if let candidate = workflowStatus?["status"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines).uppercased(),
           !candidate.isEmpty {
            return candidate
        }
        if let candidate = summary.plaudWorkflowStatus?.trimmingCharacters(in: .whitespacesAndNewlines).uppercased(),
           !candidate.isEmpty {
            return candidate
        }
        return ""
    }

    private var notionURL: String? {
        let trimmed = summary.notionPageURL?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let trimmed, !trimmed.isEmpty else {
            return nil
        }
        return trimmed
    }
}

#Preview {
    let summary = RecordingSummary(
        recordingId: "rec-preview",
        startTime: "2026-05-13T10:00:00Z",
        createdAt: nil,
        endTime: "2026-05-13T10:20:00Z",
        durationSeconds: 1200,
        durationFormatted: "20m",
        topCategory: "work",
        eventCount: 8,
        timeRangeFormatted: "10:00 AM - 10:20 AM",
        timeIsEstimated: false,
        timeEstimateReason: nil,
        title: "Preview",
        plaudAiSummary: "Quick cloud summary",
        cloudStatus: "ai",
        source: "plaud_cloud",
        processingStatus: "completed",
        hasPlaudAI: true,
        plaudWorkflowStatus: "SUCCESS",
        notionState: "linked",
        notionPageId: "page-1",
        notionPageURL: "https://www.notion.so/example",
        notionPageTitle: "Daily Notes",
        notionMatchCount: 2
    )

    VStack(alignment: .leading, spacing: 12) {
        RecordingSystemStrip(summary: summary)
        RecordingSystemStrip(summary: summary, workflowStatus: ["status": .string("PROCESSING")], detail: true)
    }
    .padding()
}
