import Foundation

// MARK: - Pipeline

/// Available pipeline stages for `POST /api/sync/run`.
enum PipelineStage: String, CaseIterable, Identifiable, Sendable {
    case full
    case backfill
    case ingest
    case process
    case index
    case graph
    case reindex

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .full: "Smart Sync"
        case .backfill: "Full History"
        case .ingest: "Ingest"
        case .process: "Process"
        case .index: "Index"
        case .graph: "Graph"
        case .reindex: "Reindex"
        }
    }

    var icon: String {
        switch self {
        case .full: "arrow.triangle.2.circlepath"
        case .backfill: "clock.arrow.trianglehead.counterclockwise.rotate.90"
        case .ingest: "arrow.down.doc"
        case .process: "gearshape.2"
        case .index: "magnifyingglass"
        case .graph: "point.3.connected.trianglepath.dotted"
        case .reindex: "arrow.clockwise"
        }
    }

    var description: String {
        switch self {
        case .full: "Fetch recent Plaud recordings, then process and index them"
        case .backfill: "Page through the full Plaud library when you explicitly need historical backfill"
        case .ingest: "Fetch new recordings from Plaud"
        case .process: "Run AI models on pending recordings"
        case .index: "Rebuild the Qdrant vector index"
        case .graph: "Rebuild the knowledge graph"
        case .reindex: "Reindex all embeddings in Qdrant"
        }
    }

    var requestStageCandidates: [String] {
        switch self {
        case .backfill:
            ["backfill", "all_history", "full_history"]
        default:
            [rawValue]
        }
    }
}

/// Pipeline run request body.
struct PipelineRunRequest: Codable, Sendable {
    var stage: String = "full"
}

/// Pipeline run response.
struct PipelineRunResponse: Codable, Sendable {
    let status: String
    let message: String
    let runId: String?
    let detail: String?
    let warning: String?
    let warnings: [String]?

    enum CodingKeys: String, CodingKey {
        case status, message, detail, warning, warnings
        case runId = "run_id"
    }
}

struct PipelinePhase: Codable, Sendable, Identifiable {
    let name: String
    let status: String
    let totalItems: Int
    let completedItems: Int
    let currentStep: String
    let currentItem: String
    let elapsedSeconds: Double
    let summary: String
    let error: String
    let warnings: [String]?

    var id: String { name }

    enum CodingKeys: String, CodingKey {
        case name, status, summary, error, warnings
        case totalItems = "total_items"
        case completedItems = "completed_items"
        case currentStep = "current_step"
        case currentItem = "current_item"
        case elapsedSeconds = "elapsed_seconds"
    }
}

/// Pipeline progress from `/api/sync/status`.
struct PipelineStatus: Codable, Sendable {
    let status: String
    let runId: String?
    let currentPhase: String?
    let elapsedSeconds: Double?
    let ageSeconds: Double?
    let trigger: String?
    let syncMode: String?
    let partialSuccess: Bool?
    let message: String?
    let detail: String?
    let warning: String?
    let warnings: [String]?
    let phases: [PipelinePhase]

    var running: Bool {
        status.lowercased() == "running"
    }

    var activePhase: PipelinePhase? {
        if let currentPhase {
            return phases.first(where: { $0.name == currentPhase })
        }
        return phases.first(where: { $0.status == "running" })
    }

    enum CodingKeys: String, CodingKey {
        case status, trigger, phases, message, detail, warning, warnings
        case runId = "run_id"
        case currentPhase = "current_phase"
        case elapsedSeconds = "elapsed_seconds"
        case ageSeconds = "age_seconds"
        case syncMode = "sync_mode"
        case partialSuccess = "partial_success"
    }

    var reliabilityMessages: [String] {
        var collected: [String] = []

        for candidate in [message, detail, warning] {
            guard let candidate,
                  !candidate.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                continue
            }
            collected.append(candidate)
        }

        collected.append(contentsOf: warnings ?? [])

        for phase in phases {
            let phaseMessages = [phase.summary, phase.error]
            for candidate in phaseMessages {
                let trimmed = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    collected.append(trimmed)
                }
            }
            collected.append(contentsOf: phase.warnings ?? [])
        }

        var seen = Set<String>()
        return collected.filter { seen.insert($0).inserted }
    }
}

struct PlaudSyncNotice: Identifiable, Sendable, Equatable {
    enum Kind: String, Sendable {
        case repeatedPage
        case partialProgress
        case rateLimited
        case warning
    }

    let kind: Kind
    let title: String
    let message: String

    var id: String {
        "\(kind.rawValue)|\(title)|\(message)"
    }
}

enum PlaudSyncNoticeBuilder {
    static func build(
        status: PipelineStatus?,
        dbStats: RecordingDbStats?,
        failures: SyncFailureSummary?
    ) -> [PlaudSyncNotice] {
        var notices: [PlaudSyncNotice] = []
        let dbCompleted = dbStats?.completed ?? 0

        let statusMessages = status?.reliabilityMessages ?? []
        for message in statusMessages {
            if let notice = classify(message: message, dbCompleted: dbCompleted) {
                notices.append(notice)
            }
        }

        if status?.partialSuccess == true,
           !notices.contains(where: { $0.kind == .partialProgress }) {
            let fallbackMessage: String
            if let leadMessage = statusMessages.first {
                fallbackMessage = leadMessage
            } else if dbCompleted > 0 {
                fallbackMessage = "Backfill stopped early, but \(dbCompleted) recording(s) already stayed synced."
            } else {
                fallbackMessage = "Backfill stopped early after preserving the pages that were already synced."
            }

            notices.append(
                PlaudSyncNotice(
                    kind: .partialProgress,
                    title: "Backfill Kept Partial Progress",
                    message: fallbackMessage
                )
            )
        }

        let failureMessages = (failures?.actionable ?? []).compactMap { item in
            let reason = [item.reason, item.error]
                .compactMap { value in
                    guard let value,
                          !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                        return nil
                    }
                    return value
                }
                .joined(separator: " ")
            return reason.isEmpty ? nil : reason
        }

        for message in failureMessages {
            if let notice = classify(message: message, dbCompleted: dbCompleted) {
                notices.append(notice)
            }
        }

        var seen = Set<String>()
        return notices.filter { seen.insert($0.id).inserted }
    }

    private static func classify(message: String, dbCompleted: Int) -> PlaudSyncNotice? {
        let normalized = message.lowercased()

        if repeatedPageKeywords.contains(where: normalized.contains) {
            return PlaudSyncNotice(
                kind: .repeatedPage,
                title: "Repeated Plaud Page Stopped Backfill",
                message: message
            )
        }

        if partialProgressKeywords.contains(where: normalized.contains) {
            let suffix = dbCompleted > 0 && !normalized.contains("\(dbCompleted)")
                ? " \(dbCompleted) recording(s) already remain available locally."
                : ""
            return PlaudSyncNotice(
                kind: .partialProgress,
                title: "Backfill Kept Partial Progress",
                message: message + suffix
            )
        }

        if rateLimitKeywords.contains(where: normalized.contains) {
            let suffix = dbCompleted > 0 && partialProgressKeywords.contains(where: normalized.contains) == false
                ? " \(dbCompleted) recording(s) were already preserved before the rate limit hit."
                : ""
            return PlaudSyncNotice(
                kind: .rateLimited,
                title: "Plaud Rate Limit Pressure",
                message: message + suffix
            )
        }

        if warningKeywords.contains(where: normalized.contains) {
            return PlaudSyncNotice(
                kind: .warning,
                title: "Plaud Sync Warning",
                message: message
            )
        }

        return nil
    }

    private static let repeatedPageKeywords = [
        "repeated page",
        "page repeated",
        "same page",
        "page signature"
    ]

    private static let partialProgressKeywords = [
        "partial",
        "preserved",
        "kept",
        "remain synced",
        "stopped early"
    ]

    private static let rateLimitKeywords = [
        "rate limit",
        "429",
        "too many requests"
    ]

    private static let warningKeywords = [
        "backfill",
        "history",
        "plaud",
        "retry later",
        "validation"
    ]
}

// MARK: - Workflows

/// Request to submit Plaud AI workflows.
struct WorkflowSubmitRequest: Codable, Sendable {
    var daysBack: Int = 30
    var limit: Int = 50
    var templateId: String?
    var model: String = "openai"

    enum CodingKeys: String, CodingKey {
        case daysBack = "days_back"
        case limit
        case templateId = "template_id"
        case model
    }
}

/// Request to refresh workflow statuses.
struct WorkflowRefreshRequest: Codable, Sendable {
    var daysBack: Int = 60
    var limit: Int = 50

    enum CodingKeys: String, CodingKey {
        case daysBack = "days_back"
        case limit
    }
}

/// Request to run a workflow on a single recording.
struct RecordingWorkflowRequest: Codable, Sendable {
    var templateId: String?
    var model: String = "openai"

    enum CodingKeys: String, CodingKey {
        case templateId = "template_id"
        case model
    }
}

/// Category override request.
struct CategoryOverrideRequest: Codable, Sendable {
    let category: String
}

/// Generic success response.
struct SuccessResponse: Codable, Sendable {
    let success: Bool
    let message: String
}
