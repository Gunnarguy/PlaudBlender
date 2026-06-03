import Foundation

/// Notion connection status.
struct NotionStatus: Decodable, Sendable {
    let isConnected: Bool
    let databaseId: String?
    let databaseTitle: String?
    let totalPages: Int
    let error: String?

    enum CodingKeys: String, CodingKey {
        case isConnected = "is_connected"
        case databaseId = "database_id"
        case databaseTitle = "database_title"
        case databaseName = "database_name"
        case totalPages = "total_pages"
        case pageCount = "page_count"
        case error
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        isConnected = try container.decodeIfPresent(Bool.self, forKey: .isConnected) ?? false
        databaseId = try container.decodeIfPresent(String.self, forKey: .databaseId)
        databaseTitle = try container.decodeIfPresent(String.self, forKey: .databaseTitle)
            ?? container.decodeIfPresent(String.self, forKey: .databaseName)
        totalPages = try container.decodeIfPresent(Int.self, forKey: .totalPages)
            ?? container.decodeIfPresent(Int.self, forKey: .pageCount)
            ?? 0
        error = try container.decodeIfPresent(String.self, forKey: .error)
    }
}

/// A Notion database available for selection.
struct NotionDatabase: Codable, Identifiable, Sendable {
    let id: String
    let title: String
    let pageCount: Int?

    enum CodingKeys: String, CodingKey {
        case id, title
        case pageCount = "page_count"
    }
}

/// Notion database select request.
struct NotionDatabaseSelectRequest: Codable, Sendable {
    let dbId: String

    enum CodingKeys: String, CodingKey {
        case dbId = "db_id"
    }
}

/// A Notion recording page.
struct NotionRecording: Codable, Identifiable, Sendable {
    let pageId: String
    let title: String
    let createdTime: String?
    let lastEditedTime: String?
    let url: String?
    let transcript: String?
    let summary: String?
    let date: String?
    let duration: String?
    let tags: [String]?
    let category: String?
    let matchedRecordingId: String?

    var id: String { pageId }

    var isImportedToChronos: Bool {
        matchedRecordingId != nil
    }

    enum CodingKeys: String, CodingKey {
        case pageId = "page_id"
        case title
        case createdTime = "created_time"
        case lastEditedTime = "last_edited_time"
        case url
        case transcript, summary, date, duration, tags, category
        case matchedRecordingId = "matched_recording_id"
    }
}

/// Paginated response wrapper for Notion recordings.
struct NotionRecordingsResponse: Decodable, Sendable {
    let recordings: [NotionRecording]
    let total: Int
    let hasMore: Bool

    enum CodingKeys: String, CodingKey {
        case recordings, total
        case hasMore = "has_more"
    }
}

/// Import request body.
struct NotionImportRequest: Codable, Sendable {
    var process: Bool = true
    var index: Bool = true
    var batchSize: Int?
    var force: Bool?

    enum CodingKeys: String, CodingKey {
        case process, index, force
        case batchSize = "batch_size"
    }
}

/// Import progress from `/api/notion/import/progress`.
struct NotionImportProgress: Decodable, Sendable {
    let status: String
    let total: Int
    let completed: Int
    let failed: Int
    let skipped: Int
    let currentTitle: String?
    let errors: [String]

    var imported: Int { completed }

    var pending: Int {
        max(0, total - completed - failed - skipped)
    }

    enum CodingKeys: String, CodingKey {
        case status, total, completed, failed, skipped, errors
        case currentTitle = "current_title"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        status = try container.decodeIfPresent(String.self, forKey: .status) ?? "idle"
        total = try container.decodeIfPresent(Int.self, forKey: .total) ?? 0
        completed = try container.decodeIfPresent(Int.self, forKey: .completed)
            ?? container.decodeIfPresent(Int.self, forKey: .skipped)
            ?? 0
        failed = try container.decodeIfPresent(Int.self, forKey: .failed) ?? 0
        skipped = try container.decodeIfPresent(Int.self, forKey: .skipped) ?? 0
        currentTitle = try container.decodeIfPresent(String.self, forKey: .currentTitle)
        errors = try container.decodeIfPresent([String].self, forKey: .errors) ?? []
    }
}

struct NotionImportPreviewSample: Codable, Identifiable, Sendable {
    let pageId: String
    let title: String
    let date: String?
    let url: String?

    var id: String { pageId }

    enum CodingKeys: String, CodingKey {
        case pageId = "page_id"
        case title, date, url
    }
}

struct NotionImportPreview: Decodable, Sendable {
    let totalPages: Int
    let completedImports: Int
    let failedImports: Int
    let matchedToExisting: Int
    let pendingImportRaw: Int
    let duplicatePagesCollapsed: Int
    let pendingImport: Int
    let blindImportLimit: Int
    let blockedWithoutForce: Bool
    let sample: [NotionImportPreviewSample]

    enum CodingKeys: String, CodingKey {
        case totalPages = "total_pages"
        case completedImports = "completed_imports"
        case failedImports = "failed_imports"
        case matchedToExisting = "matched_to_existing"
        case pendingImportRaw = "pending_import_raw"
        case duplicatePagesCollapsed = "duplicate_pages_collapsed"
        case pendingImport = "pending_import"
        case blindImportLimit = "blind_import_limit"
        case blockedWithoutForce = "blocked_without_force"
        case sample
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        totalPages = try container.decodeIfPresent(Int.self, forKey: .totalPages) ?? 0
        completedImports = try container.decodeIfPresent(Int.self, forKey: .completedImports) ?? 0
        failedImports = try container.decodeIfPresent(Int.self, forKey: .failedImports) ?? 0
        matchedToExisting = try container.decodeIfPresent(Int.self, forKey: .matchedToExisting) ?? 0
        let decodedPendingImportRaw = try container.decodeIfPresent(Int.self, forKey: .pendingImportRaw)
        let decodedPendingImport = try container.decodeIfPresent(Int.self, forKey: .pendingImport)
        pendingImportRaw = decodedPendingImportRaw ?? decodedPendingImport ?? 0
        duplicatePagesCollapsed = try container.decodeIfPresent(Int.self, forKey: .duplicatePagesCollapsed) ?? 0
        pendingImport = decodedPendingImport ?? 0
        blindImportLimit = try container.decodeIfPresent(Int.self, forKey: .blindImportLimit) ?? 25
        blockedWithoutForce = try container.decodeIfPresent(Bool.self, forKey: .blockedWithoutForce) ?? false
        sample = try container.decodeIfPresent([NotionImportPreviewSample].self, forKey: .sample) ?? []
    }
}

struct NotionTranscriptAliasCandidate: Decodable, Identifiable, Sendable {
    let pageId: String
    let title: String
    let date: String?
    let candidateRecordingId: String
    let candidateTitle: String
    let transcriptSimilarity: Double

    var id: String { pageId }

    enum CodingKeys: String, CodingKey {
        case pageId = "page_id"
        case title, date
        case candidateRecordingId = "candidate_recording_id"
        case candidateTitle = "candidate_title"
        case transcriptSimilarity = "transcript_similarity"
    }
}

struct NotionReviewPage: Decodable, Identifiable, Sendable {
    let pageId: String
    let title: String
    let date: String?
    let url: String?

    var id: String { pageId }

    enum CodingKeys: String, CodingKey {
        case pageId = "page_id"
        case title, date, url
    }
}

struct NotionDuplicateGroup: Decodable, Identifiable, Sendable {
    let groupSize: Int
    let pages: [NotionReviewPage]

    var id: String {
        pages.first?.pageId ?? "group-\(groupSize)"
    }

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case pages
    }
}

struct NotionMatchReview: Decodable, Sendable {
    let pendingTotal: Int
    let manualOverrides: [String: String]
    let manualOverrideCount: Int
    let highConfidenceTranscriptAliases: [NotionTranscriptAliasCandidate]
    let highConfidenceTranscriptAliasCount: Int
    let duplicateGroups: [NotionDuplicateGroup]
    let duplicateGroupCount: Int

    enum CodingKeys: String, CodingKey {
        case pendingTotal = "pending_total"
        case manualOverrides = "manual_overrides"
        case manualOverrideCount = "manual_override_count"
        case highConfidenceTranscriptAliases = "high_confidence_transcript_aliases"
        case highConfidenceTranscriptAliasCount = "high_confidence_transcript_alias_count"
        case duplicateGroups = "duplicate_groups"
        case duplicateGroupCount = "duplicate_group_count"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        pendingTotal = try container.decodeIfPresent(Int.self, forKey: .pendingTotal) ?? 0
        manualOverrides = try container.decodeIfPresent([String: String].self, forKey: .manualOverrides) ?? [:]
        manualOverrideCount = try container.decodeIfPresent(Int.self, forKey: .manualOverrideCount) ?? manualOverrides.count
        highConfidenceTranscriptAliases = try container.decodeIfPresent([NotionTranscriptAliasCandidate].self, forKey: .highConfidenceTranscriptAliases) ?? []
        highConfidenceTranscriptAliasCount = try container.decodeIfPresent(Int.self, forKey: .highConfidenceTranscriptAliasCount) ?? highConfidenceTranscriptAliases.count
        duplicateGroups = try container.decodeIfPresent([NotionDuplicateGroup].self, forKey: .duplicateGroups) ?? []
        duplicateGroupCount = try container.decodeIfPresent(Int.self, forKey: .duplicateGroupCount) ?? duplicateGroups.count
    }
}

struct NotionMatchOverrideRequest: Codable, Sendable {
    let pageId: String
    let recordingId: String?
    let clear: Bool

    enum CodingKeys: String, CodingKey {
        case pageId = "page_id"
        case recordingId = "recording_id"
        case clear
    }
}

struct NotionBulkMatchOverrideRequest: Codable, Sendable {
    let overrides: [NotionMatchOverrideRequest]
    let stopOnError: Bool

    init(overrides: [NotionMatchOverrideRequest], stopOnError: Bool = false) {
        self.overrides = overrides
        self.stopOnError = stopOnError
    }

    enum CodingKeys: String, CodingKey {
        case overrides
        case stopOnError = "stop_on_error"
    }
}

struct NotionBulkMatchOverrideResult: Decodable, Identifiable, Sendable {
    let pageId: String
    let action: String
    let ok: Bool
    let message: String

    var id: String { "\(action):\(pageId)" }

    enum CodingKeys: String, CodingKey {
        case pageId = "page_id"
        case action, ok, message
    }
}

struct NotionBulkMatchOverrideResponse: Decodable, Sendable {
    let applied: Int
    let cleared: Int
    let failed: Int
    let results: [NotionBulkMatchOverrideResult]
}

/// Coverage calendar entry.
struct NotionCoverageDay: Codable, Identifiable, Sendable {
    let date: String
    let hasChronos: Bool
    let hasNotion: Bool

    var id: String { date }

    enum CodingKeys: String, CodingKey {
        case date
        case hasChronos = "has_chronos"
        case hasNotion = "has_notion"
    }
}
