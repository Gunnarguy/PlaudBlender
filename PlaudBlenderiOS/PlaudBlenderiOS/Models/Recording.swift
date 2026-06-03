import Foundation

/// Matches `RecordingSummaryOut` from the FastAPI backend.
struct RecordingSummary: Codable, Identifiable, Hashable, Sendable {
    let recordingId: String
    let startTime: String?
    let createdAt: String?
    let endTime: String?
    let durationSeconds: Int
    let durationFormatted: String?
    let topCategory: String
    let eventCount: Int
    let timeRangeFormatted: String?
    let timeIsEstimated: Bool?
    let timeEstimateReason: String?
    let title: String?
    let plaudAiSummary: String?
    let cloudStatus: String?
    let source: String?
    let processingStatus: String?
    let hasPlaudAI: Bool?
    let plaudWorkflowStatus: String?
    let notionState: String?
    let notionPageId: String?
    let notionPageURL: String?
    let notionPageTitle: String?
    let notionMatchCount: Int?

    var id: String { recordingId }

    enum CodingKeys: String, CodingKey {
        case recordingId = "recording_id"
        case startTime = "start_time"
        case createdAt = "created_at"
        case endTime = "end_time"
        case durationSeconds = "duration_seconds"
        case durationFormatted = "duration_formatted"
        case topCategory = "top_category"
        case eventCount = "event_count"
        case timeRangeFormatted = "time_range_formatted"
        case timeIsEstimated = "time_is_estimated"
        case timeEstimateReason = "time_estimate_reason"
        case title
        case plaudAiSummary = "plaud_ai_summary"
        case cloudStatus = "cloud_status"
        case source
        case processingStatus = "processing_status"
        case hasPlaudAI = "has_plaud_ai"
        case plaudWorkflowStatus = "plaud_workflow_status"
        case notionState = "notion_state"
        case notionPageId = "notion_page_id"
        case notionPageURL = "notion_page_url"
        case notionPageTitle = "notion_page_title"
        case notionMatchCount = "notion_match_count"
    }

    nonisolated var canonicalStartTime: String? {
        startTime ?? createdAt
    }

    nonisolated var isSyntheticNotionPlaceholder: Bool {
        recordingId.hasPrefix("notion:")
    }

    nonisolated var canonicalStartDate: Date? {
        canonicalStartTime?.iso8601Date
    }

    nonisolated static func sortMostRecentFirst(_ lhs: RecordingSummary, _ rhs: RecordingSummary) -> Bool {
        switch (lhs.canonicalStartDate, rhs.canonicalStartDate) {
        case let (left?, right?):
            if left != right {
                return left > right
            }
        case (_?, nil):
            return true
        case (nil, _?):
            return false
        case (nil, nil):
            break
        }

        return lhs.recordingId > rhs.recordingId
    }

    nonisolated var cloudCreatedLagDescription: String? {
        guard let startDate = startTime?.iso8601Date,
              let createdDate = createdAt?.iso8601Date,
              createdDate.timeIntervalSince(startDate) >= 60 else {
            return nil
        }

        return "Plaud cloud sync lagged the recording start by \(Self.durationString(createdDate.timeIntervalSince(startDate)))."
    }

    nonisolated private static func durationString(_ interval: TimeInterval) -> String {
        let seconds = max(0, Int(interval.rounded()))
        if seconds < 3600 {
            return "\(max(1, seconds / 60))m"
        }
        let hours = seconds / 3600
        let minutes = (seconds % 3600) / 60
        if minutes == 0 {
            return "\(hours)h"
        }
        return "\(hours)h \(minutes)m"
    }
}

/// Matches `RecordingDetailOut` from the FastAPI backend.
struct RecordingDetail: Codable, Sendable {
    let summary: RecordingSummary
    let events: [Event]
    let categoryPercentages: [String: Double]?
    let transcript: String?
    let aiSummary: String?
    let extractedData: [String: JSONValue]?
    let workflowStatus: [String: JSONValue]?
    let plaudTranscript: String?
    let processing: RecordingProcessing?

    enum CodingKeys: String, CodingKey {
        case summary, events
        case categoryPercentages = "category_percentages"
        case transcript
        case aiSummary = "ai_summary"
        case extractedData = "extracted_data"
        case workflowStatus = "workflow_status"
        case plaudTranscript = "plaud_transcript"
        case processing
    }

    init(
        summary: RecordingSummary,
        events: [Event],
        categoryPercentages: [String: Double]?,
        transcript: String?,
        aiSummary: String?,
        extractedData: [String: JSONValue]?,
        workflowStatus: [String: JSONValue]?,
        plaudTranscript: String?,
        processing: RecordingProcessing? = nil
    ) {
        self.summary = summary
        self.events = events
        self.categoryPercentages = categoryPercentages
        self.transcript = transcript
        self.aiSummary = aiSummary
        self.extractedData = extractedData
        self.workflowStatus = workflowStatus
        self.plaudTranscript = plaudTranscript
        self.processing = processing
    }
}
