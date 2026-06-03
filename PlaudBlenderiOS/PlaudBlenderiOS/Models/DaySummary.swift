import Foundation

/// Matches `DaySummaryOut` from the FastAPI backend.
struct DaySummary: Codable, Identifiable, Sendable {
    let date: String
    let dateDisplay: String?
    let totalDurationSeconds: Double
    let recordingCount: Int
    let eventCount: Int
    let coverageStatus: String?
    let coverageNote: String?
    let topCategory: String?
    let categoryPercentages: [String: Double]?
    let topKeywords: [String]?
    let aiSummary: String?
    let recordings: [RecordingSummary]?

    var id: String { date }

    var hasTimelineContent: Bool {
        if let recordings {
            return !recordings.isEmpty
        }

        return recordingCount > 0 || eventCount > 0
    }

    var shouldDisplayInTimeline: Bool {
        hasTimelineContent
    }

    nonisolated static func sortMostRecentFirst(_ lhs: DaySummary, _ rhs: DaySummary) -> Bool {
        if lhs.date != rhs.date {
            return lhs.date > rhs.date
        }
        return false
    }

    enum CodingKeys: String, CodingKey {
        case date
        case dateDisplay = "date_display"
        case totalDurationSeconds = "total_duration_seconds"
        case recordingCount = "recording_count"
        case eventCount = "event_count"
        case coverageStatus = "coverage_status"
        case coverageNote = "coverage_note"
        case topCategory = "top_category"
        case categoryPercentages = "category_percentages"
        case topKeywords = "top_keywords"
        case aiSummary = "ai_summary"
        case recordings
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        date = try container.decode(String.self, forKey: .date)
        dateDisplay = try container.decodeIfPresent(String.self, forKey: .dateDisplay)
        totalDurationSeconds = try container.decode(Double.self, forKey: .totalDurationSeconds)
        recordingCount = try container.decode(Int.self, forKey: .recordingCount)
        eventCount = try container.decode(Int.self, forKey: .eventCount)
        coverageStatus = try container.decodeIfPresent(String.self, forKey: .coverageStatus)
        coverageNote = try container.decodeIfPresent(String.self, forKey: .coverageNote)
        topCategory = try container.decodeIfPresent(String.self, forKey: .topCategory)
        categoryPercentages = try container.decodeIfPresent([String: Double].self, forKey: .categoryPercentages)
        topKeywords = try container.decodeIfPresent([String].self, forKey: .topKeywords)
        aiSummary = try container.decodeIfPresent(String.self, forKey: .aiSummary)
        recordings = try container.decodeIfPresent([RecordingSummary].self, forKey: .recordings)?
            .filter { !$0.isSyntheticNotionPlaceholder }
            .sorted(by: RecordingSummary.sortMostRecentFirst)
    }
}

/// Wrapper for the `/api/days` and `/api/days/filled` responses.
struct DaysResponse: Codable, Sendable {
    let days: [DaySummary]
    let total: Int?
}
