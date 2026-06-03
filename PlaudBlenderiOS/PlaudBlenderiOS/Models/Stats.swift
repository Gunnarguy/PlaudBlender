import Foundation

/// Matches `StatsOut` from the FastAPI backend.
struct Stats: Codable, Sendable {
    let totalRecordings: Int
    let totalEvents: Int
    let totalDays: Int
    let totalDurationHours: Double
    let categories: [String: Int]
    let sentimentAvg: Double?
    let topKeywords: [[String: JSONValue]]?
    let categoriesByHour: [String: JSONValue]?
    let sentimentDistribution: [String: Int]?
    let recentDays: [[String: JSONValue]]?
    let avgEventsPerRecording: Double?
    let avgRecordingDurationMin: Double?
    let mostProductiveDay: String?
    let mostProductiveHour: Int?
    let longestRecordingMin: Double?
    let pipelineCompletionRate: Double?
    let eventsByHour: [String: Int]?
    let eventsByDayOfWeek: [String: Int]?
    let plaudCloudStats: [String: JSONValue]?

    enum CodingKeys: String, CodingKey {
        case totalRecordings = "total_recordings"
        case totalEvents = "total_events"
        case totalDays = "total_days"
        case totalDurationHours = "total_duration_hours"
        case categories
        case sentimentAvg = "sentiment_avg"
        case topKeywords = "top_keywords"
        case categoriesByHour = "categories_by_hour"
        case sentimentDistribution = "sentiment_distribution"
        case recentDays = "recent_days"
        case avgEventsPerRecording = "avg_events_per_recording"
        case avgRecordingDurationMin = "avg_recording_duration_min"
        case mostProductiveDay = "most_productive_day"
        case mostProductiveHour = "most_productive_hour"
        case longestRecordingMin = "longest_recording_min"
        case pipelineCompletionRate = "pipeline_completion_rate"
        case eventsByHour = "events_by_hour"
        case eventsByDayOfWeek = "events_by_day_of_week"
        case plaudCloudStats = "plaud_cloud_stats"
    }

    // MARK: - Computed fallbacks (derive from existing data if backend doesn't send)

    var effectiveAvgEvents: Double {
        avgEventsPerRecording ?? (totalRecordings > 0 ? Double(totalEvents) / Double(totalRecordings) : 0)
    }

    var effectiveAvgDuration: Double {
        avgRecordingDurationMin ?? (totalRecordings > 0 ? totalDurationHours * 60.0 / Double(totalRecordings) : 0)
    }

    var averageEventsPerDay: Double {
        totalDays > 0 ? Double(totalEvents) / Double(totalDays) : 0
    }

    var averageHoursPerDay: Double {
        totalDays > 0 ? totalDurationHours / Double(totalDays) : 0
    }

    var averageRecordingsPerDay: Double {
        totalDays > 0 ? Double(totalRecordings) / Double(totalDays) : 0
    }

    var dominantCategory: (name: String, count: Int, share: Double)? {
        guard let top = categories.max(by: { $0.value < $1.value }) else { return nil }
        let totalCategoryEvents = max(categories.values.reduce(0, +), totalEvents)
        let share = totalCategoryEvents > 0 ? Double(top.value) / Double(totalCategoryEvents) : 0
        return (top.key, top.value, share)
    }

    var categoryDiversity: Int {
        categories.values.filter { $0 > 0 }.count
    }

    var effectivePeakHour: Int? {
        if let mostProductiveHour {
            return mostProductiveHour
        }
        return effectiveEventsByHour.max(by: { $0.value < $1.value })?.key
    }

    var positiveShare: Double? {
        guard let sentimentDistribution else { return nil }
        let total = sentimentDistribution.values.reduce(0, +)
        guard total > 0 else { return nil }
        return Double(sentimentDistribution["positive"] ?? 0) / Double(total)
    }

    var negativeShare: Double? {
        guard let sentimentDistribution else { return nil }
        let total = sentimentDistribution.values.reduce(0, +)
        guard total > 0 else { return nil }
        return Double(sentimentDistribution["negative"] ?? 0) / Double(total)
    }

    var recentDayStats: [RecentDayStat] {
        guard let recentDays else { return [] }
        return recentDays.compactMap { raw in
            let dateString = raw["date"]?.stringValue
                ?? raw["day"]?.stringValue
                ?? raw["label"]?.stringValue
                ?? raw["recorded_on"]?.stringValue
            guard let dateString else { return nil }

            let events = raw["events"]?.intValue
                ?? raw["total_events"]?.intValue
                ?? raw["event_count"]?.intValue
                ?? 0
            let recordings = raw["recordings"]?.intValue
                ?? raw["total_recordings"]?.intValue
                ?? raw["recording_count"]?.intValue
                ?? raw["count"]?.intValue
            let hours = raw["hours"]?.doubleValue
                ?? raw["duration_hours"]?.doubleValue
                ?? raw["total_duration_hours"]?.doubleValue
                ?? raw["duration_minutes"]?.doubleValue.map { $0 / 60.0 }
                ?? raw["total_duration_minutes"]?.doubleValue.map { $0 / 60.0 }
            let summary = raw["summary"]?.stringValue
                ?? raw["headline"]?.stringValue
                ?? raw["note"]?.stringValue

            return RecentDayStat(
                dateString: dateString,
                date: Stats.parseDate(dateString),
                events: events,
                recordings: recordings,
                durationHours: hours,
                summary: summary
            )
        }
        .sorted { lhs, rhs in
            switch (lhs.date, rhs.date) {
            case let (left?, right?):
                return left < right
            default:
                return lhs.dateString < rhs.dateString
            }
        }
    }

    /// Derive events-by-hour from categoriesByHour if not provided directly
    var effectiveEventsByHour: [Int: Int] {
        if let direct = eventsByHour {
            return direct.reduce(into: [:]) { result, pair in
                if let h = Int(pair.key) { result[h] = pair.value }
            }
        }
        guard let byHour = categoriesByHour else { return [:] }
        return byHour.reduce(into: [:]) { result, pair in
            guard let h = Int(pair.key), let cats = pair.value.objectValue else { return }
            result[h] = cats.values.compactMap(\.intValue).reduce(0, +)
        }
    }

    /// Parse heatmap data: [hour][category] = count
    var heatmapData: [(hour: Int, category: String, count: Int)] {
        guard let byHour = categoriesByHour else { return [] }
        var rows: [(hour: Int, category: String, count: Int)] = []
        for (hourStr, catObj) in byHour {
            guard let h = Int(hourStr), let cats = catObj.objectValue else { continue }
            for (cat, val) in cats {
                if let c = val.intValue, c > 0 { rows.append((h, cat, c)) }
            }
        }
        return rows
    }

    private static func parseDate(_ value: String) -> Date? {
        if let iso = value.iso8601Date {
            return iso
        }

        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter.date(from: value)
    }
}

struct RecentDayStat: Identifiable, Sendable {
    let dateString: String
    let date: Date?
    let events: Int
    let recordings: Int?
    let durationHours: Double?
    let summary: String?

    var id: String { dateString }

    var displayLabel: String {
        if let date {
            return date.shortDateString
        }
        return dateString
    }
}

/// DB-level recording stats from `/api/sync/db-stats`.
struct RecordingDbStats: Decodable, Sendable {
    let total: Int
    let completed: Int
    let pending: Int
    let processing: Int
    let failed: Int
    let noTranscript: Int

    enum CodingKeys: String, CodingKey {
        case total, completed, pending, processing, failed
        case noTranscript = "no_transcript"
    }

    init(
        total: Int,
        completed: Int,
        pending: Int,
        processing: Int,
        failed: Int,
        noTranscript: Int
    ) {
        self.total = total
        self.completed = completed
        self.pending = pending
        self.processing = processing
        self.failed = failed
        self.noTranscript = noTranscript
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        completed = try container.decodeIfPresent(Int.self, forKey: .completed) ?? 0
        pending = try container.decodeIfPresent(Int.self, forKey: .pending) ?? 0
        processing = try container.decodeIfPresent(Int.self, forKey: .processing) ?? 0
        failed = try container.decodeIfPresent(Int.self, forKey: .failed) ?? 0
        noTranscript = try container.decodeIfPresent(Int.self, forKey: .noTranscript) ?? 0

        total = try container.decodeIfPresent(Int.self, forKey: .total)
            ?? completed + pending + processing + failed + noTranscript
    }
}

/// Workflow stats from `/api/sync/workflows/stats`.
struct WorkflowStats: Codable, Sendable {
    let recentRecordings: Int?
    let withAiSummary: Int?
    let missingAiSummary: Int?
    let readyForEnrichment: Int?
    let workflowPending: Int?
    let workflowFailed: Int?
    let workflowSuccess: Int?
    let lastSubmittedAt: String?
    let activeWorkflows: [JSONValue]?

    enum CodingKeys: String, CodingKey {
        case recentRecordings = "recent_recordings"
        case withAiSummary = "with_ai_summary"
        case missingAiSummary = "missing_ai_summary"
        case readyForEnrichment = "ready_for_enrichment"
        case workflowPending = "workflow_pending"
        case workflowFailed = "workflow_failed"
        case workflowSuccess = "workflow_success"
        case lastSubmittedAt = "last_submitted_at"
        case activeWorkflows = "active_workflows"
    }
}
