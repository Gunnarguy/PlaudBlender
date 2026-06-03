import Foundation

/// Matches `EventOut` from the FastAPI backend.
struct Event: Codable, Identifiable, Hashable, Sendable {
    let id: String
    let recordingId: String
    let startTs: String
    let endTs: String
    let dayOfWeek: String
    let hourOfDay: Int
    let cleanText: String
    let category: String
    let categoryConfidence: Double?
    let sentiment: Double?
    let keywords: [String]
    let speaker: String
    let durationSeconds: Double

    enum CodingKeys: String, CodingKey {
        case id
        case recordingId = "recording_id"
        case startTs = "start_ts"
        case endTs = "end_ts"
        case dayOfWeek = "day_of_week"
        case hourOfDay = "hour_of_day"
        case cleanText = "clean_text"
        case category
        case categoryConfidence = "category_confidence"
        case sentiment
        case keywords
        case speaker
        case durationSeconds = "duration_seconds"
    }
}

enum EventCategory: String, Codable, CaseIterable, Sendable {
    case work
    case personal
    case meeting
    case deepWork = "deep_work"
    case breakTime = "break"
    case reflection
    case idea
    case unknown
}
