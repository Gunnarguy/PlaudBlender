import Foundation

/// Matches `TopicOut` from the FastAPI backend.
struct Topic: Codable, Identifiable, Hashable, Sendable {
    let name: String
    let count: Int

    var id: String { name }
}

/// Matches `TopicOccurrenceOut` from the FastAPI backend.
struct TopicOccurrence: Codable, Identifiable, Sendable {
    let eventId: String
    let recordingId: String
    let timestamp: String
    let textSnippet: String
    let category: String

    var id: String { eventId }

    enum CodingKeys: String, CodingKey {
        case eventId = "event_id"
        case recordingId = "recording_id"
        case timestamp
        case textSnippet = "text_snippet"
        case category
    }
}

/// Matches `TopicTimelineOut` from the FastAPI backend.
struct TopicTimeline: Codable, Sendable {
    let topic: String
    let totalOccurrences: Int
    let recordingCount: Int
    let occurrences: [TopicOccurrence]

    enum CodingKeys: String, CodingKey {
        case topic
        case totalOccurrences = "total_occurrences"
        case recordingCount = "recording_count"
        case occurrences
    }
}
