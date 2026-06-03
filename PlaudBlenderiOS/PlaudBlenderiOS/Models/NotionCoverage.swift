import Foundation

/// Coverage calendar response from `GET /api/v1/notion/coverage`.
struct NotionCoverageResponse: Decodable, Sendable {
    let calendar: [NotionCoverageDay]?
    let totalNotion: Int?
    let totalChronos: Int?

    enum CodingKeys: String, CodingKey {
        case calendar
        case totalNotion = "total_notion"
        case totalChronos = "total_chronos"
    }
}
