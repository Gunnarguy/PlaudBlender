import Foundation

/// Session cost from `/api/costs/session`.
struct SessionCost: Codable, Sendable {
    let totalCostUsd: Double
    let totalCalls: Int
    let totalInputTokens: Int
    let totalOutputTokens: Int
    let byModel: [String: JSONValue]
    let byType: [String: JSONValue]
    let recent: [CostRecentCall]?
    let sessionMinutes: Double

    enum CodingKeys: String, CodingKey {
        case totalCostUsd = "total_cost_usd"
        case totalCalls = "total_calls"
        case totalInputTokens = "total_input_tokens"
        case totalOutputTokens = "total_output_tokens"
        case byModel = "by_model"
        case byType = "by_type"
        case recent
        case sessionMinutes = "session_minutes"
    }
}

struct CostRecentCall: Codable, Sendable, Identifiable {
    let model: String
    let type: String
    let inputTokens: Int
    let outputTokens: Int
    let costUsd: Double
    let agoS: Double

    var id: String {
        "\(model)-\(type)-\(agoS)-\(costUsd)"
    }

    enum CodingKeys: String, CodingKey {
        case model
        case type
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case costUsd = "cost_usd"
        case agoS = "ago_s"
    }
}

/// Cost history from `/api/costs/history`.
struct CostHistory: Codable, Sendable {
    let days: Int
    let totalCostUsd: Double
    let totalCalls: Int
    let byModel: [String: JSONValue]
    let byDay: [[String: JSONValue]]?

    enum CodingKeys: String, CodingKey {
        case days
        case totalCostUsd = "total_cost_usd"
        case totalCalls = "total_calls"
        case byModel = "by_model"
        case byDay = "by_day"
    }
}

/// Model pricing table from `/api/costs/pricing`.
struct ModelPricing: Codable, Sendable {
    let models: [[String: JSONValue]]
}
