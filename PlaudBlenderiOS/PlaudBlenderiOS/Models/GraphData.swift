import Foundation

/// Matches `GraphDataOut` from the FastAPI backend.
struct GraphData: Decodable, Sendable {
    let nodes: [[String: JSONValue]]
    let edges: [[String: JSONValue]]

    private enum CodingKeys: String, CodingKey {
        case nodes, edges
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        nodes = try container.decode([GraphElement].self, forKey: .nodes).map(\.values)
        edges = try container.decode([GraphElement].self, forKey: .edges).map(\.values)
    }
}

private struct GraphElement: Decodable {
    let values: [String: JSONValue]

    private enum CodingKeys: String, CodingKey {
        case data
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        if let data = try container.decodeIfPresent([String: JSONValue].self, forKey: .data) {
            values = data
            return
        }

        let valueContainer = try decoder.singleValueContainer()
        values = try valueContainer.decode([String: JSONValue].self)
    }
}

/// Parsed graph node for display.
struct GraphNode: Identifiable, Encodable, Sendable {
    let id: String
    let label: String
    let fullLabel: String
    let type: String
    let size: Double
    let color: String
    let count: Int?
    let categories: [String]
    let mentionCount: Int?
    let sentiment: Double?
    let avg_ts: Double?

    init(from dict: [String: JSONValue]) {
        self.id = dict["id"]?.stringValue ?? UUID().uuidString
        self.label = dict["label"]?.stringValue ?? "?"
        self.fullLabel = dict["full_label"]?.stringValue ?? label
        self.type = dict["type"]?.stringValue ?? "unknown"
        self.size = dict["size"]?.doubleValue ?? 20
        self.color = dict["color"]?.stringValue ?? Self.fallbackColor(for: type)
        self.count = dict["count"]?.intValue
        self.categories = dict["categories"]?.stringValue?
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty } ?? []
        self.mentionCount = dict["mention_count"]?.intValue
        self.sentiment = dict["sentiment"]?.doubleValue
        self.avg_ts = dict["avg_ts"]?.doubleValue
    }

    var metricValue: Int? {
        mentionCount ?? count
    }

    private static func fallbackColor(for type: String) -> String {
        switch type {
        case "category":
            return "#334155"
        case "topic":
            return "#0f766e"
        default:
            return "#64748b"
        }
    }
}

/// Parsed graph edge for display.
struct GraphEdge: Identifiable, Encodable, Sendable {
    let id: String
    let source: String
    let target: String
    let label: String?
    let weight: Double

    init(from dict: [String: JSONValue]) {
        self.source = dict["source"]?.stringValue ?? ""
        self.target = dict["target"]?.stringValue ?? ""
        self.label = dict["label"]?.stringValue
        self.weight = dict["weight"]?.doubleValue ?? 1.0
        self.id = dict["id"]?.stringValue ?? "\(source)-\(target)"
    }
}

struct GraphRenderPayload: Encodable, Sendable {
    let nodes: [GraphNode]
    let edges: [GraphEdge]
}
