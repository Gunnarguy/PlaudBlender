import Foundation
import Observation

struct GraphLayoutOption: Identifiable, Hashable, Sendable {
    let id: String
    let title: String
    let subtitle: String

    static let lanes = GraphLayoutOption(
        id: "lanes",
        title: "Map",
        subtitle: "Category → its most-mentioned topics"
    )

    static let concentric = GraphLayoutOption(
        id: "concentric",
        title: "Most active",
        subtitle: "Topics ranked by mentions"
    )

    static let circle = GraphLayoutOption(
        id: "circle",
        title: "Over time",
        subtitle: "Topics ordered from earlier to later"
    )

    static let all = [lanes, concentric, circle]
}

struct GraphConnection: Identifiable, Sendable {
    let node: GraphNode
    let weight: Double

    var id: String { node.id }
}

@Observable
final class GraphViewModel {
    private static let lowValueTopicWords: Set<String> = [
        "about", "actually", "after", "again", "all", "also", "anything", "ask",
        "asked", "asking", "asks", "basically", "before", "being", "bought",
        "bring", "bringing", "brings", "brought", "buy", "buying", "buys", "call",
        "called", "calling", "calls", "can", "check", "checked", "checking", "checks",
        "conversation", "could", "did", "do", "does", "doing", "done", "dude",
        "even", "everyone", "everything", "feel", "feeling", "feels", "felt",
        "from", "fucking", "gave", "general", "get", "gets", "getting", "give",
        "given", "gives", "giving", "go", "goes", "going", "gone", "gonna", "good",
        "got", "gotta", "guy", "guys", "had", "has", "have", "having", "held", "here",
        "hey", "hold", "holding", "holds", "how", "into", "just", "keep", "keeping",
        "keeps", "kept", "kind", "kinds", "knew", "know", "knowing", "knows", "like",
        "literally", "look", "looked", "looking", "looks", "lot", "lots", "make",
        "makes", "making", "maybe", "misc", "more", "most", "need", "needed",
        "needing", "needs", "not", "nothing", "ok", "okay", "other", "part", "parts",
        "people", "person", "point", "points", "put", "puts", "putting", "ran",
        "really", "run", "running", "runs", "said", "saw", "say", "saying", "says",
        "see", "seeing", "seen", "sees", "set", "sets", "setting", "should", "some",
        "somebody", "someone", "something", "sort", "sorts", "stuff", "swap",
        "swapped", "swapping", "swaps", "take", "taken", "takes", "taking", "talk",
        "talked", "talking", "talks", "tell", "telling", "tells", "that", "the",
        "them", "there", "they", "thing", "things", "think", "thinking", "thinks",
        "this", "thought", "through", "time", "today", "told", "took", "tried",
        "tries", "try", "trying", "turn", "turned", "turning", "turns", "type",
        "types", "unknown", "use", "used", "uses", "using", "very", "want",
        "wanted", "wanting", "wants", "wanna", "was", "way", "ways", "week", "well",
        "went", "were", "what", "when", "where", "which", "who", "why", "will",
        "with", "work", "worked", "working", "works", "would", "yeah", "yes", "your"
    ]
    private static let disallowedTopicWords: Set<String> = [
        "damn", "fuck", "fucked", "fucking", "fucks", "shit", "shits", "shitty"
    ]

    var graphData: GraphData?
    var nodes: [GraphNode] = []
    var edges: [GraphEdge] = []
    var isLoading = false
    var error: String?
    var selectedLayout = GraphLayoutOption.lanes.id

    let availableLayouts = GraphLayoutOption.all

    private let api: APIClient

    var categoryNodes: [GraphNode] {
        nodes
            .filter { $0.type == "category" }
            .sorted(by: sortNodes)
    }

    var topicNodes: [GraphNode] {
        nodes
            .filter { $0.type == "topic" }
            .sorted(by: sortNodes)
    }

    var graphSignature: String {
        let nodePart = nodes
            .sorted { $0.id < $1.id }
            .map {
                [
                    $0.id,
                    $0.label,
                    $0.fullLabel,
                    $0.type,
                    $0.color,
                    String(format: "%.2f", $0.size),
                    "\($0.count ?? -1)",
                    "\($0.mentionCount ?? -1)",
                    $0.categories.joined(separator: ","),
                    "\($0.avg_ts ?? 0.0)"
                ].joined(separator: "|")
            }
            .joined(separator: ";")

        let edgePart = edges
            .sorted { lhs, rhs in
                if lhs.source != rhs.source { return lhs.source < rhs.source }
                if lhs.target != rhs.target { return lhs.target < rhs.target }
                return lhs.id < rhs.id
            }
            .map {
                [
                    $0.id,
                    $0.source,
                    $0.target,
                    String(format: "%.2f", $0.weight),
                    $0.label ?? ""
                ].joined(separator: "|")
            }
            .joined(separator: ";")

        return "\(nodePart)#\(edgePart)"
    }

    init(api: APIClient) {
        self.api = api
    }

    func loadGraph() async {
        isLoading = true
        error = nil
        do {
            let data: GraphData = try await api.get("/api/graph")
            graphData = data
            let parsedNodes = data.nodes.map { GraphNode(from: $0) }
            nodes = parsedNodes.filter {
                $0.type == "category" || ($0.type == "topic" && shouldDisplayTopic($0))
            }
            let visibleNodeIDs = Set(nodes.map(\.id))
            edges = data.edges
                .map { GraphEdge(from: $0) }
                .filter {
                    visibleNodeIDs.contains($0.source) && visibleNodeIDs.contains($0.target)
                }
        } catch {
            self.error = error.localizedDescription
        }
        isLoading = false
    }

    func refresh() async {
        await loadGraph()
    }

    func node(withID id: String?) -> GraphNode? {
        guard let id else { return nil }
        return nodes.first(where: { $0.id == id })
    }

    func strongestConnections(for focusNode: GraphNode, limit: Int = 6) -> [GraphConnection] {
        edges
            .compactMap { edge -> GraphConnection? in
                if edge.source == focusNode.id, let related = node(withID: edge.target) {
                    return GraphConnection(node: related, weight: edge.weight)
                }

                if edge.target == focusNode.id, let related = node(withID: edge.source) {
                    return GraphConnection(node: related, weight: edge.weight)
                }

                return nil
            }
            .sorted { lhs, rhs in
                if lhs.weight != rhs.weight { return lhs.weight > rhs.weight }
                return lhs.node.fullLabel < rhs.node.fullLabel
            }
            .prefix(limit)
            .map { $0 }
    }

    func topics(in category: GraphNode, limit: Int = 8) -> [GraphNode] {
        let categoryKeys = Set([
            normalizedCategoryKey(category.id),
            normalizedCategoryKey(category.label)
        ])

        return topicNodes
            .filter { node in
                node.categories.contains { categoryKeys.contains(normalizedCategoryKey($0)) }
            }
            .prefix(limit)
            .map { $0 }
    }

    private func shouldDisplayTopic(_ node: GraphNode) -> Bool {
        let words = node.fullLabel
            .lowercased()
            .split { !$0.isLetter && !$0.isNumber }
            .map(String.init)

        guard !words.isEmpty,
              !words.contains(where: Self.disallowedTopicWords.contains),
              words.contains(where: { $0.count >= 3 && !Self.lowValueTopicWords.contains($0) }) else {
            return false
        }
        return true
    }

    private func normalizedCategoryKey(_ value: String) -> String {
        value
            .lowercased()
            .filter { $0.isLetter || $0.isNumber }
    }

    private func sortNodes(lhs: GraphNode, rhs: GraphNode) -> Bool {
        let lhsMetric = lhs.metricValue ?? 0
        let rhsMetric = rhs.metricValue ?? 0
        if lhsMetric != rhsMetric { return lhsMetric > rhsMetric }
        return lhs.fullLabel < rhs.fullLabel
    }
}
