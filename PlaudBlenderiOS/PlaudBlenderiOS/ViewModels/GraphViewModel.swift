import Foundation
import Observation

struct GraphLayoutOption: Identifiable, Hashable, Sendable {
    let id: String
    let title: String
    let subtitle: String

    static let lanes = GraphLayoutOption(
        id: "lanes",
        title: "Lanes",
        subtitle: "Topics grouped under their strongest category"
    )

    static let breadthfirst = GraphLayoutOption(
        id: "breadthfirst",
        title: "Levels",
        subtitle: "Category hubs above the topics they anchor"
    )

    static let concentric = GraphLayoutOption(
        id: "concentric",
        title: "Orbit",
        subtitle: "Large nodes centered, smaller ones around them"
    )

    static let circle = GraphLayoutOption(
        id: "circle",
        title: "Circle",
        subtitle: "Quick scan of the whole graph"
    )

    static let force = GraphLayoutOption(
        id: "cose",
        title: "Force",
        subtitle: "Freeform exploration of cross-category spillover"
    )

    static let all = [lanes, breadthfirst, concentric, circle, force]
}

struct GraphConnection: Identifiable, Sendable {
    let node: GraphNode
    let weight: Double

    var id: String { node.id }
}

@Observable
final class GraphViewModel {
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
            nodes = data.nodes.map { GraphNode(from: $0) }
            edges = data.edges.map { GraphEdge(from: $0) }
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

    private func sortNodes(lhs: GraphNode, rhs: GraphNode) -> Bool {
        let lhsMetric = lhs.metricValue ?? 0
        let rhsMetric = rhs.metricValue ?? 0
        if lhsMetric != rhsMetric { return lhsMetric > rhsMetric }
        return lhs.fullLabel < rhs.fullLabel
    }
}
