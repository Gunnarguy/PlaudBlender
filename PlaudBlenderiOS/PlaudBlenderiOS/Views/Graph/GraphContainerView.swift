import SwiftUI
import WebKit

#if canImport(UIKit)
import UIKit
private typealias PlatformViewRepresentable = UIViewRepresentable
#elseif canImport(AppKit)
import AppKit
private typealias PlatformViewRepresentable = NSViewRepresentable
#endif

/// Knowledge graph rendered via WKWebView with Cytoscape.js.
struct GraphContainerView: View {
    let viewModel: GraphViewModel
    @State private var selectedNodeID: String?
    @State private var rendererError: String?

    private var selectedNode: GraphNode? {
        viewModel.node(withID: selectedNodeID)
    }

    private var activeError: String? {
        rendererError ?? viewModel.error
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                controlsCard
                    .padding([.horizontal, .top])

                if viewModel.isLoading {
                    Spacer()
                    LoadingView(message: "Building knowledge graph...")
                    Spacer()
                } else if let error = activeError {
                    Spacer()
                    EmptyStateView(
                        icon: "exclamationmark.triangle",
                        title: "Graph Unavailable",
                        message: error,
                        actionTitle: "Retry",
                        action: {
                            rendererError = nil
                            Task { await viewModel.refresh() }
                        }
                    )
                    Spacer()
                } else if viewModel.nodes.isEmpty {
                    Spacer()
                    EmptyStateView(
                        icon: "point.3.connected.trianglepath.dotted",
                        title: "No Graph Data",
                        message: "Run the pipeline with --graph to build the knowledge graph.",
                        actionTitle: "Refresh",
                        action: { Task { await viewModel.refresh() } }
                    )
                    Spacer()
                } else {
                    graphWebView
                        .padding(.horizontal)
                        .padding(.top, 12)
                        .padding(.bottom, 8)
                }
            }
            .navigationTitle("Knowledge Graph")
            .toolbar {
                ToolbarItem(placement: platformTrailingToolbarPlacement) {
                    HStack(spacing: 4) {
                        Text("\(viewModel.nodes.count) nodes")
                        Text("·")
                        Text("\(viewModel.edges.count) edges")
                    }
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                }

                if selectedNode != nil {
                    ToolbarItem(placement: platformTrailingToolbarPlacement) {
                        Button("Clear") {
                            selectedNodeID = nil
                        }
                        .font(.caption)
                    }
                }
            }
            .safeAreaInset(edge: .bottom) {
                if let node = selectedNode,
                   !viewModel.isLoading,
                   activeError == nil,
                   !viewModel.nodes.isEmpty {
                    selectedNodeInspector(node)
                        .padding(.horizontal)
                        .padding(.top, 8)
                        .padding(.bottom, 10)
                }
            }
            .task {
                await viewModel.loadGraph()
            }
        }
    }

    private var controlsCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Make It Legible")
                        .font(.headline)
                    Text("The default view groups topics into category lanes instead of throwing everything into a force blob. Tap any node to isolate its strongest links.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Button {
                    rendererError = nil
                    Task { await viewModel.refresh() }
                } label: {
                    Label("Reload", systemImage: "arrow.clockwise")
                        .font(.caption.weight(.semibold))
                }
                .buttonStyle(.bordered)
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(viewModel.availableLayouts) { option in
                        let isSelected = viewModel.selectedLayout == option.id
                        Button {
                            rendererError = nil
                            viewModel.selectedLayout = option.id
                        } label: {
                            layoutOptionCard(option, isSelected: isSelected)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(viewModel.categoryNodes) { node in
                        HStack(spacing: 6) {
                            Circle()
                                .fill(Color(hex: node.color))
                                .frame(width: 8, height: 8)
                            Text(node.label)
                                .font(.caption.weight(.semibold))
                            if let count = node.count {
                                Text("\(count)")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(Color.secondary.opacity(0.08))
                        .clipShape(Capsule())
                    }
                }
            }
        }
        .padding(16)
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 22))
        .overlay(
            RoundedRectangle(cornerRadius: 22)
                .stroke(Color.primary.opacity(0.06), lineWidth: 1)
        )
    }

    private var graphWebView: some View {
        CytoscapeWebView(
            payload: GraphRenderPayload(nodes: viewModel.nodes, edges: viewModel.edges),
            layout: viewModel.selectedLayout,
            selectedNodeID: selectedNodeID,
            renderSignature: viewModel.graphSignature,
            onNodeTap: { nodeID in
                selectedNodeID = nodeID
            },
            onRenderStatus: { error in
                rendererError = error
            }
        )
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.thickMaterial)
        .overlay(alignment: .topLeading) {
            VStack(alignment: .leading, spacing: 4) {
                Text(selectedLayoutTitle)
                    .font(.caption.weight(.semibold))
                Text("Drag to pan, pinch to zoom, and tap nodes to focus the graph.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .padding(12)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 14))
            .padding(12)
        }
        .clipShape(RoundedRectangle(cornerRadius: 26))
        .overlay(
            RoundedRectangle(cornerRadius: 26)
                .stroke(Color.primary.opacity(0.06), lineWidth: 1)
        )
    }

    private var selectedLayoutTitle: String {
        viewModel.availableLayouts.first(where: { $0.id == viewModel.selectedLayout })?.title ?? "Graph"
    }

    private func selectedNodeInspector(_ node: GraphNode) -> some View {
        let connections = viewModel.strongestConnections(for: node)

        return VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 12) {
                RoundedRectangle(cornerRadius: node.type == "category" ? 10 : 14)
                    .fill(Color(hex: node.color))
                    .frame(width: node.type == "category" ? 18 : 14, height: node.type == "category" ? 18 : 14)
                    .padding(.top, 4)

                VStack(alignment: .leading, spacing: 4) {
                    Text(node.fullLabel)
                        .font(.headline)
                    Text(node.type == "category" ? "Category hub" : "Topic node")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Button("Clear") {
                    selectedNodeID = nil
                }
                .font(.caption.weight(.semibold))
            }

            HStack(spacing: 8) {
                metricBadge(title: metricTitle(for: node), value: metricValue(for: node))
                if !node.categories.isEmpty {
                    metricBadge(title: "Categories", value: node.categories.joined(separator: " • "))
                }
            }

            Text(nodeInsight(for: node))
                .font(.caption)
                .foregroundStyle(.secondary)

            if !connections.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Strongest Connections")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)

                    ForEach(connections) { connection in
                        HStack(spacing: 10) {
                            Circle()
                                .fill(Color(hex: connection.node.color))
                                .frame(width: 8, height: 8)

                            VStack(alignment: .leading, spacing: 2) {
                                Text(connection.node.fullLabel)
                                    .font(.subheadline.weight(.medium))
                                    .lineLimit(1)
                                Text(connection.node.type.capitalized)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }

                            Spacer()

                            Text(connection.weight.formatted(.number.precision(.fractionLength(0))))
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 22))
    }

    private func metricBadge(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption.weight(.semibold))
                .lineLimit(1)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(Color.secondary.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func layoutOptionCard(_ option: GraphLayoutOption, isSelected: Bool) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(option.title)
                .font(.subheadline.weight(.semibold))
            Text(option.subtitle)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.leading)
        }
        .frame(width: 150, alignment: .leading)
        .padding(12)
        .background(isSelected ? Color.accentPrimary.opacity(0.14) : Color.secondary.opacity(0.08))
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(isSelected ? Color.accentPrimary : Color.primary.opacity(0.08), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    private func metricTitle(for node: GraphNode) -> String {
        node.type == "category" ? "Events" : "Mentions"
    }

    private func metricValue(for node: GraphNode) -> String {
        "\(node.metricValue ?? 0)"
    }

    private func nodeInsight(for node: GraphNode) -> String {
        if node.type == "category" {
            return "This hub anchors the topics whose strongest link lands in \(node.label). Cross-lane edges show where those same ideas spill into the rest of your timeline."
        }

        if let primaryCategory = node.categories.first {
            return "This topic is strongest in \(primaryCategory), but the linked categories below show where it crosses into the rest of your recordings."
        }

        return "This topic node is positioned wherever its weighted links make the most sense for the active layout."
    }
}

// MARK: - Cytoscape WKWebView

struct CytoscapeWebView: PlatformViewRepresentable {
    let payload: GraphRenderPayload
    let layout: String
    let selectedNodeID: String?
    let renderSignature: String
    let onNodeTap: (String?) -> Void
    let onRenderStatus: (String?) -> Void

    final class Coordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
        var parent: CytoscapeWebView
        var isPageLoaded = false
        var lastRenderSignature = ""
        var lastLayout = ""
        var lastSelectedNodeID: String?
        private var pendingRender: RenderRequest?

        init(parent: CytoscapeWebView) {
            self.parent = parent
        }

        func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
            switch message.name {
            case "nodeTapped":
                let nodeID = (message.body as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
                parent.onNodeTap(nodeID?.isEmpty == false ? nodeID : nil)
            case "graphReady":
                parent.onRenderStatus(nil)
            case "graphError":
                if let text = message.body as? String, !text.isEmpty {
                    parent.onRenderStatus(text)
                } else {
                    parent.onRenderStatus("Graph renderer failed.")
                }
            default:
                break
            }
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            isPageLoaded = true
            flushPendingRender(in: webView)
        }

        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            parent.onRenderStatus(error.localizedDescription)
        }

        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            parent.onRenderStatus(error.localizedDescription)
        }

        func queueRender(payload: GraphRenderPayload, layout: String, selectedNodeID: String?, signature: String, in webView: WKWebView) {
            do {
                let payloadJSON = try Self.jsonLiteral(for: payload)
                let layoutJSON = try Self.jsonLiteral(for: layout)
                let selectedNodeJSON = try Self.jsonLiteral(for: selectedNodeID)

                let request = RenderRequest(
                    payloadJSON: payloadJSON,
                    layoutJSON: layoutJSON,
                    selectedNodeJSON: selectedNodeJSON,
                    signature: signature,
                    layout: layout,
                    selectedNodeID: selectedNodeID
                )

                parent.onRenderStatus(nil)

                if signature != lastRenderSignature || layout != lastLayout {
                    pendingRender = request
                    if isPageLoaded {
                        flushPendingRender(in: webView)
                    }
                } else if selectedNodeID != lastSelectedNodeID, isPageLoaded {
                    applySelection(selectedNodeJSON: selectedNodeJSON, selectedNodeID: selectedNodeID, in: webView)
                }
            } catch {
                parent.onRenderStatus(error.localizedDescription)
            }
        }

        func loadBaseDocument(in webView: WKWebView) {
            guard let url = Self.graphHTMLURL() else {
                parent.onRenderStatus("Bundled graph assets are missing from the app target.")
                return
            }

            isPageLoaded = false
            webView.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
        }

        private func flushPendingRender(in webView: WKWebView) {
            guard let request = pendingRender else { return }

            let js = "window.updateGraph(\(request.payloadJSON), \(request.layoutJSON), \(request.selectedNodeJSON));"
            webView.evaluateJavaScript(js) { [weak self] _, error in
                guard let self else { return }
                if let error {
                    self.parent.onRenderStatus(error.localizedDescription)
                    return
                }

                self.lastRenderSignature = request.signature
                self.lastLayout = request.layout
                self.lastSelectedNodeID = request.selectedNodeID
                self.pendingRender = nil
            }
        }

        private func applySelection(selectedNodeJSON: String, selectedNodeID: String?, in webView: WKWebView) {
            let js = "window.setSelectedNode(\(selectedNodeJSON));"
            webView.evaluateJavaScript(js) { [weak self] _, error in
                guard let self else { return }
                if let error {
                    self.parent.onRenderStatus(error.localizedDescription)
                    return
                }

                self.lastSelectedNodeID = selectedNodeID
            }
        }

        private static func graphHTMLURL() -> URL? {
            Bundle.main.url(forResource: "graph", withExtension: "html", subdirectory: "Resources")
                ?? Bundle.main.url(forResource: "graph", withExtension: "html")
        }

        private static func jsonLiteral<T: Encodable>(for value: T) throws -> String {
            let data = try JSONEncoder().encode(value)
            return String(decoding: data, as: UTF8.self)
        }
    }

    private struct RenderRequest {
        let payloadJSON: String
        let layoutJSON: String
        let selectedNodeJSON: String
        let signature: String
        let layout: String
        let selectedNodeID: String?
    }

    func makeCoordinator() -> Coordinator { Coordinator(parent: self) }

    #if canImport(UIKit)
    func makeUIView(context: Context) -> WKWebView {
        let userContentController = WKUserContentController()
        userContentController.add(context.coordinator, name: "nodeTapped")
        userContentController.add(context.coordinator, name: "graphReady")
        userContentController.add(context.coordinator, name: "graphError")

        let config = WKWebViewConfiguration()
        config.userContentController = userContentController
        let webView = WKWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = context.coordinator
        webView.isOpaque = false
        webView.backgroundColor = .clear
        webView.scrollView.isScrollEnabled = false
        context.coordinator.loadBaseDocument(in: webView)
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        context.coordinator.parent = self
        context.coordinator.queueRender(
            payload: payload,
            layout: layout,
            selectedNodeID: selectedNodeID,
            signature: renderSignature,
            in: webView
        )
    }

    static func dismantleUIView(_ webView: WKWebView, coordinator: Coordinator) {
        let controller = webView.configuration.userContentController
        controller.removeScriptMessageHandler(forName: "nodeTapped")
        controller.removeScriptMessageHandler(forName: "graphReady")
        controller.removeScriptMessageHandler(forName: "graphError")
    }
    #else
    func makeNSView(context: Context) -> WKWebView {
        let userContentController = WKUserContentController()
        userContentController.add(context.coordinator, name: "nodeTapped")
        userContentController.add(context.coordinator, name: "graphReady")
        userContentController.add(context.coordinator, name: "graphError")

        let config = WKWebViewConfiguration()
        config.userContentController = userContentController
        let webView = WKWebView(frame: .zero, configuration: config)
        webView.setValue(false, forKey: "drawsBackground")
        context.coordinator.loadBaseDocument(in: webView)
        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        context.coordinator.parent = self
        context.coordinator.queueRender(
            payload: payload,
            layout: layout,
            selectedNodeID: selectedNodeID,
            signature: renderSignature,
            in: webView
        )
    }

    static func dismantleNSView(_ webView: WKWebView, coordinator: Coordinator) {
        let controller = webView.configuration.userContentController
        controller.removeScriptMessageHandler(forName: "nodeTapped")
        controller.removeScriptMessageHandler(forName: "graphReady")
        controller.removeScriptMessageHandler(forName: "graphError")
    }
    #endif
}
