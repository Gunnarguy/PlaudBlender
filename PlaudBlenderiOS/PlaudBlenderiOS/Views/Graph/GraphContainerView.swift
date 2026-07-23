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
            ZStack {
                Color(hex: "070a12")
                    .ignoresSafeArea()

                VStack(spacing: 0) {
                    controlsCard
                        .padding([.horizontal, .top])

                    if viewModel.isLoading {
                        Spacer()
                        LoadingView(message: "Building knowledge neural map...")
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
                            .padding(.horizontal, 8)
                            .padding(.top, 12)
                            .padding(.bottom, 8)
                    }
                }
            }
            .navigationTitle("Knowledge Graph")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.visible, for: .navigationBar)
            .toolbarBackground(Color(hex: "070a12"), for: .navigationBar)
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
        @Bindable var vm = viewModel
        return VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .center) {
                HStack(spacing: 8) {
                    Image(systemName: "cpu.fill")
                        .font(.title3)
                        .foregroundStyle(.cyan)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Knowledge Neural Map")
                            .font(.headline.weight(.bold))
                        Text("Showing \(viewModel.displayedNodes.count) key nodes in 3D space")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }

                Spacer()

                Button {
                    rendererError = nil
                    Task { await viewModel.refresh() }
                } label: {
                    Image(systemName: "arrow.clockwise")
                        .font(.caption.weight(.bold))
                        .padding(8)
                        .background(Color.white.opacity(0.08))
                        .clipShape(Circle())
                }
                .buttonStyle(.plain)
            }

            // Search Bar
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextField("Search 3D concepts & categories...", text: $vm.searchText)
                    .font(.caption)
                if !viewModel.searchText.isEmpty {
                    Button {
                        viewModel.searchText = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(Color.white.opacity(0.06))
            .clipShape(RoundedRectangle(cornerRadius: 10))

            // Density Limit & Category Filters Row
            HStack(spacing: 8) {
                // Density Limit Menu
                Menu {
                    ForEach(viewModel.availableDensityLimits, id: \.self) { limit in
                        Button {
                            viewModel.selectedDensityLimit = limit
                        } label: {
                            HStack {
                                Text(limit >= 100 ? "All Nodes (\(viewModel.nodes.count))" : "Top \(limit) Key Concepts")
                                if viewModel.selectedDensityLimit == limit {
                                    Image(systemName: "checkmark")
                                }
                            }
                        }
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "slider.horizontal.3")
                            .font(.caption2)
                        Text(viewModel.selectedDensityLimit >= 100 ? "All Nodes" : "Top \(viewModel.selectedDensityLimit)")
                            .font(.caption.weight(.bold))
                        Image(systemName: "chevron.down")
                            .font(.caption2)
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Color.cyan.opacity(0.15))
                    .foregroundStyle(.cyan)
                    .clipShape(Capsule())
                }

                // Category Chips
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(viewModel.availableCategories, id: \.self) { cat in
                            let isSelected = viewModel.selectedCategoryFilter == cat
                            Button {
                                viewModel.selectedCategoryFilter = cat
                            } label: {
                                Text(cat.capitalized)
                                    .font(.caption2.weight(.bold))
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 6)
                                    .background(isSelected ? Color.cyan : Color.white.opacity(0.06))
                                    .foregroundStyle(isSelected ? .black : .secondary)
                                    .clipShape(Capsule())
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }

            // 3D View Mode Selector
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
        }
        .padding(14)
        .background(Color(hex: "0f172a").opacity(0.85))
        .clipShape(RoundedRectangle(cornerRadius: 20))
        .overlay(
            RoundedRectangle(cornerRadius: 20)
                .stroke(Color.white.opacity(0.08), lineWidth: 1)
        )
    }

    private var graphWebView: some View {
        CytoscapeWebView(
            payload: GraphRenderPayload(nodes: viewModel.displayedNodes, edges: viewModel.displayedEdges),
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
                Text(interactionHint)
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
        viewModel.availableLayouts.first(where: { $0.id == viewModel.selectedLayout })?.title ?? "3D Graph"
    }

    private var interactionHint: String {
        switch viewModel.selectedLayout {
        case GraphLayoutOption.constellation3d.id:
            return "3D Neural Constellation: Touch drag to orbit 360°, pinch to zoom, tap any node sphere."
        case GraphLayoutOption.vectorSpace3d.id:
            return "Qdrant Vector Embedding Space: Semantic similarity distance projection in 3D."
        case GraphLayoutOption.isometric25d.id:
            return "2.5D Isometric Matrix: Elevated category grid with 3D frequency pillars."
        case GraphLayoutOption.galaxyOrbit3d.id:
            return "3D Galaxy Orbit: Category star hubs with orbiting topic nodes."
        case GraphLayoutOption.volumetric3d.id:
            return "3D Volumetric Matrix: Translucent 3D frequency & sentiment bars."
        default:
            return "Touch drag to rotate space 360°, pinch to zoom."
        }
    }

    private func selectedNodeInspector(_ node: GraphNode) -> some View {
        let connections = viewModel.strongestConnections(for: node)
        let categoryTopics = node.type == "category" ? viewModel.topics(in: node) : []

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

            if !categoryTopics.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Top Topics in \(node.label)")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)

                    topicPills(categoryTopics)
                }
            }

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

    private func topicPills(_ topics: [GraphNode]) -> some View {
        FlowLayout(spacing: 6) {
            ForEach(topics) { topic in
                Button {
                    selectedNodeID = topic.id
                } label: {
                    HStack(spacing: 5) {
                        Circle()
                            .fill(Color(hex: topic.color))
                            .frame(width: 6, height: 6)
                        Text(topic.fullLabel)
                            .lineLimit(1)
                        Text("\(topic.metricValue ?? 0)")
                            .foregroundStyle(.secondary)
                    }
                    .font(.caption.weight(.medium))
                    .padding(.horizontal, 9)
                    .padding(.vertical, 6)
                    .background(Color.secondary.opacity(0.08))
                    .clipShape(Capsule())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("\(topic.fullLabel), \(topic.metricValue ?? 0) mentions")
            }
        }
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
        HStack(spacing: 8) {
            Image(systemName: option.icon)
                .font(.subheadline.weight(.bold))
                .foregroundStyle(isSelected ? Color.cyan : Color.secondary)

            VStack(alignment: .leading, spacing: 2) {
                Text(option.title)
                    .font(.caption.weight(.bold))
                    .foregroundStyle(isSelected ? .white : .secondary)
                Text(option.subtitle)
                    .font(.system(size: 9))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(isSelected ? Color.cyan.opacity(0.18) : Color.white.opacity(0.04))
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .stroke(isSelected ? Color.cyan.opacity(0.5) : Color.white.opacity(0.08), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private func metricTitle(for node: GraphNode) -> String {
        node.type == "category" ? "Events" : "Mentions"
    }

    private func metricValue(for node: GraphNode) -> String {
        "\(node.metricValue ?? 0)"
    }

    private func nodeInsight(for node: GraphNode) -> String {
        if node.type == "category" {
            return "This category groups the topics that most often show up in \(node.label). Its topic list below is ordered by mentions."
        }

        if let primaryCategory = node.categories.first {
            return "This topic is strongest in \(primaryCategory), but the linked categories below show where it crosses into the rest of your recordings."
        }

        return "This topic is part of the map because it has meaningful relationships in your recordings."
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
