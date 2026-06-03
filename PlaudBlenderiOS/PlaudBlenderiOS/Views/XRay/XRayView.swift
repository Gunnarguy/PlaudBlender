import SwiftUI

struct XRayView: View {
    @Environment(APIClient.self) private var api
    @Environment(XRayViewModel.self) private var viewModel
    @State private var scope: TelemetryScope = .all
    @State private var expandedServerEvents: Set<Int> = []
    @State private var expandedNetworkEvents: Set<UUID> = []
    @State private var isShowingCostQuickLook = false
    @State private var isShowingExport = false
    @State private var exportData: String = ""

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                headerControls

                if !viewModel.throughputBuckets.isEmpty {
                    throughputSection
                }

                HStack {
                    Circle()
                        .fill(viewModel.isLiveUpdating ? .green : .red)
                        .frame(width: 6, height: 6)
                    Text(viewModel.isLiveUpdating ? "Live" : "Paused")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    if viewModel.isPipelineActive {
                        Text("PIPELINE")
                            .font(.caption2.weight(.bold))
                            .foregroundStyle(.green)
                    }
                    Spacer()
                    Text("session: \(sessionUptimeString)")
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                    Text("seq: \(viewModel.latestSeq)")
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                    Text(viewModel.liveTransport)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text("net: \(api.networkEvents.count)")
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal)
                .padding(.vertical, 4)

                if isLoadingTelemetry {
                    LoadingView(message: "Loading telemetry...")
                } else if isTelemetryEmpty {
                    EmptyStateView(
                        icon: "waveform.path.ecg",
                        title: "No Telemetry",
                        message: "Server events and client network traffic will appear here as the app runs."
                    )
                } else {
                    eventsList
                }
            }
            .navigationTitle("X-ray Monitor")
            .toolbar {
                ToolbarItem(placement: platformTrailingToolbarPlacement) {
                    Button {
                        isShowingCostQuickLook = true
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "dollarsign.circle.fill")
                                .font(.caption.weight(.semibold))
                            Text(costToolbarLabel)
                                .font(.caption.weight(.semibold))
                                .monospacedDigit()
                        }
                        .foregroundStyle(costToolbarTint)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(costToolbarTint.opacity(0.12))
                        .clipShape(Capsule())
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Open AI spend quick look")
                }

                ToolbarItem(placement: platformTrailingToolbarPlacement) {
                    HStack(spacing: 12) {
                        Button {
                            Task { await viewModel.clearEvents() }
                        } label: {
                            Image(systemName: "trash")
                        }
                        Button {
                            exportData = buildExportPayload()
                            isShowingExport = true
                        } label: {
                            Image(systemName: "square.and.arrow.up")
                        }
                    }
                }
            }
            .sheet(isPresented: $isShowingExport) {
                ExportTraceSheet(payload: exportData)
            }
            .sheet(isPresented: $isShowingCostQuickLook) {
                CostQuickLookSheet(
                    sessionCost: viewModel.sessionCost,
                    costHistory: viewModel.costHistory,
                    pricing: viewModel.modelPricing,
                    errorMessage: viewModel.error,
                    onRefresh: {
                        async let sessionLoad: Void = viewModel.loadSessionCost()
                        async let historyLoad: Void = viewModel.loadCostHistory()
                        async let pricingLoad: Void = viewModel.loadPricing()
                        _ = await (sessionLoad, historyLoad, pricingLoad)
                    }
                )
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
            }
            .refreshable { await viewModel.refresh() }
            .task { await viewModel.bootstrapIfNeeded() }
        }
    }

    private var throughputSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            telemetrySummaryRow

            HStack {
                Text("Server Pulse")
                    .font(.caption.weight(.semibold))
                Spacer()
                Text("peak \(viewModel.peakThroughput)/s")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            HStack(alignment: .bottom, spacing: 3) {
                ForEach(Array(viewModel.throughputBuckets.enumerated()), id: \.offset) { _, bucket in
                    RoundedRectangle(cornerRadius: 2)
                        .fill(bucket > 0 ? Color.accentPrimary : Color.secondary.opacity(0.25))
                        .frame(maxWidth: .infinity)
                        .frame(height: throughputHeight(for: bucket))
                }
            }
            .frame(height: 36, alignment: .bottom)
        }
        .padding(.horizontal)
        .padding(.top, 8)
        .padding(.bottom, 6)
        .background(.thinMaterial)
    }

    private var telemetrySummaryRow: some View {
        VStack(spacing: 6) {
            HStack(spacing: 8) {
                summaryPill(title: "Server", value: "\(viewModel.filteredEvents.count)", color: .blue)
                summaryPill(title: "Network", value: "\(api.networkEvents.count)", color: .green)
                summaryPill(title: "Errors", value: "\(networkErrorCount)", color: networkErrorCount > 0 ? .red : .secondary)
                summaryPill(title: "Traffic", value: byteString(for: totalTrafficBytes), color: .orange)
            }
            HStack(spacing: 8) {
                summaryPill(title: "Err Rate", value: errorRateString, color: errorRate > 5 ? .red : .secondary)
                summaryPill(title: "Avg Latency", value: avgLatencyString, color: avgLatency > 2000 ? .orange : .green)
                summaryPill(title: "P95 Latency", value: p95LatencyString, color: p95Latency > 5000 ? .red : .orange)
                summaryPill(title: "Uptime", value: uptimeString, color: .green)
            }
        }
    }

    private var headerControls: some View {
        VStack(spacing: 0) {
            Picker("Telemetry Scope", selection: $scope) {
                ForEach(TelemetryScope.allCases, id: \.self) { item in
                    Text(item.rawValue).tag(item)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)
            .padding(.top, 8)

            if scope != .network {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        filterChip("All", selected: viewModel.selectedSource == nil) {
                            viewModel.selectedSource = nil
                        }
                        ForEach(viewModel.sources, id: \.self) { source in
                            filterChip(source, selected: viewModel.selectedSource == source) {
                                viewModel.selectedSource = source
                            }
                        }
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 8)
                }
            }
        }
        .background(.ultraThinMaterial)
    }

    private var isLoadingTelemetry: Bool {
        viewModel.isLoading && viewModel.events.isEmpty && api.networkEvents.isEmpty
    }

    private var isTelemetryEmpty: Bool {
        filteredServerEvents.isEmpty && filteredNetworkEvents.isEmpty && (scope != .runs || viewModel.runs.isEmpty)
    }

    private var eventsList: some View {
        ScrollView {
            VStack(spacing: 12) {
                if scope != .network, !filteredServerEvents.isEmpty {
                    telemetrySection(title: "Server X-Ray", count: filteredServerEvents.count, color: .blue) {
                        LazyVStack(spacing: 8) {
                            ForEach(filteredServerEvents) { event in
                                serverEventRow(event)
                            }
                        }
                    }
                }

                if scope != .server, !filteredNetworkEvents.isEmpty {
                    telemetrySection(title: "Client Network", count: filteredNetworkEvents.count, color: .green) {
                        LazyVStack(spacing: 8) {
                            ForEach(filteredNetworkEvents) { event in
                                networkEventRow(event)
                            }
                        }
                    }
                }

                if (scope == .runs || scope == .all), !viewModel.runs.isEmpty {
                    telemetrySection(title: "Execution Runs", count: viewModel.runs.count, color: .purple) {
                        LazyVStack(spacing: 8) {
                            ForEach(viewModel.runs.prefix(12)) { run in
                                runCard(run)
                            }
                        }
                    }
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
        }
    }

    private func telemetrySection<Content: View>(title: String, count: Int, color: Color, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(title)
                    .font(.headline)
                Spacer()
                Text("\(count)")
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(color.opacity(0.15))
                    .clipShape(Capsule())
            }
            content()
        }
    }

    private var filteredServerEvents: [XRayEvent] {
        switch scope {
        case .all, .server:
            return viewModel.filteredEvents
        case .network, .runs:
            return []
        }
    }

    private var filteredNetworkEvents: [ClientNetworkEvent] {
        switch scope {
        case .all, .network:
            return api.networkEvents
        case .server, .runs:
            return []
        }
    }

    private var networkErrorCount: Int {
        api.networkEvents.filter(\.isError).count
    }

    private var totalTrafficBytes: Int {
        api.networkEvents.reduce(0) { partial, event in
            partial + event.requestBytes + event.responseBytes
        }
    }

    private var errorRate: Double {
        let total = api.networkEvents.count
        guard total > 0 else { return 0 }
        return Double(networkErrorCount) / Double(total) * 100
    }

    private var errorRateString: String {
        String(format: "%.1f%%", errorRate)
    }

    private var avgLatency: Double {
        let events = api.networkEvents.filter { $0.kind == "http" }
        guard !events.isEmpty else { return 0 }
        let total = events.reduce(0) { $0 + $1.durationMs }
        return Double(total) / Double(events.count)
    }

    private var avgLatencyString: String {
        avgLatency < 1000 ? String(format: "%.0fms", avgLatency) : String(format: "%.1fs", avgLatency / 1000)
    }

    private var p95Latency: Double {
        let sorted = api.networkEvents.filter { $0.kind == "http" }.map(\.durationMs).sorted()
        guard !sorted.isEmpty else { return 0 }
        let idx = Int(Double(sorted.count) * 0.95)
        return Double(sorted[min(idx, sorted.count - 1)])
    }

    private var p95LatencyString: String {
        p95Latency < 1000 ? String(format: "%.0fms", p95Latency) : String(format: "%.1fs", p95Latency / 1000)
    }

    private var uptimeString: String {
        let total = api.networkEvents.filter { $0.kind == "health" }.count
        guard total > 0 else { return "-" }
        let successes = api.networkEvents.filter { $0.kind == "health" && !$0.isError }.count
        return String(format: "%.0f%%", Double(successes) / Double(total) * 100)
    }

    private var maxNetworkLatency: Int {
        api.networkEvents.map(\.durationMs).max() ?? 1
    }

    private var sessionUptimeString: String {
        let elapsed = Date().timeIntervalSince(viewModel.sessionStartTime)
        return TimeInterval(elapsed).durationFormatted
    }

    private var costToolbarLabel: String {
        if let total = viewModel.sessionCost?.totalCostUsd {
            return formatCost(total)
        }
        if let total = viewModel.costHistory?.totalCostUsd {
            return formatCost(total)
        }
        return "Costs"
    }

    private var costToolbarTint: Color {
        let amount = viewModel.sessionCost?.totalCostUsd ?? viewModel.costHistory?.totalCostUsd ?? 0
        switch amount {
        case ..<0.01:
            return .secondary
        case ..<0.1:
            return .accentGreen
        case ..<1:
            return .accentOrange
        default:
            return .accentRed
        }
    }

    private func serverEventRow(_ event: XRayEvent) -> some View {
        DisclosureGroup(
            isExpanded: binding(for: event.id, in: $expandedServerEvents)
        ) {
            VStack(alignment: .leading, spacing: 8) {
                if let detail = event.detail, !detail.isEmpty {
                    detailBlock(title: "Detail", value: detail)
                }
                if !eventTraceFacts(event).isEmpty {
                    detailBlock(title: "Trace", value: eventTraceFacts(event).joined(separator: "\n"))
                }
                HStack(spacing: 10) {
                    statText("Level", event.level.capitalized)
                    statText("Seq", "\(event.seq)")
                    if let ms = event.durationMs {
                        statText("Duration", String(format: "%.0fms", ms))
                    }
                }
            }
            .padding(.top, 6)
        } label: {
            HStack(alignment: .top, spacing: 8) {
                Text(event.source)
                    .font(.caption2)
                    .fontWeight(.medium)
                    .foregroundStyle(sourceColor(event.source))
                    .frame(width: 56, alignment: .leading)

                VStack(alignment: .leading, spacing: 3) {
                    Text(event.message)
                        .font(.caption)
                        .lineLimit(3)
                    HStack(spacing: 8) {
                        Text(event.op)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        if let provider = event.provider, !provider.isEmpty {
                            Text(provider.uppercased())
                                .font(.caption2.weight(.semibold))
                                .foregroundStyle(sourceColor(provider))
                        }
                        if let model = event.model, !model.isEmpty {
                            Text(model)
                                .font(.caption2)
                                .foregroundStyle(.tertiary)
                                .lineLimit(1)
                        }
                        Spacer()
                        Text(event.date.timeString)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }
            }
        }
        .padding(10)
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func networkEventRow(_ event: ClientNetworkEvent) -> some View {
        DisclosureGroup(
            isExpanded: binding(for: event.id, in: $expandedNetworkEvents)
        ) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    statText("Request ID", event.requestId)
                    Spacer()
                }
                detailBlock(title: "URL", value: event.url)
                if !event.requestHeaders.isEmpty {
                    detailBlock(title: "Request Headers", value: prettyPrintedDictionary(event.requestHeaders))
                }
                if let requestPreview = event.requestPreview {
                    detailBlock(title: "Request Body", value: requestPreview)
                }
                if !event.responseHeaders.isEmpty {
                    detailBlock(title: "Response Headers", value: prettyPrintedDictionary(event.responseHeaders))
                }
                if let responsePreview = event.responsePreview {
                    detailBlock(title: "Response Body", value: responsePreview)
                }
                if let errorMessage = event.errorMessage, !errorMessage.isEmpty {
                    detailBlock(title: "Error", value: errorMessage)
                }
            }
            .padding(.top, 6)
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                HStack(alignment: .top, spacing: 10) {
                    VStack(alignment: .leading, spacing: 3) {
                        HStack(spacing: 8) {
                            Text(event.method)
                                .font(.caption2.weight(.bold))
                                .foregroundStyle(event.isError ? .red : .green)
                            Text(event.path)
                                .font(.caption)
                                .lineLimit(1)
                        }

                        HStack(spacing: 10) {
                            statText("Status", event.statusCode.map(String.init) ?? "ERR")
                            statText("Time", "\(event.durationMs)ms")
                            statText("Out", byteString(for: event.requestBytes))
                            statText("In", byteString(for: event.responseBytes))
                        }
                    }

                    Spacer()

                    Text(event.timestamp.timeString)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                latencyBar(durationMs: event.durationMs, isError: event.isError)
            }
        }
        .padding(10)
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func runCard(_ run: TraceRun) -> some View {
        let runSpans = viewModel.spans.filter { $0.runId == run.runId }
        let failed = run.status.lowercased() == "failed"
        let running = run.status.lowercased() == "running"

        return VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: running ? "dot.radiowaves.left.and.right" : (failed ? "xmark.octagon.fill" : "checkmark.circle.fill"))
                    .foregroundStyle(running ? .green : (failed ? .red : .blue))
                VStack(alignment: .leading, spacing: 3) {
                    Text(run.title ?? "Chronos Run")
                        .font(.caption.weight(.semibold))
                    Text(run.runId)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    Text(run.status.capitalized)
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(running ? .green : (failed ? .red : .secondary))
                    if let duration = run.durationMs {
                        Text(durationString(duration))
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            }

            HStack(spacing: 8) {
                if let trigger = run.trigger { traceChip(trigger, color: .blue) }
                if let source = run.source { traceChip(source, color: sourceColor(source)) }
                traceChip("\(runSpans.count) spans", color: .purple)
            }

            if let error = run.errorMessage, !error.isEmpty {
                Text(error)
                    .font(.caption2)
                    .foregroundStyle(.red)
                    .lineLimit(2)
            }

            ForEach(runSpans.prefix(5)) { span in
                spanMiniRow(span)
            }
        }
        .padding(10)
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func spanMiniRow(_ span: TraceSpan) -> some View {
        HStack(spacing: 6) {
            Circle()
                .fill(span.status.lowercased() == "failed" ? .red : (span.status.lowercased() == "running" ? .green : .secondary))
                .frame(width: 6, height: 6)
            Text(span.stage ?? span.source ?? "span")
                .font(.caption2.weight(.semibold))
                .foregroundStyle(sourceColor(span.source ?? span.provider ?? ""))
                .frame(width: 58, alignment: .leading)
            Text(span.message ?? span.operation)
                .font(.caption2)
                .lineLimit(1)
            Spacer()
            if let provider = span.provider {
                Text(provider.uppercased())
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(sourceColor(provider))
            }
            if let duration = span.durationMs {
                Text(durationString(duration))
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(6)
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private func traceChip(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .foregroundStyle(color)
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(color.opacity(0.12))
            .clipShape(Capsule())
    }

    private func filterChip(_ label: String, selected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(label.capitalized)
                .font(.caption2)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(selected ? Color.accentPrimary.opacity(0.2) : Color.clear)
                .clipShape(Capsule())
                .overlay(Capsule().stroke(Color.accentPrimary.opacity(0.3), lineWidth: 0.5))
        }
        .buttonStyle(.plain)
    }

    private func sourceColor(_ source: String) -> Color {
        switch source {
        case "gemini": return .cyan
        case "openai": return .purple
        case "local", "ollama", "llama_cpp": return .mint
        case "qdrant": return .orange
        case "graph": return .green
        case "search": return .blue
        case "pipeline": return .purple
        case "ingest": return .yellow
        case "embed": return .pink
        default: return .secondary
        }
    }

    private func throughputHeight(for bucket: Int) -> CGFloat {
        let peak = max(viewModel.peakThroughput, 1)
        let normalized = CGFloat(bucket) / CGFloat(peak)
        return max(4, normalized * 32)
    }

    private func summaryPill(title: String, value: String, color: Color) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.caption.weight(.bold))
                .foregroundStyle(color)
                .monospacedDigit()
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 6)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private func byteString(for bytes: Int) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }

    private func formatCost(_ usd: Double) -> String {
        if usd == 0 {
            return "$0.00"
        }
        if usd < 0.01 {
            return String(format: "$%.4f", usd)
        }
        if usd < 1 {
            return String(format: "$%.3f", usd)
        }
        return String(format: "$%.2f", usd)
    }

    private func durationString(_ ms: Double) -> String {
        ms < 1000 ? String(format: "%.0fms", ms) : TimeInterval(ms / 1000).durationFormatted
    }

    private func statText(_ title: String, _ value: String) -> some View {
        HStack(spacing: 4) {
            Text(title)
                .foregroundStyle(.secondary)
            Text(value)
                .monospacedDigit()
        }
        .font(.caption2)
    }

    private func detailBlock(title: String, value: String) -> some View {
        let displayValue = prettyPrintedIfJSON(value)

        return VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(displayValue)
                .font(.caption2.monospaced())
                .textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(8)
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private func eventTraceFacts(_ event: XRayEvent) -> [String] {
        var facts: [String] = []
        if let runId = event.runId { facts.append("run_id: \(runId)") }
        if let spanId = event.spanId { facts.append("span_id: \(spanId)") }
        if let parent = event.parentSpanId { facts.append("parent_span_id: \(parent)") }
        if let recordingId = event.recordingId { facts.append("recording_id: \(recordingId)") }
        if let eventId = event.eventId { facts.append("event_id: \(eventId)") }
        if let stage = event.stage { facts.append("stage: \(stage)") }
        if let provider = event.provider { facts.append("provider: \(provider)") }
        if let model = event.model { facts.append("model: \(model)") }
        if let status = event.status { facts.append("status: \(status)") }
        if let input = event.inputTokens { facts.append("input_tokens: \(input)") }
        if let output = event.outputTokens { facts.append("output_tokens: \(output)") }
        if let cost = event.costUsd { facts.append("cost_usd: \(formatCost(cost))") }
        if let requestId = event.requestId { facts.append("request_id: \(requestId)") }
        return facts
    }

    private func binding<ID: Hashable>(for id: ID, in set: Binding<Set<ID>>) -> Binding<Bool> {
        Binding(
            get: { set.wrappedValue.contains(id) },
            set: { isExpanded in
                if isExpanded {
                    set.wrappedValue.insert(id)
                } else {
                    set.wrappedValue.remove(id)
                }
            }
        )
    }

    private func prettyPrintedIfJSON(_ value: String) -> String {
        guard let data = value.data(using: .utf8) else { return value }

        do {
            let object = try JSONSerialization.jsonObject(with: data)
            let prettyData = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
            return String(decoding: prettyData, as: UTF8.self)
        } catch {
            return value
        }
    }

    private func prettyPrintedDictionary(_ dictionary: [String: String]) -> String {
        do {
            let data = try JSONSerialization.data(withJSONObject: dictionary, options: [.prettyPrinted, .sortedKeys])
            return String(decoding: data, as: UTF8.self)
        } catch {
            return dictionary
                .sorted(by: { $0.key < $1.key })
                .map { "\($0.key): \($0.value)" }
                .joined(separator: "\n")
        }
    }

    private func latencyBar(durationMs: Int, isError: Bool) -> some View {
        let peak = max(maxNetworkLatency, 1)
        let fraction = CGFloat(durationMs) / CGFloat(peak)
        let color: Color = isError ? .red : (durationMs < 500 ? .green : (durationMs < 2000 ? .yellow : .orange))

        return GeometryReader { geo in
            RoundedRectangle(cornerRadius: 2)
                .fill(color.opacity(0.6))
                .frame(width: max(4, geo.size.width * fraction), height: 4)
        }
        .frame(height: 4)
    }

    private func buildExportPayload() -> String {
        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]

        var trace: [[String: Any]] = []

        for event in viewModel.events.prefix(200) {
            var entry: [String: Any] = [
                "type": "server",
                "seq": event.seq,
                "ts": iso.string(from: event.date),
                "source": event.source,
                "op": event.op,
                "message": event.message,
                "level": event.level
            ]
            if let ms = event.durationMs { entry["duration_ms"] = ms }
            if let detail = event.detail { entry["detail"] = detail }
            trace.append(entry)
        }

        for event in api.networkEvents.prefix(200) {
            var entry: [String: Any] = [
                "type": "network",
                "ts": iso.string(from: event.timestamp),
                "kind": event.kind,
                "method": event.method,
                "path": event.path,
                "status": event.statusCode ?? -1,
                "duration_ms": event.durationMs,
                "request_bytes": event.requestBytes,
                "response_bytes": event.responseBytes,
                "request_id": event.requestId,
                "is_error": event.isError
            ]
            if let err = event.errorMessage { entry["error"] = err }
            trace.append(entry)
        }

        let wrapper: [String: Any] = [
            "exported_at": iso.string(from: Date()),
            "server_seq": viewModel.latestSeq,
            "event_count": trace.count,
            "events": trace
        ]

        guard let data = try? JSONSerialization.data(withJSONObject: wrapper, options: [.prettyPrinted, .sortedKeys]) else {
            return "{\"error\": \"serialization failed\"}"
        }
        return String(decoding: data, as: UTF8.self)
    }
}

private struct ExportTraceSheet: View {
    let payload: String
    @Environment(\.dismiss) private var dismiss
    @State private var copied = false

    var body: some View {
        NavigationStack {
            ScrollView {
                Text(payload)
                    .font(.caption2.monospaced())
                    .textSelection(.enabled)
                    .padding()
            }
            .navigationTitle("Trace Export")
            .platformNavigationBarTitleDisplayModeInline()
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
                ToolbarItemGroup(placement: platformTrailingToolbarPlacement) {
                    Button {
                        copyToClipboard(payload)
                        copied = true
                        Haptics.success()
                        Task {
                            try? await Task.sleep(nanoseconds: 1_500_000_000)
                            copied = false
                        }
                    } label: {
                        Label(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc")
                    }
                    ShareLink(item: payload)
                }
            }
        }
    }
}

private enum TelemetryScope: String, CaseIterable {
    case all = "All"
    case runs = "Runs"
    case server = "Server"
    case network = "Network"
}
