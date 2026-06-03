import Foundation
import Observation

@MainActor
@Observable
final class XRayViewModel {
    var events: [XRayEvent] = []
    var throughputBuckets: [Int] = []
    var latestSeq: Int = 0
    var selectedSource: String?
    var sessionCost: SessionCost?
    var costHistory: CostHistory?
    var modelPricing: ModelPricing?
    var runs: [TraceRun] = []
    var spans: [TraceSpan] = []
    var selectedRunId: String?
    var liveTransport: String = "Polling"
    var isLoading = false
    var isLiveUpdating = false
    var hasBootstrapped = false
    var error: String?
    var isPipelineActive = false
    let sessionStartTime = Date()

    private let api: APIClient
    @ObservationIgnored private var liveTask: Task<Void, Never>?
    @ObservationIgnored private let initialHistoryLimit = 250
    @ObservationIgnored private let maxEventCount = 800
    @ObservationIgnored private var quietPollCount = 0
    @ObservationIgnored private var livePollIteration = 0
    @ObservationIgnored private var webSocket = WebSocketManager()

    let sources = [
        "ingest", "gemini", "openai", "local", "embed", "qdrant", "graph",
        "search", "data", "nav", "pipeline", "detail", "day", "sync"
    ]

    init(api: APIClient) {
        self.api = api
    }

    var filteredEvents: [XRayEvent] {
        guard let source = selectedSource else { return events }
        return events.filter { $0.source == source }
    }

    var syncEvents: [XRayEvent] {
        events.filter { event in
            ["pipeline", "ingest", "gemini", "local", "embed", "qdrant", "sync", "data"].contains(event.source)
                || event.op.localizedCaseInsensitiveContains("notion")
        }
    }

    var notionEvents: [XRayEvent] {
        events.filter { event in
            event.op.localizedCaseInsensitiveContains("notion")
                || event.message.localizedCaseInsensitiveContains("notion")
                || event.detail?.localizedCaseInsensitiveContains("notion") == true
        }
    }

    var activityHighlights: [XRayEvent] {
        Array(
            events
                .filter { event in
                    ["pipeline", "ingest", "gemini", "local", "embed", "qdrant", "sync", "data"].contains(event.source)
                        || event.op.localizedCaseInsensitiveContains("notion")
                        || event.message.localizedCaseInsensitiveContains("notion")
                }
                .prefix(8)
        )
    }

    var latestActivity: XRayEvent? {
        activityHighlights.first
    }

    var latestRun: TraceRun? {
        runs.first
    }

    var selectedRunSpans: [TraceSpan] {
        guard let selectedRunId else { return spans }
        return spans.filter { $0.runId == selectedRunId }
    }

    var activeTraceSpans: [TraceSpan] {
        spans.filter { $0.status.lowercased() == "running" }
    }

    var peakThroughput: Int {
        throughputBuckets.max() ?? 0
    }

    func timelineEvents(for recordingToken: String?, limit: Int = 8) -> [XRayEvent] {
        let trimmedToken = recordingToken?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

        let relevantEvents = syncEvents.filter { event in
            guard !trimmedToken.isEmpty else { return true }
            return event.detail?.localizedCaseInsensitiveContains(trimmedToken) == true
                || event.message.localizedCaseInsensitiveContains(trimmedToken)
        }

        if !relevantEvents.isEmpty {
            return Array(relevantEvents.prefix(limit))
        }

        return Array(syncEvents.prefix(limit))
    }

    deinit {
        liveTask?.cancel()
    }

    func loadEvents(sinceSeq: Int? = nil, limit: Int = 100) async {
        if sinceSeq == nil {
            isLoading = true
            error = nil
        }
        do {
            var query = ["limit": "\(limit)"]
            if let sinceSeq { query["since_seq"] = "\(sinceSeq)" }
            let response: XRayEventsResponse = try await api.get("/api/xray/events", query: query)
            if sinceSeq != nil {
                let newEvents = response.events.filter { event in
                    !events.contains(where: { $0.seq == event.seq })
                }
                events.insert(contentsOf: newEvents, at: 0)
            } else {
                events = response.events
            }
            if events.count > maxEventCount {
                events = Array(events.prefix(maxEventCount))
            }
            latestSeq = response.latestSeq
        } catch {
            self.error = error.localizedDescription
        }
        if sinceSeq == nil {
            isLoading = false
        }
    }

    func bootstrapIfNeeded() async {
        guard !hasBootstrapped else { return }
        hasBootstrapped = true
        async let eventsLoad: Void = loadEvents(limit: initialHistoryLimit)
        async let throughputLoad: Void = loadThroughput()
        async let costLoad: Void = refreshCostTelemetry(includePricing: true)
        async let traceLoad: Void = refreshTraceTelemetry()
        _ = await (eventsLoad, throughputLoad, costLoad, traceLoad)
        startLiveUpdates()
    }

    func startLiveUpdates() {
        guard liveTask == nil else { return }
        isLiveUpdating = true
        quietPollCount = 0
        connectWebSocketIfNeeded()
        liveTask = Task { [weak self] in
            guard let self else { return }
            await self.liveLoop()
        }
    }

    func stopLiveUpdates() {
        liveTask?.cancel()
        liveTask = nil
        webSocket.disconnect()
        liveTransport = "Polling"
        isLiveUpdating = false
    }

    func refresh() async {
        async let eventsLoad: Void = loadEvents()
        async let throughputLoad: Void = loadThroughput()
        async let costLoad: Void = refreshCostTelemetry(includePricing: modelPricing == nil)
        async let traceLoad: Void = refreshTraceTelemetry()
        _ = await (eventsLoad, throughputLoad, costLoad, traceLoad)
    }

    func loadRuns(limit: Int = 20) async {
        do {
            runs = try await api.get("/api/xray/runs", query: ["limit": "\(limit)"])
            if selectedRunId == nil {
                selectedRunId = runs.first?.runId
            }
        } catch is CancellationError {
            return
        } catch let urlError as URLError where urlError.code == .cancelled {
            return
        } catch {
            // Runs are additive telemetry; don't break the main X-Ray stream.
        }
    }

    func loadSpans(runId: String? = nil, limit: Int = 120) async {
        do {
            var query = ["limit": "\(limit)"]
            if let runId { query["run_id"] = runId }
            spans = try await api.get("/api/xray/spans", query: query)
        } catch is CancellationError {
            return
        } catch let urlError as URLError where urlError.code == .cancelled {
            return
        } catch {
            // Non-fatal; old backends won't have this endpoint.
        }
    }

    func refreshTraceTelemetry() async {
        await loadRuns()
        await loadSpans(runId: selectedRunId)
    }

    func loadSessionCost() async {
        do {
            sessionCost = try await api.get("/api/costs/session")
        } catch is CancellationError {
            return
        } catch let urlError as URLError where urlError.code == .cancelled {
            return
        } catch {
            if sessionCost == nil && self.error == nil {
                self.error = error.localizedDescription
            }
        }
    }

    func loadCostHistory(days: Int = 30) async {
        do {
            costHistory = try await api.get("/api/costs/history", query: ["days": "\(days)"])
        } catch is CancellationError {
            return
        } catch let urlError as URLError where urlError.code == .cancelled {
            return
        } catch {
            if costHistory == nil && self.error == nil {
                self.error = error.localizedDescription
            }
        }
    }

    func loadPricing() async {
        do {
            modelPricing = try await api.get("/api/costs/pricing")
        } catch is CancellationError {
            return
        } catch let urlError as URLError where urlError.code == .cancelled {
            return
        } catch {
            if modelPricing == nil && self.error == nil {
                self.error = error.localizedDescription
            }
        }
    }

    func clearEvents() async {
        do {
            let _: SuccessResponse = try await api.post("/api/xray/clear")
            events = []
            latestSeq = 0
            throughputBuckets = []
        } catch {
            self.error = error.localizedDescription
        }
    }

    func loadThroughput() async {
        do {
            let response: XRayThroughput = try await api.get("/api/xray/throughput", query: ["buckets": "24"])
            throughputBuckets = response.buckets
        } catch {
            self.error = error.localizedDescription
        }
    }

    func refreshCostTelemetry(includePricing: Bool = false) async {
        async let sessionLoad: Void = loadSessionCost()
        async let historyLoad: Void = loadCostHistory()

        if includePricing || modelPricing == nil {
            async let pricingLoad: Void = loadPricing()
            _ = await (sessionLoad, historyLoad, pricingLoad)
        } else {
            _ = await (sessionLoad, historyLoad)
        }
    }

    private func connectWebSocketIfNeeded() {
        guard !webSocket.isConnected else { return }
        webSocket.onEvent = { [weak self] event in
            self?.mergeLiveEvent(event)
        }
        webSocket.connect(baseURL: api.resolvedServerURL, token: api.authManager.getToken())
        liveTransport = "WebSocket + polling fallback"
    }

    private func mergeLiveEvent(_ event: XRayEvent) {
        guard !events.contains(where: { $0.seq == event.seq }) else { return }
        events.insert(event, at: 0)
        if events.count > maxEventCount {
            events = Array(events.prefix(maxEventCount))
        }
        latestSeq = max(latestSeq, event.seq)
        quietPollCount = 0
    }

    private func loadIncrementalEvents(limit: Int = 80) async -> Int {
        do {
            let previousLatestSeq = latestSeq
            let query = [
                "limit": "\(limit)",
                "since_seq": "\(latestSeq)"
            ]
            let response: XRayEventsResponse = try await api.get("/api/xray/events", query: query)

            let newEvents = response.events.filter { event in
                !events.contains(where: { $0.seq == event.seq })
            }

            if !newEvents.isEmpty {
                for event in newEvents.reversed() {
                    mergeLiveEvent(event)
                }
            }

            latestSeq = max(previousLatestSeq, response.latestSeq)
            return newEvents.count
        } catch {
            self.error = error.localizedDescription
            return 0
        }
    }

    private func nextLivePollDelayNanoseconds() -> UInt64 {
        if isPipelineActive {
            switch quietPollCount {
            case 0: return 300_000_000
            case 1: return 500_000_000
            default: return 1_000_000_000
            }
        }
        switch quietPollCount {
        case 0:
            return 500_000_000
        case 1:
            return 1_000_000_000
        case 2:
            return 2_000_000_000
        default:
            return 4_000_000_000
        }
    }

    private func liveLoop() async {
        defer {
            liveTask = nil
            isLiveUpdating = false
        }

        while !Task.isCancelled {
            livePollIteration += 1
            let newEventCount = await loadIncrementalEvents(limit: 80)

            if newEventCount > 0 || livePollIteration % 4 == 0 {
                await loadThroughput()
            }

            if newEventCount > 0 || livePollIteration % 5 == 0 {
                await loadSessionCost()
            }

            if newEventCount > 0 || livePollIteration % 6 == 0 {
                await refreshTraceTelemetry()
            }

            if newEventCount > 0 || livePollIteration % 15 == 0 {
                await loadCostHistory()
            }

            if newEventCount > 0 {
                quietPollCount = 0
            } else {
                quietPollCount = min(quietPollCount + 1, 3)
            }

            if Task.isCancelled {
                break
            }

            try? await Task.sleep(nanoseconds: nextLivePollDelayNanoseconds())
        }
    }
}
