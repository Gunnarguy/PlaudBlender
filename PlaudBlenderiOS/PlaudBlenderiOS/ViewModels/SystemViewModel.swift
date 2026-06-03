import Foundation
import Observation

struct SystemServiceEntry: Identifiable, Sendable {
    let id: String
    let title: String
    let state: String
    let isHealthy: Bool
    let detail: String?
    let footnote: String?
}

struct SystemPortEntry: Identifiable, Sendable {
    let id: String
    let name: String
    let summary: String
    let isReachable: Bool?
    let detail: String?
}

struct SystemSignalEntry: Identifiable, Sendable {
    let id: String
    let source: String
    let level: String
    let title: String
    let message: String
    let footnote: String?
    let timestamp: Date?
}

struct SystemRuntimeManagerInfo: Sendable {
    let name: String
    let mode: String
    let state: String
    let isHealthy: Bool
    let detail: String
    let watchdog: String
    let verified: String
    let version: String
}

@MainActor
@Observable
final class SystemViewModel {
    var systemStatus: SystemStatus?
    var runtimeSnapshot: SystemRuntimeSnapshot?
    var isLoading = false
    var hasBootstrapped = false
    var supportsRuntimeEndpoint = true
    var supportsXRayEndpoint = true
    var error: String?
    var runtimeNotice: String?
    var lastUpdated: Date?
    var recentXRayEvents: [XRayEvent] = []

    private let api: APIClient

    init(api: APIClient) {
        self.api = api
    }

    var healthyServiceCount: Int {
        guard let status = systemStatus else { return 0 }

        var count = 0
        if status.database?.ok == true { count += 1 }
        if status.qdrant?.ok == true { count += 1 }
        if status.gemini?.isSafeOrOperational == true { count += 1 }
        if status.openai?.isSafeOrOperational == true { count += 1 }
        if status.plaud?.isUp == true { count += 1 }
        if status.notion?.isUp == true { count += 1 }
        if status.localLLM?.isUp == true || status.localLLM?.enabled != true { count += 1 }
        return count
    }

    var totalServiceCount: Int {
        // Chronos now exposes local inference as its own explicit row. If an older
        // backend does not report it yet, keep the historical six-service denominator.
        systemStatus?.localLLM == nil ? 6 : 7
    }

    var runtimeStateText: String {
        if !api.isServerReachable {
            return "Offline"
        }
        return runtimeIsHealthy ? "Healthy" : "Needs Attention"
    }

    var runtimeIsHealthy: Bool {
        if let ok = runtimeSnapshot?.runtimeHealth?.ok {
            return ok
        }

        guard let status = systemStatus else {
            return false
        }

        let databaseReady = status.database?.ok == true
        let qdrantReady = status.qdrant?.ok == true
        let aiReady = status.gemini?.isUp == true || status.localLLM?.isUp == true || status.openai?.isOperational == true
        return databaseReady && qdrantReady && aiReady
    }

    var runtimeSummary: String {
        if let summary = runtimeSnapshot?.runtimeHealth?.summary, !summary.isEmpty {
            return summary
        }

        if let detail = runtimeSnapshot?.runtimeHealth?.detail, !detail.isEmpty {
            return detail
        }

        if systemStatus != nil {
            if runtimeIsHealthy {
                return "Primary runtime services are healthy"
            }
            return "\(healthyServiceCount)/\(totalServiceCount) integrations are healthy"
        }

        return api.lastError ?? "System diagnostics unavailable"
    }

    var runtimeSourceLabel: String {
        if runtimeSnapshot?.hasDetails == true {
            return "Runtime endpoint"
        }
        if !recentXRayEvents.isEmpty {
            return "Status + X-ray fallback"
        }
        if !api.networkEvents.isEmpty {
            return "Status + client telemetry"
        }
        return "Baseline status fallback"
    }

    var runtimeManagerInfo: SystemRuntimeManagerInfo {
        if let manager = runtimeSnapshot?.runtimeManager {
            return SystemRuntimeManagerInfo(
                name: manager.name ?? "Chronos Runtime",
                mode: manager.mode?.displayLabel ?? connectionModeLabel,
                state: displayState(rawValue: manager.healthy == true ? "healthy" : nil, healthy: manager.healthy),
                isHealthy: manager.healthy ?? runtimeIsHealthy,
                detail: manager.detail ?? fallbackManagerDetail,
                watchdog: manager.watchdogStatus?.displayLabel
                    ?? (manager.watchdogEnabled == true ? "Enabled" : "Unknown"),
                verified: manager.lastVerifiedAt?.iso8601Date?.relativeString
                    ?? api.lastHealthCheck?.relativeString
                    ?? "Unknown",
                version: manager.version ?? "Not exposed"
            )
        }

        return SystemRuntimeManagerInfo(
            name: "Chronos API",
            mode: connectionModeLabel,
            state: runtimeStateText,
            isHealthy: api.isServerReachable && runtimeIsHealthy,
            detail: fallbackManagerDetail,
            watchdog: "Not exposed",
            verified: api.lastHealthCheck?.relativeString ?? "Unknown",
            version: "Not exposed"
        )
    }

    var managerName: String {
        runtimeManagerInfo.name
    }

    var managerDetail: String {
        runtimeManagerInfo.detail
    }

    var plaudAuthSummary: String {
        if let plaudAuth = runtimeSnapshot?.plaudAuth {
            if let detail = plaudAuth.detail, !detail.isEmpty {
                return detail
            }
            if plaudAuth.isAuthenticated == true {
                return "Authenticated"
            }
            if let state = plaudAuth.state, !state.isEmpty {
                return state.displayLabel
            }
        }

        guard let plaud = systemStatus?.plaud else {
            return "Unknown"
        }

        if let error = plaud.error, !error.isEmpty {
            return error
        }
        if plaud.isAuthenticated == true {
            return "Authenticated"
        }
        if plaud.hasCredentials == true {
            return "Credentials present"
        }
        return "Not connected"
    }

    var serviceEntries: [SystemServiceEntry] {
        if let runtimeSnapshot, !runtimeSnapshot.serviceList.isEmpty {
            return runtimeSnapshot.serviceList.map { service in
                var footnotes: [String] = []
                if let category = service.category, !category.isEmpty {
                    footnotes.append(category.displayLabel)
                }
                if let enabled = service.enabled {
                    footnotes.append(enabled ? "Enabled" : "Disabled")
                }
                if let unitName = service.unitName, !unitName.isEmpty {
                    footnotes.append(unitName)
                }
                if let transition = relativeLabel(for: service.lastTransitionAt) {
                    footnotes.append("Updated \(transition)")
                }

                return SystemServiceEntry(
                    id: service.id,
                    title: service.displayName ?? service.name.displayLabel,
                    state: displayState(rawValue: service.state, healthy: service.healthy),
                    isHealthy: service.healthy ?? false,
                    detail: service.detail ?? service.url,
                    footnote: footnotes.isEmpty ? nil : footnotes.joined(separator: " · ")
                )
            }
        }

        guard let systemStatus else {
            return []
        }

        return fallbackServiceEntries(from: systemStatus)
    }

    var portEntries: [SystemPortEntry] {
        if let runtimeSnapshot, !runtimeSnapshot.portList.isEmpty {
            return runtimeSnapshot.portList.map { port in
                let transport = (port.protocolName ?? "tcp").uppercased()
                return SystemPortEntry(
                    id: port.id,
                    name: port.name,
                    summary: "\(transport) · \(port.port)",
                    isReachable: port.reachable,
                    detail: port.detail ?? port.url
                )
            }
        }

        var entries: [SystemPortEntry] = []

        if let apiEntry = derivedPortEntry(
            name: api.isServerReachable ? "Chronos API" : "Configured API",
            urlString: api.resolvedServerURL,
            reachable: api.isServerReachable,
            detail: "Current backend endpoint"
        ) {
            entries.append(apiEntry)
        }

        if let qdrantURL = systemStatus?.qdrant?.url,
           let qdrantEntry = derivedPortEntry(
               name: "Qdrant",
               urlString: qdrantURL,
               reachable: systemStatus?.qdrant?.ok,
               detail: systemStatus?.qdrant?.error ?? qdrantSummary(systemStatus?.qdrant) ?? "Vector database endpoint"
           ) {
            entries.append(qdrantEntry)
        }

        var seen = Set<String>()
        return entries.filter { seen.insert($0.id).inserted }
    }

    var signalEntries: [SystemSignalEntry] {
        if let runtimeSnapshot, !runtimeSnapshot.signalList.isEmpty {
            return runtimeSnapshot.signalList.map { signal in
                var footnotes: [String] = []
                if let service = signal.service, !service.isEmpty {
                    footnotes.append(service.displayLabel)
                }
                if let timestamp = relativeLabel(for: signal.timestamp) {
                    footnotes.append(timestamp)
                }

                return SystemSignalEntry(
                    id: signal.id,
                    source: signal.source.displayLabel,
                    level: (signal.level ?? "info").lowercased(),
                    title: signal.title?.displayLabel ?? signal.source.displayLabel,
                    message: signal.message ?? "No message",
                    footnote: footnotes.isEmpty ? nil : footnotes.joined(separator: " · "),
                    timestamp: signal.timestamp?.iso8601Date
                )
            }
            .sorted { lhs, rhs in
                (lhs.timestamp ?? .distantPast) > (rhs.timestamp ?? .distantPast)
            }
        }

        if !recentXRayEvents.isEmpty {
            return recentXRayEvents.map { event in
                var footnotes: [String] = [event.date.relativeString]
                if let durationMs = event.durationMs {
                    footnotes.append("\(Int(durationMs))ms")
                }

                let message: String
                if let detail = event.detail, !detail.isEmpty, detail != event.message {
                    message = "\(event.message) — \(detail)"
                } else {
                    message = event.message
                }

                return SystemSignalEntry(
                    id: "xray-\(event.id)",
                    source: event.source.displayLabel,
                    level: event.level.lowercased(),
                    title: event.op.displayLabel,
                    message: message,
                    footnote: footnotes.joined(separator: " · "),
                    timestamp: event.date
                )
            }
        }

        return api.networkEvents.prefix(8).map { event in
            var footnotes = [event.timestamp.relativeString, event.requestId]
            if let statusCode = event.statusCode {
                footnotes.append("HTTP \(statusCode)")
            }

            let message: String
            if let errorMessage = event.errorMessage, !errorMessage.isEmpty {
                message = errorMessage
            } else if let statusCode = event.statusCode {
                message = "HTTP \(statusCode) in \(event.durationMs)ms"
            } else {
                message = "Completed in \(event.durationMs)ms"
            }

            return SystemSignalEntry(
                id: "net-\(event.id.uuidString)",
                source: "Client Network",
                level: event.isError ? "error" : "info",
                title: "\(event.method) \(event.path)",
                message: message,
                footnote: footnotes.joined(separator: " · "),
                timestamp: event.timestamp
            )
        }
    }

    var notes: [String] {
        if let notes = runtimeSnapshot?.noteList, !notes.isEmpty {
            return notes
        }

        var derived: [String] = []

        if supportsRuntimeEndpoint == false {
            derived.append("Using fallback diagnostics from /api/status and live telemetry because the backend does not expose /api/admin/runtime.")
        }

        if let qdrant = systemStatus?.qdrant, qdrant.ok, let collections = qdrant.collections {
            derived.append("Qdrant is reachable with \(collections) collection\(collections == 1 ? "" : "s").")
        }

        if systemStatus?.plaud?.isAuthenticated == true {
            derived.append("Plaud authentication is active for sync and workflow operations.")
        } else if systemStatus?.plaud?.hasCredentials == true {
            derived.append("Plaud credentials exist, but the service is not currently authenticated.")
        }

        if let workspaceName = systemStatus?.notion?.workspaceName, !workspaceName.isEmpty {
            derived.append("Notion is connected to \(workspaceName).")
        }

        if let local = systemStatus?.localLLM {
            if local.isUp {
                derived.append("Local LLM sidecar is reachable via \(local.provider ?? "local") using \(local.model ?? "configured model").")
            } else if local.enabled == true {
                derived.append("Local LLM is enabled but not reachable: \(local.error ?? local.detail ?? "unknown reason").")
            } else {
                derived.append("Local LLM sidecar is disabled; cloud providers remain the primary Chronos brain.")
            }
        }

        if !api.isServerReachable, let lastError = api.lastError, !lastError.isEmpty {
            derived.append("Last connection error: \(lastError)")
        }

        return derived
    }

    var shouldShowRuntimeUpgradeHint: Bool {
        false
    }

    var portSourceLabel: String {
        if runtimeSnapshot?.portList.isEmpty == false {
            return "Core backend entry points exposed by the runtime"
        }
        return "Derived from the active server and known service URLs"
    }

    var signalSourceLabel: String {
        if runtimeSnapshot?.signalList.isEmpty == false {
            return "Operational events from the runtime endpoint"
        }
        if !recentXRayEvents.isEmpty {
            return "Recent backend telemetry from X-ray"
        }
        if !api.networkEvents.isEmpty {
            return "Recent client-side network activity"
        }
        return "No recent operational events"
    }

    func bootstrapIfNeeded() async {
        guard !hasBootstrapped else { return }
        hasBootstrapped = true
        await loadAll()
    }

    func refresh() async {
        await loadAll(showLoading: false)
    }

    func loadAll(showLoading: Bool = true) async {
        if showLoading {
            isLoading = true
        }

        error = nil
        runtimeNotice = nil

        await loadSystemStatus()
        await loadRuntimeSnapshot()
        await loadXRaySignals()

        lastUpdated = runtimeSnapshot?.capturedAt?.iso8601Date ?? recentXRayEvents.first?.date ?? Date()

        if systemStatus == nil && runtimeSnapshot == nil && error == nil {
            error = api.lastError ?? "Could not load system diagnostics"
        }

        if showLoading {
            isLoading = false
        }
    }

    private func loadSystemStatus() async {
        do {
            systemStatus = try await api.get("/api/status")
        } catch is CancellationError {
            return
        } catch let urlError as URLError where urlError.code == .cancelled {
            return
        } catch {
            // Keep last successful systemStatus; only surface the error if we have nothing
            if systemStatus == nil, self.error == nil {
                self.error = error.localizedDescription
            }
        }
    }

    private func loadRuntimeSnapshot() async {
        guard supportsRuntimeEndpoint else {
            runtimeSnapshot = nil
            runtimeNotice = "Detailed runtime manager data is unavailable, so the app is using /api/status and live telemetry fallbacks."
            return
        }

        do {
            let snapshot: SystemRuntimeSnapshot = try await api.get("/api/admin/runtime")
            runtimeSnapshot = snapshot

            if !snapshot.hasDetails {
                runtimeNotice = "The runtime endpoint responded without detailed manager or signal data, so the app is supplementing with local fallbacks."
            }
        } catch {
            runtimeSnapshot = nil

            if isNotFound(error) {
                supportsRuntimeEndpoint = false
                runtimeNotice = "This backend only exposes baseline /api/status health, so the app is deriving manager, ports, and signals from existing endpoints."
                return
            }

            runtimeNotice = "Detailed runtime diagnostics failed to load: \(error.localizedDescription)"
        }
    }

    private func loadXRaySignals() async {
        guard supportsXRayEndpoint else {
            recentXRayEvents = []
            return
        }

        do {
            let response: XRayEventsResponse = try await api.get("/api/xray/events", query: ["limit": "12"])
            recentXRayEvents = response.events
        } catch {
            if isNotFound(error) {
                supportsXRayEndpoint = false
                recentXRayEvents = []
            }
        }
    }

    private func fallbackServiceEntries(from status: SystemStatus) -> [SystemServiceEntry] {
        [
            SystemServiceEntry(
                id: "database",
                title: "Database",
                state: status.database?.ok == true ? "Healthy" : "Down",
                isHealthy: status.database?.ok == true,
                detail: status.database?.error ?? status.database?.detail,
                footnote: status.database?.url
            ),
            SystemServiceEntry(
                id: "qdrant",
                title: "Qdrant",
                state: status.qdrant?.ok == true ? "Healthy" : "Down",
                isHealthy: status.qdrant?.ok == true,
                detail: status.qdrant?.error ?? qdrantSummary(status.qdrant),
                footnote: status.qdrant?.url
            ),
            SystemServiceEntry(
                id: "gemini",
                title: status.gemini?.enabled == false ? "Gemini Off" : "Gemini",
                state: status.gemini?.enabled == false ? "Quota Safe" : (status.gemini?.isUp == true ? "Configured" : "Unavailable"),
                isHealthy: status.gemini?.isSafeOrOperational == true,
                detail: status.gemini?.error ?? status.gemini?.detail,
                footnote: nil
            ),
            SystemServiceEntry(
                id: "openai",
                title: status.openai?.enabled == false ? "OpenAI Off" : "OpenAI",
                state: status.openai?.enabled == false ? "Quota Safe" : (status.openai?.ok == true ? "Healthy" : "Unavailable"),
                isHealthy: status.openai?.isSafeOrOperational == true,
                detail: status.openai?.error ?? status.openai?.detail,
                footnote: status.openai?.url
            ),
            SystemServiceEntry(
                id: "plaud",
                title: "Plaud",
                state: status.plaud?.isUp == true ? "Authenticated" : "Disconnected",
                isHealthy: status.plaud?.isUp == true,
                detail: status.plaud?.error,
                footnote: plaudAuthSummary
            ),
            SystemServiceEntry(
                id: "notion",
                title: "Notion",
                state: status.notion?.isUp == true ? "Connected" : "Disconnected",
                isHealthy: status.notion?.isUp == true,
                detail: status.notion?.error,
                footnote: status.notion?.workspaceName
            ),
            SystemServiceEntry(
                id: "local-llm",
                title: "Local LLM",
                state: status.localLLM?.isUp == true
                    ? "Ready"
                    : (status.localLLM?.enabled == true ? "Unavailable" : "Disabled"),
                isHealthy: status.localLLM?.isUp == true || status.localLLM?.enabled != true,
                detail: status.localLLM?.error ?? status.localLLM?.detail,
                footnote: localLLMSummary(status.localLLM)
            )
        ]
    }

    private func localLLMSummary(_ local: SystemStatus.LocalLLMCheck?) -> String? {
        guard let local else { return nil }
        var parts: [String] = []
        if let provider = local.provider, !provider.isEmpty { parts.append(provider) }
        if let model = local.model, !model.isEmpty { parts.append(model) }
        if let maxContext = local.maxContext { parts.append("ctx \(maxContext)") }
        if let tasks = local.allowedTasks, !tasks.isEmpty { parts.append(tasks.joined(separator: ", ")) }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    private func qdrantSummary(_ qdrant: SystemStatus.QdrantCheck?) -> String? {
        guard let qdrant else { return nil }
        if let collections = qdrant.collections {
            return "\(collections) collection\(collections == 1 ? "" : "s")"
        }
        return nil
    }

    private var fallbackManagerDetail: String {
        if !api.isServerReachable {
            return api.lastError ?? "The configured server is currently unreachable."
        }

        if runtimeIsHealthy {
            return "Using baseline health checks and live telemetry as a runtime proxy for the current \(connectionModeLabel.lowercased()) connection."
        }

        return "The server is reachable, but one or more runtime-critical services need attention."
    }

    private var connectionModeLabel: String {
        guard let host = URLComponents(string: api.resolvedServerURL)?.host?.lowercased() else {
            return "Custom"
        }

        if host == "localhost" || host == "127.0.0.1" || host == "::1" {
            return "Debug Loopback"
        }
        if host.contains("ngrok") {
            return "Ngrok Tunnel"
        }
        if host.contains("tailscale") || host.hasSuffix("ts.net") {
            return "Tailscale"
        }
        if host.hasSuffix(".local") || isPrivateIPv4Host(host) {
            return "LAN"
        }
        return "Custom"
    }

    private func derivedPortEntry(
        name: String,
        urlString: String,
        reachable: Bool?,
        detail: String?
    ) -> SystemPortEntry? {
        guard let components = URLComponents(string: urlString),
              let host = components.host,
              !host.isEmpty else {
            return nil
        }

        let port = components.port ?? defaultPort(for: components.scheme)
        let scheme = (components.scheme ?? "http").uppercased()
        let summary = "\(scheme) · \(port)"
        let hostDetail = components.port == nil ? host : "\(host):\(port)"

        return SystemPortEntry(
            id: "\(name)-\(host)-\(port)",
            name: name,
            summary: summary,
            isReachable: reachable,
            detail: [detail, hostDetail].compactMap { $0 }.joined(separator: " · ")
        )
    }

    private func defaultPort(for scheme: String?) -> Int {
        switch scheme?.lowercased() {
        case "https":
            return 443
        default:
            return 80
        }
    }

    private func isPrivateIPv4Host(_ host: String) -> Bool {
        let octets = host.split(separator: ".").compactMap { Int($0) }
        guard octets.count == 4 else { return false }

        if octets[0] == 10 {
            return true
        }
        if octets[0] == 192 && octets[1] == 168 {
            return true
        }
        if octets[0] == 172 && (16...31).contains(octets[1]) {
            return true
        }
        return false
    }

    private func displayState(rawValue: String?, healthy: Bool?) -> String {
        if let rawValue, !rawValue.isEmpty {
            return rawValue.displayLabel
        }
        if let healthy {
            return healthy ? "Healthy" : "Unhealthy"
        }
        return "Unknown"
    }

    private func relativeLabel(for isoTimestamp: String?) -> String? {
        guard let date = isoTimestamp?.iso8601Date else {
            return nil
        }
        return date.relativeString
    }

    private func isNotFound(_ error: Error) -> Bool {
        guard case APIError.httpError(let status, _) = error else {
            return false
        }
        return status == 404
    }
}

private extension String {
    var displayLabel: String {
        replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")
            .capitalized
    }
}
