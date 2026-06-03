import Foundation

/// Optional richer runtime diagnostics from `GET /api/v1/admin/runtime`.
/// The current backend may not expose this yet, so every field is optional.
struct SystemRuntimeSnapshot: Decodable, Sendable {
    let capturedAt: String?
    let runtimeHealth: RuntimeHealth?
    let runtimeManager: RuntimeManager?
    let access: RuntimeAccess?
    let services: [ManagedServiceStatus]?
    let ports: [PortStatus]?
    let signals: [OperationalSignal]?
    let plaudAuth: RuntimeAuthState?
    let notes: [String]?

    var hasDetails: Bool {
        runtimeHealth != nil
            || runtimeManager != nil
            || access != nil
            || !(services ?? []).isEmpty
            || !(ports ?? []).isEmpty
            || !(signals ?? []).isEmpty
            || plaudAuth != nil
            || !(notes ?? []).isEmpty
    }

    var serviceList: [ManagedServiceStatus] {
        services ?? []
    }

    var portList: [PortStatus] {
        ports ?? []
    }

    var signalList: [OperationalSignal] {
        signals ?? []
    }

    var noteList: [String] {
        notes ?? []
    }

    enum CodingKeys: String, CodingKey {
        case access, services, ports, signals, notes
        case capturedAt = "captured_at"
        case runtimeHealth = "runtime_health"
        case runtimeManager = "runtime_manager"
        case plaudAuth = "plaud_auth"
    }

    struct RuntimeAccess: Decodable, Sendable {
        let preferredKind: String?
        let preferredLabel: String?
        let preferredUIURL: String?
        let preferredAPIURL: String?
        let entries: [AccessEntry]?

        var entryList: [AccessEntry] {
            entries ?? []
        }

        enum CodingKeys: String, CodingKey {
            case entries
            case preferredKind = "preferred_kind"
            case preferredLabel = "preferred_label"
            case preferredUIURL = "preferred_ui_url"
            case preferredAPIURL = "preferred_api_url"
        }
    }

    struct AccessEntry: Decodable, Identifiable, Sendable {
        let label: String
        let url: String
        let kind: String?

        var id: String { "\(label)-\(url)" }
    }

    struct RuntimeHealth: Decodable, Sendable {
        let ok: Bool?
        let summary: String?
        let detail: String?
        let passedChecks: Int?
        let warningCount: Int?
        let failureCount: Int?

        enum CodingKeys: String, CodingKey {
            case ok, summary, detail
            case passedChecks = "passed_checks"
            case warningCount = "warning_count"
            case failureCount = "failure_count"
        }
    }

    struct RuntimeManager: Decodable, Sendable {
        let name: String?
        let mode: String?
        let healthy: Bool?
        let detail: String?
        let version: String?
        let watchdogEnabled: Bool?
        let watchdogStatus: String?
        let lastVerifiedAt: String?

        enum CodingKeys: String, CodingKey {
            case name, mode, healthy, detail, version
            case watchdogEnabled = "watchdog_enabled"
            case watchdogStatus = "watchdog_status"
            case lastVerifiedAt = "last_verified_at"
        }
    }

    struct ManagedServiceStatus: Decodable, Identifiable, Sendable {
        let name: String
        let displayName: String?
        let category: String?
        let state: String?
        let healthy: Bool?
        let enabled: Bool?
        let detail: String?
        let unitName: String?
        let url: String?
        let port: Int?
        let lastTransitionAt: String?

        var id: String { name }

        enum CodingKeys: String, CodingKey {
            case name, category, state, healthy, enabled, detail, url, port
            case displayName = "display_name"
            case unitName = "unit_name"
            case lastTransitionAt = "last_transition_at"
        }
    }

    struct PortStatus: Decodable, Identifiable, Sendable {
        let name: String
        let port: Int
        let protocolName: String?
        let reachable: Bool?
        let url: String?
        let detail: String?

        var id: String { "\(name)-\(port)" }

        enum CodingKeys: String, CodingKey {
            case name, port, reachable, url, detail
            case protocolName = "protocol"
        }
    }

    struct OperationalSignal: Decodable, Identifiable, Sendable {
        let source: String
        let level: String?
        let title: String?
        let message: String?
        let service: String?
        let timestamp: String?

        var id: String {
            [source, title ?? "", timestamp ?? "", service ?? ""]
                .joined(separator: "|")
        }
    }

    struct RuntimeAuthState: Decodable, Sendable {
        let state: String?
        let detail: String?
        let isAuthenticated: Bool?
        let workspaceName: String?

        enum CodingKeys: String, CodingKey {
            case state, detail
            case isAuthenticated = "is_authenticated"
            case workspaceName = "workspace_name"
        }
    }
}
