import Foundation

/// Deep system connectivity status from `GET /api/v1/status`.
struct SystemStatus: Decodable, Sendable {
    let database: ServiceCheck?
    let qdrant: QdrantCheck?
    let gemini: ServiceFlag?
    let openai: ServiceCheck?
    let plaud: ServiceFlag?
    let notion: ServiceFlag?
    let localLLM: LocalLLMCheck?

    struct ServiceCheck: Decodable, Sendable {
        let ok: Bool
        let configured: Bool?
        let enabled: Bool?
        let skipped: Bool?
        let url: String?
        let detail: String?
        let error: String?

        var isEnabled: Bool { enabled != false }
        var isOperational: Bool { ok && isEnabled }
        var isSafeOrOperational: Bool { isOperational || skipped == true }
    }

    struct QdrantCheck: Decodable, Sendable {
        let ok: Bool
        let url: String?
        let collections: Int?
        let error: String?
    }

    struct ServiceFlag: Decodable, Sendable {
        let configured: Bool?
        let ok: Bool?
        let enabled: Bool?
        let skipped: Bool?
        let isAuthenticated: Bool?
        let hasCredentials: Bool?
        let hasAccessToken: Bool?
        let workspaceName: String?
        let detail: String?
        let error: String?

        var isEnabled: Bool { enabled != false }

        var isUp: Bool {
            (ok == true && isEnabled)
                || isAuthenticated == true
                || hasCredentials == true
                || hasAccessToken == true
                || configured == true
        }

        var isSafeOrOperational: Bool { isUp || skipped == true }

        enum CodingKeys: String, CodingKey {
            case configured, ok, enabled, skipped, detail, error
            case hasCredentials = "has_credentials"
            case hasAccessToken = "has_access_token"
            case workspaceName = "workspace_name"
            case isAuthenticated = "is_authenticated"
        }
    }

    struct LocalLLMCheck: Decodable, Sendable {
        let enabled: Bool?
        let ok: Bool?
        let provider: String?
        let baseURL: String?
        let model: String?
        let maxContext: Int?
        let allowedTasks: [String]?
        let modelAvailable: Bool?
        let detail: String?
        let error: String?

        var isUp: Bool { enabled == true && ok == true }

        enum CodingKeys: String, CodingKey {
            case enabled, ok, provider, model, detail, error
            case baseURL = "base_url"
            case maxContext = "max_context"
            case allowedTasks = "allowed_tasks"
            case modelAvailable = "model_available"
        }
    }

    enum CodingKeys: String, CodingKey {
        case database, qdrant, gemini, openai, plaud, notion
        case localLLM = "local_llm"
    }
}
