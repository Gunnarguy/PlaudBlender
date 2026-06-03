import Foundation

// MARK: - Auth URL

/// Response from `/api/auth/plaud/authorize` or `/api/auth/notion/authorize`.
struct AuthURLResponse: Codable, Sendable {
    let authUrl: String
    let state: String

    enum CodingKeys: String, CodingKey {
        case authUrl = "auth_url"
        case state
    }
}

/// Request body for token exchange.
struct TokenExchangeRequest: Codable, Sendable {
    let code: String
    var state: String?
}

/// Response from `/api/auth/plaud/status` or `/api/auth/notion/status`.
struct TokenStatus: Codable, Sendable {
    let isAuthenticated: Bool
    let hasAccessToken: Bool
    let expiresAt: String?
    let workspaceName: String?
    let extra: [String: JSONValue]?

    enum CodingKeys: String, CodingKey {
        case isAuthenticated = "is_authenticated"
        case hasAccessToken = "has_access_token"
        case expiresAt = "expires_at"
        case workspaceName = "workspace_name"
        case extra
    }
}

extension TokenStatus {
    var expiresAtDate: Date? {
        Self.parseTimestamp(expiresAt)
    }

    func hasUsableSession(relativeTo date: Date) -> Bool {
        guard isAuthenticated else {
            return false
        }
        guard let expiresAtDate else {
            return true
        }
        return expiresAtDate > date
    }

    func canSkipValidation(minimumRemainingValidity: TimeInterval, relativeTo date: Date) -> Bool {
        guard hasUsableSession(relativeTo: date) else {
            return false
        }
        guard let expiresAtDate else {
            return true
        }
        return expiresAtDate.timeIntervalSince(date) > minimumRemainingValidity
    }

    private static func parseTimestamp(_ value: String?) -> Date? {
        guard let value, !value.isEmpty else {
            return nil
        }

        if let parsed = fractionalTimestampFormatter.date(from: value) {
            return parsed
        }
        return basicTimestampFormatter.date(from: value)
    }

    private static let fractionalTimestampFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    private static let basicTimestampFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()
}
