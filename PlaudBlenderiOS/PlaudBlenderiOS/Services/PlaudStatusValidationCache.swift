import Foundation

enum PlaudStatusValidationState: String, Sendable {
    case validatedWithServer
    case reusedCachedValidation
    case usedCachedAfterTransientFailure
    case requiresReauthentication
}

struct PlaudStatusValidationDiagnostics: Sendable {
    let state: PlaudStatusValidationState
    let message: String
    let validatedAt: Date?
}

struct PlaudStatusValidationResult: Sendable {
    let status: TokenStatus?
    let diagnostics: PlaudStatusValidationDiagnostics
}

struct PlaudStatusValidationCache: Sendable {
    var cachedStatus: TokenStatus?
    var lastValidatedAt: Date?
    var validationTTL: TimeInterval = 5 * 60
    var minimumRemainingValidity: TimeInterval = 30 * 60

    mutating func load(
        forceRefresh: Bool = false,
        now: Date = Date(),
        fetch: () async throws -> TokenStatus
    ) async throws -> PlaudStatusValidationResult {
        if !forceRefresh, let cachedValidation = cachedValidationResult(now: now) {
            return cachedValidation
        }

        let status = try await fetch()
        return recordServerStatus(status, now: now)
    }

    func cachedValidationResult(now: Date = Date()) -> PlaudStatusValidationResult? {
        guard let cachedStatus,
              let lastValidatedAt,
              now.timeIntervalSince(lastValidatedAt) <= validationTTL,
              cachedStatus.canSkipValidation(
                minimumRemainingValidity: minimumRemainingValidity,
                relativeTo: now
              ) else {
            return nil
        }

        return PlaudStatusValidationResult(
            status: cachedStatus,
            diagnostics: PlaudStatusValidationDiagnostics(
                state: .reusedCachedValidation,
                message: "Reused Plaud validation from \(durationString(now.timeIntervalSince(lastValidatedAt))) ago.",
                validatedAt: lastValidatedAt
            )
        )
    }

    mutating func recordServerStatus(
        _ status: TokenStatus,
        now: Date = Date()
    ) -> PlaudStatusValidationResult {
        cachedStatus = status
        lastValidatedAt = now

        if status.hasUsableSession(relativeTo: now) {
            return PlaudStatusValidationResult(
                status: status,
                diagnostics: PlaudStatusValidationDiagnostics(
                    state: .validatedWithServer,
                    message: "Plaud auth confirmed with the server.",
                    validatedAt: now
                )
            )
        }

        return PlaudStatusValidationResult(
            status: status,
            diagnostics: PlaudStatusValidationDiagnostics(
                state: .requiresReauthentication,
                message: "Plaud session is no longer valid. Reconnect Plaud.",
                validatedAt: now
            )
        )
    }

    mutating func recoverFromTransientFailure(
        _ error: Error,
        now: Date = Date()
    ) -> PlaudStatusValidationResult? {
        guard isTransientValidationFailure(error),
              let cachedStatus,
              cachedStatus.hasUsableSession(relativeTo: now) else {
            return nil
        }

        return PlaudStatusValidationResult(
            status: cachedStatus,
            diagnostics: PlaudStatusValidationDiagnostics(
                state: .usedCachedAfterTransientFailure,
                message: "Plaud validation hit a transient error, so the last confirmed session is still in use.",
                validatedAt: lastValidatedAt
            )
        )
    }

    mutating func requireReauthentication(
        message: String,
        now: Date = Date()
    ) -> PlaudStatusValidationResult {
        cachedStatus = nil
        lastValidatedAt = nil

        return PlaudStatusValidationResult(
            status: nil,
            diagnostics: PlaudStatusValidationDiagnostics(
                state: .requiresReauthentication,
                message: message,
                validatedAt: now
            )
        )
    }

    mutating func invalidate() {
        cachedStatus = nil
        lastValidatedAt = nil
    }

    private func isTransientValidationFailure(_ error: Error) -> Bool {
        if let urlError = error as? URLError {
            return urlError.code != .cancelled
        }

        if case APIError.httpError(let status, _) = error {
            return status == 408 || status == 425 || status == 429 || status >= 500
        }

        let nsError = error as NSError
        return nsError.domain == NSURLErrorDomain
    }

    private func durationString(_ interval: TimeInterval) -> String {
        let seconds = max(0, Int(interval.rounded()))
        if seconds < 60 {
            return "\(seconds)s"
        }
        if seconds < 3600 {
            return "\(seconds / 60)m"
        }
        return "\(seconds / 3600)h"
    }
}
