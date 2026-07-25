import Foundation
import Observation
import OSLog

private let logger = Logger(subsystem: "com.gunndamental.PlaudBlenderiOS", category: "APIClient")

enum IntegrationTransport: String, Codable, Sendable {
    case chronosREST
    case plaudAccountREST
    case plaudMCP
    case plaudEmbeddedREST
    case plaudUpload
}

struct ClientNetworkEvent: Identifiable, Sendable {
    let id = UUID()
    let timestamp: Date
    let kind: String
    let method: String
    let url: String
    let path: String
    let statusCode: Int?
    let durationMs: Int
    let requestBytes: Int
    let responseBytes: Int
    let requestHeaders: [String: String]
    let responseHeaders: [String: String]
    let requestPreview: String?
    let responsePreview: String?
    let errorMessage: String?
    let requestId: String
    let transport: IntegrationTransport

    var isError: Bool {
        if errorMessage != nil {
            return true
        }
        if let statusCode {
            return !(200..<300).contains(statusCode)
        }
        return false
    }
}

/// Central HTTP client for all Chronos API calls.
/// Uses async/await with URLSession. Adds Bearer token auth automatically.
///
/// All ViewModels pass paths like "/api/search" but the FastAPI backend
/// mounts routes at "/api/v1/...". This client rewrites `/api/` → `/api/v1/`
/// so callers don't need to know the versioned path.
@Observable
final class APIClient: Sendable {
    let authManager: AuthManager
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder
    private let session: URLSession
    private var preferredServerURLSnapshot: String
    private var activeServerURL: String

    /// Observable connectivity state — drives UI banners
    var isServerReachable: Bool = false
    var lastHealthCheck: Date?
    var lastError: String?
    var networkEvents: [ClientNetworkEvent] = []

    var resolvedServerURL: String {
        activeServerURL
    }

    private let maxNetworkEvents = 300

    init(authManager: AuthManager) {
        self.authManager = authManager
        self.preferredServerURLSnapshot = authManager.serverURL
        self.activeServerURL = authManager.serverURL

        self.decoder = JSONDecoder()
        self.encoder = JSONEncoder()
        self.encoder.keyEncodingStrategy = .convertToSnakeCase

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 15
        config.waitsForConnectivity = false  // fail fast so we see errors
        self.session = URLSession(configuration: config)

        logger.info("APIClient initialized — server: \(authManager.serverURL, privacy: .public)")
    }

    private var baseURL: URL {
        syncPreferredServerURLIfNeeded()

        guard let url = URL(string: activeServerURL) else {
            // Fallback — should never happen if AuthManager validates
            logger.fault("Invalid server URL: \(self.activeServerURL, privacy: .public)")
            return URL(string: AuthManager.defaultServerURL)!
        }
        return url
    }

    /// Rewrites `/api/foo` → `/api/v1/foo` to match the FastAPI router prefixes.
    private func versionedPath(_ path: String) -> String {
        if path.hasPrefix("/api/") && !path.hasPrefix("/api/v1/") {
            return path.replacingOccurrences(of: "/api/", with: "/api/v1/", options: [], range: path.startIndex..<path.index(path.startIndex, offsetBy: min(5, path.count)))
        }
        return path
    }

    // MARK: - HTTP Methods

    func get<T: Decodable>(_ path: String, query: [String: String] = [:], timeoutInterval: TimeInterval? = nil) async throws -> T {
        let request = try buildRequest(
            path: versionedPath(path),
            method: "GET",
            query: query,
            timeoutInterval: timeoutInterval
        )
        return try await executeWithRetry(request)
    }

    func post<T: Decodable, B: Encodable>(_ path: String, body: B) async throws -> T {
        var request = try buildRequest(path: versionedPath(path), method: "POST")
        request.httpBody = try encoder.encode(body)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        return try await execute(request)
    }

    func post<T: Decodable>(_ path: String) async throws -> T {
        let request = try buildRequest(path: versionedPath(path), method: "POST")
        return try await execute(request)
    }

    func put<T: Decodable, B: Encodable>(_ path: String, body: B) async throws -> T {
        var request = try buildRequest(path: versionedPath(path), method: "PUT")
        request.httpBody = try encoder.encode(body)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        return try await execute(request)
    }

    func delete<T: Decodable>(_ path: String) async throws -> T {
        let request = try buildRequest(path: versionedPath(path), method: "DELETE")
        return try await execute(request)
    }

    func downloadFile(_ path: String) async throws -> URL {
        let request = try buildRequest(path: versionedPath(path), method: "GET")
        let (data, response) = try await session.data(for: request)

        guard let http = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }

        guard 200..<300 ~= http.statusCode else {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw APIError.httpError(status: http.statusCode, body: body)
        }

        let suggestedName: String = {
            if let disposition = http.value(forHTTPHeaderField: "Content-Disposition"),
               let range = disposition.range(of: "filename=") {
                return disposition[range.upperBound...]
                    .trimmingCharacters(in: CharacterSet(charactersIn: "\" "))
            }
            return request.url?.lastPathComponent ?? "download.bin"
        }()

        let destination = FileManager.default.temporaryDirectory.appendingPathComponent(suggestedName)
        if FileManager.default.fileExists(atPath: destination.path) {
            try? FileManager.default.removeItem(at: destination)
        }
        try data.write(to: destination, options: .atomic)
        return destination
    }

    // MARK: - Health Check (no auth required)

    func healthCheck() async -> Bool {
        syncPreferredServerURLIfNeeded()

        if await checkHealth(at: activeServerURL) {
            return true
        }

        logger.warning("🏥 Health check failed for configured server; probing fallbacks")
        return await bootstrapConnection()
    }

    @discardableResult
    func bootstrapConnection() async -> Bool {
        syncPreferredServerURLIfNeeded()

        for candidate in authManager.candidateServerURLs() {
            if await checkHealth(at: candidate) {
                if candidate != activeServerURL {
                    activeServerURL = candidate
                    logger.info("✅ Using reachable server URL \(candidate, privacy: .public)")
                }
                return true
            }
        }

        isServerReachable = false
        lastHealthCheck = Date()
        lastError = "Could not connect to the Chronos API at the configured, Tailscale, public, or LAN endpoints"
        return false
    }

    private func syncPreferredServerURLIfNeeded() {
        let preferred = authManager.serverURL
        guard preferred != preferredServerURLSnapshot else { return }
        preferredServerURLSnapshot = preferred
        activeServerURL = preferred
        logger.info("🔄 Preferred server URL changed to \(preferred, privacy: .public)")
    }

    // MARK: - Private

    /// AVURLAsset cannot consume a URLRequest, so streaming callers need the
    /// resolved URL and auth headers handed back separately.
    func streamingTarget(_ path: String) -> (url: URL, headers: [String: String])? {
        let cleanBase = baseURL.absoluteString
        let versioned = versionedPath(path)
        let cleanPath = versioned.hasPrefix("/") ? String(versioned.dropFirst()) : versioned
        let full = cleanBase.hasSuffix("/") ? "\(cleanBase)\(cleanPath)" : "\(cleanBase)/\(cleanPath)"

        guard let url = URL(string: full) else { return nil }
        var headers = ["ngrok-skip-browser-warning": "true"]
        if let token = authManager.getToken() {
            headers["Authorization"] = "Bearer \(token)"
        }
        return (url, headers)
    }

    private func buildRequest(
        path: String,
        method: String,
        query: [String: String] = [:],
        timeoutInterval: TimeInterval? = nil
    ) throws -> URLRequest {
        let cleanBase = baseURL.absoluteString
        let cleanPath = path.hasPrefix("/") ? String(path.dropFirst()) : path
        let fullURLString = cleanBase.hasSuffix("/") ? "\(cleanBase)\(cleanPath)" : "\(cleanBase)/\(cleanPath)"

        guard let url = URL(string: fullURLString),
              var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else {
            throw APIError.invalidURL(path)
        }
        if !query.isEmpty {
            components.queryItems = query.map {
                URLQueryItem(name: $0.key, value: $0.value)
            }
        }

        guard let url = components.url else {
            throw APIError.invalidURL(path)
        }

        var request = URLRequest(url: url)
        request.httpMethod = method
        if let timeoutInterval {
            request.timeoutInterval = timeoutInterval
        }

        if let token = authManager.getToken() {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        // Required for ngrok free-tier to skip the browser interstitial page
        request.setValue("true", forHTTPHeaderField: "ngrok-skip-browser-warning")

        let requestId = UUID().uuidString.prefix(8).lowercased()
        request.setValue("ios-\(requestId)", forHTTPHeaderField: "X-Request-ID")

        logger.debug("📡 \(method, privacy: .public) \(url.path(percentEncoded: false), privacy: .public) [\(requestId, privacy: .public)]")
        return request
    }

    private func execute<T: Decodable>(_ request: URLRequest) async throws -> T {
        let urlString = sanitizedURLString(request.url)
        let path = request.url?.path(percentEncoded: false) ?? urlString
        let method = request.httpMethod ?? "?"
        let start = CFAbsoluteTimeGetCurrent()
        let requestBody = request.httpBody

        do {
            let (data, response) = try await session.data(for: request)
            let elapsed = Int((CFAbsoluteTimeGetCurrent() - start) * 1000)

            guard let http = response as? HTTPURLResponse else {
                logger.error("❌ \(method) \(urlString) — no HTTP response (\(elapsed)ms)")
                throw APIError.invalidResponse
            }

            let statusCode = http.statusCode
            guard 200..<300 ~= statusCode else {
                let body = previewString(from: data) ?? ""
                await recordNetworkEvent(
                    kind: "http",
                    method: method,
                    url: urlString,
                    path: path,
                    statusCode: statusCode,
                    durationMs: elapsed,
                    requestHeaders: sanitizedHeaders(from: request.allHTTPHeaderFields ?? [:]),
                    responseHeaders: sanitizedHeaders(from: http.allHeaderFields),
                    requestBody: requestBody,
                    responseData: data,
                    errorMessage: body
                )
                logger.error("❌ \(method) \(urlString) — HTTP \(statusCode) (\(elapsed)ms)")
                isServerReachable = true  // server is reachable, just returned an error
                throw APIError.httpError(status: statusCode, body: body)
            }

            await recordNetworkEvent(
                kind: "http",
                method: method,
                url: urlString,
                path: path,
                statusCode: statusCode,
                durationMs: elapsed,
                requestHeaders: sanitizedHeaders(from: request.allHTTPHeaderFields ?? [:]),
                responseHeaders: sanitizedHeaders(from: http.allHeaderFields),
                requestBody: requestBody,
                responseData: data,
                errorMessage: nil
            )
            logger.info("✅ \(method) \(urlString) — HTTP \(statusCode) (\(elapsed)ms) \(data.count) bytes")
            isServerReachable = true

            do {
                return try decoder.decode(T.self, from: data)
            } catch {
                logger.error("🔴 DECODE FAILED \(method) \(urlString): \(error)")
                throw APIError.decodingFailed(error)
            }
        } catch let error as APIError {
            throw error  // re-throw our own errors
        } catch {
            let elapsed = Int((CFAbsoluteTimeGetCurrent() - start) * 1000)
            await recordNetworkEvent(
                kind: "http",
                method: method,
                url: urlString,
                path: path,
                statusCode: nil,
                durationMs: elapsed,
                requestHeaders: sanitizedHeaders(from: request.allHTTPHeaderFields ?? [:]),
                responseHeaders: [:],
                requestBody: requestBody,
                responseData: nil,
                errorMessage: error.localizedDescription
            )
            logger.error("🔴 NETWORK ERROR \(method) \(urlString) (\(elapsed)ms): \(error.localizedDescription, privacy: .public)")
            // Treat cancellations as non-fatal — don't flip server reachability.
            let isCancelled = (error as? URLError)?.code == .cancelled || error is CancellationError
            if !isCancelled {
                if let urlError = error as? URLError {
                    switch urlError.code {
                    case .cannotConnectToHost, .cannotFindHost, .dnsLookupFailed, .notConnectedToInternet, .networkConnectionLost:
                        isServerReachable = false
                    default:
                        break
                    }
                }
                lastError = error.localizedDescription
            }
            throw error
        }
    }

    /// Retries GET requests on transient failures and can rebase them to a
    /// reachable fallback server if the configured endpoint is unavailable.
    private func executeWithRetry<T: Decodable>(_ request: URLRequest) async throws -> T {
        let retryDelays: [TimeInterval] = [0.4, 1.0]
        var lastError: Error?
        var currentRequest = request
        var didBootstrapFallback = false

        for attempt in 0...(retryDelays.count) {
            do {
                return try await execute(currentRequest)
            } catch let urlError as URLError {
                lastError = urlError
                if urlError.code == .timedOut && attempt < retryDelays.count {
                    let delay = retryDelays[attempt]
                    logger.warning("⏱ Timeout \(currentRequest.url?.path(percentEncoded: false) ?? "?", privacy: .public) — retry \(attempt + 1) in \(delay, format: .fixed(precision: 1))s")
                    try await Task.sleep(for: .seconds(delay))
                    continue
                }

                if shouldBootstrapFallback(after: urlError), !didBootstrapFallback {
                    let previousServerURL = activeServerURL
                    logger.warning("🔁 GET failed against \(previousServerURL, privacy: .public); probing fallback servers")
                    if await bootstrapConnection(),
                       activeServerURL != previousServerURL,
                       let fallbackRequest = rebasedRequest(currentRequest, to: activeServerURL) {
                        currentRequest = fallbackRequest
                        didBootstrapFallback = true
                        continue
                    }
                }

                throw urlError
            } catch {
                throw error
            }
        }
        throw lastError ?? APIError.invalidResponse
    }

    private func shouldBootstrapFallback(after error: URLError) -> Bool {
        switch error.code {
        case .cannotConnectToHost,
             .cannotFindHost,
             .dnsLookupFailed,
             .notConnectedToInternet,
             .timedOut,
             .dataNotAllowed:
            return true
        default:
            return false
        }
    }

    private func rebasedRequest(_ request: URLRequest, to serverURL: String) -> URLRequest? {
        guard let originalURL = request.url,
              let originalComponents = URLComponents(url: originalURL, resolvingAgainstBaseURL: false),
              var targetComponents = URLComponents(string: serverURL) else {
            return nil
        }

        targetComponents.path = originalComponents.path
        targetComponents.percentEncodedQuery = originalComponents.percentEncodedQuery
        targetComponents.fragment = nil

        guard let fallbackURL = targetComponents.url else {
            return nil
        }

        var fallbackRequest = request
        fallbackRequest.url = fallbackURL
        return fallbackRequest
    }

    private func checkHealth(at serverURL: String) async -> Bool {
        let cleanBase = serverURL
        let fullURLString = cleanBase.hasSuffix("/") ? "\(cleanBase)api/v1/health" : "\(cleanBase)/api/v1/health"
        guard let url = URL(string: fullURLString) else {
            return false
        }

        logger.info("🏥 Health check → \(url.absoluteString, privacy: .public)")

        do {
            var request = URLRequest(url: url)
            request.httpMethod = "GET"
            request.timeoutInterval = 5
            request.setValue("true", forHTTPHeaderField: "ngrok-skip-browser-warning")

            let start = CFAbsoluteTimeGetCurrent()
            let (data, response) = try await session.data(for: request)
            let elapsed = Int((CFAbsoluteTimeGetCurrent() - start) * 1000)
            let status = (response as? HTTPURLResponse)?.statusCode ?? -1
            let body = String(data: data, encoding: .utf8) ?? "(empty)"
            await recordNetworkEvent(
                kind: "health",
                method: "GET",
                url: url.absoluteString,
                path: url.path(percentEncoded: false),
                statusCode: status,
                durationMs: elapsed,
                requestHeaders: sanitizedHeaders(from: request.allHTTPHeaderFields ?? [:]),
                responseHeaders: sanitizedHeaders(from: (response as? HTTPURLResponse)?.allHeaderFields ?? [:]),
                requestBody: nil,
                responseData: data,
                errorMessage: status == 200 ? nil : body
            )
            logger.info("🏥 Health check ← HTTP \(status) body=\(body.prefix(200), privacy: .public)")

            let ok = status == 200
            isServerReachable = ok
            lastHealthCheck = Date()
            lastError = ok ? nil : "HTTP \(status)"
            return ok
        } catch {
            await recordNetworkEvent(
                kind: "health",
                method: "GET",
                url: url.absoluteString,
                path: url.path(percentEncoded: false),
                statusCode: nil,
                durationMs: 0,
                requestHeaders: [:],
                responseHeaders: [:],
                requestBody: nil,
                responseData: nil,
                errorMessage: error.localizedDescription
            )
            logger.error("🏥 Health check FAILED: \(error.localizedDescription, privacy: .public)")
            isServerReachable = false
            lastHealthCheck = Date()
            lastError = error.localizedDescription
            return false
        }
    }

    private func recordNetworkEvent(
        kind: String,
        method: String,
        url: String,
        path: String,
        statusCode: Int?,
        durationMs: Int,
        requestHeaders: [String: String],
        responseHeaders: [String: String],
        requestBody: Data?,
        responseData: Data?,
        errorMessage: String?
    ) async {
        let requestId = requestHeaders["X-Request-ID"] ?? "-"
        let event = ClientNetworkEvent(
            timestamp: Date(),
            kind: kind,
            method: method,
            url: url,
            path: path,
            statusCode: statusCode,
            durationMs: durationMs,
            requestBytes: requestBody?.count ?? 0,
            responseBytes: responseData?.count ?? 0,
            requestHeaders: requestHeaders,
            responseHeaders: responseHeaders,
            requestPreview: previewString(from: requestBody),
            responsePreview: previewString(from: responseData),
            errorMessage: errorMessage,
            requestId: requestId,
            transport: integrationTransport(for: path)
        )

        await MainActor.run {
            networkEvents.insert(event, at: 0)
            if networkEvents.count > maxNetworkEvents {
                networkEvents.removeLast(networkEvents.count - maxNetworkEvents)
            }
        }
    }

    private func previewString(from data: Data?) -> String? {
        guard let data, !data.isEmpty else { return nil }
        if let object = try? JSONSerialization.jsonObject(with: data),
           JSONSerialization.isValidJSONObject(object),
           let redactedData = try? JSONSerialization.data(withJSONObject: redactJSON(object)) {
            return truncatedPreview(String(decoding: redactedData, as: UTF8.self))
        }
        return truncatedPreview(String(decoding: data, as: UTF8.self))
    }

    private func truncatedPreview(_ string: String) -> String {
        let previewLimit = 600
        if string.utf8.count > previewLimit {
            return String(string.prefix(previewLimit)) + "…"
        }
        return string
    }

    private func redactJSON(_ value: Any, key: String? = nil) -> Any {
        let sensitiveKeys = [
            "authorization", "access_token", "refresh_token", "client_secret",
            "secret_key", "api_key", "x-client-api-key", "webhook_secret", "code"
        ]
        if let key {
            let normalized = key.lowercased().replacingOccurrences(of: "-", with: "_")
            if sensitiveKeys.contains(where: { normalized.contains($0.replacingOccurrences(of: "-", with: "_")) }) {
                return "[REDACTED]"
            }
        }
        if let dictionary = value as? [String: Any] {
            var result: [String: Any] = [:]
            for (childKey, childValue) in dictionary {
                result[childKey] = redactJSON(childValue, key: childKey)
            }
            return result
        }
        if let dictionary = value as? [AnyHashable: Any] {
            var result: [String: Any] = [:]
            for (childKey, childValue) in dictionary {
                result[String(describing: childKey)] = redactJSON(childValue, key: String(describing: childKey))
            }
            return result
        }
        if let array = value as? [Any] {
            return array.map { redactJSON($0) }
        }
        return value
    }

    private func sanitizedURLString(_ url: URL?) -> String {
        guard let url, var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else {
            return "?"
        }
        components.query = nil
        components.fragment = nil
        return components.string ?? url.path(percentEncoded: false)
    }

    private func integrationTransport(for path: String) -> IntegrationTransport {
        if path.contains("/plaud/integrations/mcp/") { return .plaudMCP }
        if path.contains("/plaud/integrations/embedded/uploads") { return .plaudUpload }
        if path.contains("/plaud/integrations/embedded/") { return .plaudEmbeddedREST }
        if path.contains("/auth/plaud") { return .plaudAccountREST }
        return .chronosREST
    }

    private func sanitizedHeaders(from headers: [AnyHashable: Any]) -> [String: String] {
        var sanitized: [String: String] = [:]

        for (key, value) in headers {
            let headerKey = String(describing: key)
            let lowercasedKey = headerKey.lowercased()
            if ["authorization", "cookie", "set-cookie", "x-api-key", "x-client-api-key", "x-client-secret"].contains(lowercasedKey) {
                sanitized[headerKey] = "<redacted>"
            } else {
                sanitized[headerKey] = String(describing: value)
            }
        }

        return sanitized
    }
}

// MARK: - API Error

enum APIError: Error, LocalizedError {
    case invalidURL(String)
    case invalidResponse
    case httpError(status: Int, body: String)
    case decodingFailed(Error)
    case unauthorized

    var errorDescription: String? {
        switch self {
        case .invalidURL(let path):
            return "Invalid URL: \(path)"
        case .invalidResponse:
            return "Invalid server response"
        case .httpError(let status, let body):
            return "HTTP \(status): \(body.prefix(200))"
        case .decodingFailed(let error):
            return "Decoding failed: \(error.localizedDescription)"
        case .unauthorized:
            return "Authentication required"
        }
    }
}
