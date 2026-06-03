import Foundation
import AuthenticationServices
import Observation

@MainActor
@Observable
final class SettingsViewModel: NSObject {
    var serverURL: String = ""
    var apiToken: String = ""
    var plaudStatus: TokenStatus?
    var plaudValidationDiagnostics: PlaudStatusValidationDiagnostics?
    var notionAuthStatus: TokenStatus?
    var systemStatus: SystemStatus?
    var processingProvider = ""
    var cleaningModel = ""
    var analystModel = ""
    var embeddingModel = ""
    var openAIModel = ""
    var thinkingLevel = ""
    var openAITemperature = ""
    var embeddingDim = ""
    var plaudLanguage = ""
    var plaudDiarization = true
    var logLevel = ""
    var customCategories = ""
    var notionWeekdayStart = ""
    var notionWeekendStart = ""
    var qdrantURL = ""
    var qdrantCollectionName = ""
    var chronosOpenAIEnabled = false
    var chronosLocalLLMEnabled = false
    var chronosLocalLLMProvider = ""
    var chronosLocalLLMBaseURL = ""
    var chronosLocalLLMModel = ""
    var chronosLocalLLMMaxContext = ""
    var chronosLocalLLMAllowedTasks = ""
    var supportsServerSettingsEndpoint = true
    var serverConfigMessage: String?
    var serverConfigNotice: String?
    var isLoadingServerConfig = false
    var isSavingServerConfig = false
    var hasGeminiAPIKey = false
    var hasOpenAIAPIKey = false
    var hasQdrantAPIKey = false
    var hasNotionToken = false
    var hasNotionOAuth = false
    var isServerReachable = false
    var isCheckingServer = false
    var isLoadingSystemStatus = false
    var isAuthorizingPlaud = false
    var error: String?

    var supportsOpenAITemperatureControl: Bool {
        let model = openAIModel.trimmingCharacters(in: .whitespacesAndNewlines)
        return AskChronosSettings.supportsTemperatureControl(for: model)
    }

    private let api: APIClient
    private let authManager: AuthManager
    @ObservationIgnored private var plaudStatusValidation = PlaudStatusValidationCache()
    @ObservationIgnored private var webAuthCoordinator: PlaudWebAuthCoordinator?

    init(api: APIClient, authManager: AuthManager) {
        self.api = api
        self.authManager = authManager
        self.serverURL = authManager.serverURL
        super.init()
    }

    func checkServer() async {
        isCheckingServer = true
        isServerReachable = await api.healthCheck()
        isCheckingServer = false
    }

    func saveServerURL() {
        do {
            try authManager.setServerURL(serverURL)
            serverURL = authManager.serverURL
        } catch {
            self.error = error.localizedDescription
        }
    }

    func saveToken() {
        guard !apiToken.isEmpty else { return }
        do {
            try authManager.setToken(apiToken)
            apiToken = ""
        } catch {
            self.error = error.localizedDescription
        }
    }

    func logout() {
        authManager.logout()
    }

    func loadPlaudStatus(forceRefresh: Bool = false) async {
        let now = Date()
        var validation = plaudStatusValidation

        do {
            let result = try await validation.load(
                forceRefresh: forceRefresh,
                now: now
            ) {
                try await api.get("/api/auth/plaud/status")
            }
            plaudStatusValidation = validation
            applyPlaudStatusValidation(result)
        } catch is CancellationError {
            return
        } catch let urlError as URLError where urlError.code == .cancelled {
            return
        } catch {
            if let recovered = validation.recoverFromTransientFailure(error, now: now) {
                plaudStatusValidation = validation
                applyPlaudStatusValidation(recovered)
                return
            }

            let invalidation = validation.requireReauthentication(
                message: plaudHardFailureMessage(for: error),
                now: now
            )
            plaudStatusValidation = validation
            applyPlaudStatusValidation(invalidation)
        }
    }

    func startPlaudOAuthFlow(anchor: ASPresentationAnchor) async {
        isAuthorizingPlaud = true
        error = nil
        plaudStatusValidation.invalidate()
        plaudValidationDiagnostics = nil
        defer {
            isAuthorizingPlaud = false
            webAuthCoordinator = nil
        }

        do {
            // Always request with mobile=true so the server uses the deep-link callback
            // rather than a desktop redirect_uri.
            let response: AuthURLResponse = try await api.get(
                "/api/auth/plaud/authorize",
                query: ["mobile": "true"]
            )
            guard let authURL = URL(string: response.authUrl) else {
                error = "Invalid Plaud authorization URL"
                return
            }

            // Present OAuth in-app via ASWebAuthenticationSession.
            // The session calls back to plaudblender://plaud-callback
            let callbackURL = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<URL, Error>) in
                let coordinator = PlaudWebAuthCoordinator(
                    url: authURL,
                    callbackScheme: "plaudblender",
                    anchor: anchor
                ) { callbackURL, sessionError in
                    if let sessionError {
                        continuation.resume(throwing: sessionError)
                    } else if let url = callbackURL {
                        continuation.resume(returning: url)
                    } else {
                        continuation.resume(throwing: APIError.invalidResponse)
                    }
                }
                webAuthCoordinator = coordinator
                coordinator.session.start()
            }

            // Parse plaudblender://plaud-callback?success=true or ?error=...
            let queryItems = URLComponents(url: callbackURL, resolvingAgainstBaseURL: false)?.queryItems ?? []
            if let errorMsg = queryItems.first(where: { $0.name == "error" })?.value, !errorMsg.isEmpty {
                error = "Plaud authorization failed: \(errorMsg)"
                return
            }
            guard queryItems.first(where: { $0.name == "success" })?.value == "true" else {
                error = "Plaud authorization was not confirmed. Please try again."
                return
            }

            // Poll /api/auth/plaud/status for up to 10 seconds (20 × 500 ms)
            for _ in 0..<20 {
                try await Task.sleep(for: .milliseconds(500))
                await loadPlaudStatus(forceRefresh: true)
                if plaudStatus?.isAuthenticated == true {
                    error = nil
                    await loadSystemStatus()
                    return
                }
            }
            error = "Plaud authorization completed but status confirmation timed out. Try refreshing Settings."

        } catch let authError as ASWebAuthenticationSessionError where authError.code == .canceledLogin {
            error = nil  // user cancelled — not an error
        } catch is CancellationError {
            error = nil
        } catch {
            self.error = error.localizedDescription
        }
    }

    func refreshAfterPlaudAuthorization() async {
        await loadPlaudStatus(forceRefresh: true)
        await loadSystemStatus()
    }

    func loadNotionStatus() async {
        do {
            notionAuthStatus = try await api.get("/api/auth/notion/status")
        } catch {
            notionAuthStatus = nil
        }
    }

    func loadSystemStatus() async {
        isLoadingSystemStatus = true
        do {
            systemStatus = try await api.get("/api/status")
        } catch {
            systemStatus = nil
        }
        isLoadingSystemStatus = false
    }

    func loadServerSettings() async {
        isLoadingServerConfig = true
        do {
            let settings: ServerSettings = try await api.get("/api/settings")
            supportsServerSettingsEndpoint = true
            serverConfigNotice = nil
            processingProvider = settings.processingProvider
            cleaningModel = settings.cleaningModel
            analystModel = settings.analystModel
            embeddingModel = settings.embeddingModel
            openAIModel = settings.openAIModel
            thinkingLevel = settings.thinkingLevel
            openAITemperature = String(settings.openAITemperature)
            embeddingDim = String(settings.embeddingDim)
            plaudLanguage = settings.plaudLanguage
            plaudDiarization = settings.plaudDiarization
            logLevel = settings.logLevel
            customCategories = settings.customCategories
            notionWeekdayStart = settings.notionWeekdayStart
            notionWeekendStart = settings.notionWeekendStart
            qdrantURL = settings.qdrantURL
            qdrantCollectionName = settings.qdrantCollectionName
            chronosOpenAIEnabled = settings.chronosOpenAIEnabled
            chronosLocalLLMEnabled = settings.chronosLocalLLMEnabled
            chronosLocalLLMProvider = settings.chronosLocalLLMProvider
            chronosLocalLLMBaseURL = settings.chronosLocalLLMBaseURL
            chronosLocalLLMModel = settings.chronosLocalLLMModel
            chronosLocalLLMMaxContext = String(settings.chronosLocalLLMMaxContext)
            chronosLocalLLMAllowedTasks = settings.chronosLocalLLMAllowedTasks
            hasGeminiAPIKey = settings.flags.hasGeminiAPIKey
            hasOpenAIAPIKey = settings.flags.hasOpenAIAPIKey
            hasQdrantAPIKey = settings.flags.hasQdrantAPIKey
            hasNotionToken = settings.flags.hasNotionToken
            hasNotionOAuth = settings.flags.hasNotionOAuth
        } catch {
            if isNotFound(error) {
                supportsServerSettingsEndpoint = false
                serverConfigNotice = "This backend does not expose `/api/v1/settings` yet, so server-side model and runtime configuration is unavailable here."
            } else {
                self.error = error.localizedDescription
            }
        }
        isLoadingServerConfig = false
    }

    func saveServerSettings() async {
        guard supportsServerSettingsEndpoint else {
            serverConfigNotice = "This backend does not expose `/api/v1/settings` yet, so there is nothing to save from Settings."
            return
        }
        guard let openAITemperatureValue = Double(openAITemperature) else {
            error = "OpenAI temperature must be a number"
            return
        }
        guard let embeddingDimValue = Int(embeddingDim) else {
            error = "Embedding dimension must be a whole number"
            return
        }
        guard let localLLMMaxContextValue = Int(chronosLocalLLMMaxContext) else {
            error = "Local LLM max context must be a whole number"
            return
        }

        isSavingServerConfig = true
        error = nil
        serverConfigMessage = nil

        do {
            let body = ServerSettingsUpdateRequest(
                processingProvider: processingProvider,
                cleaningModel: cleaningModel,
                analystModel: analystModel,
                embeddingModel: embeddingModel,
                openAIModel: openAIModel,
                thinkingLevel: thinkingLevel,
                openAITemperature: openAITemperatureValue,
                embeddingDim: embeddingDimValue,
                plaudLanguage: plaudLanguage,
                plaudDiarization: plaudDiarization,
                logLevel: logLevel,
                customCategories: customCategories,
                notionWeekdayStart: notionWeekdayStart,
                notionWeekendStart: notionWeekendStart,
                qdrantURL: qdrantURL,
                qdrantCollectionName: qdrantCollectionName,
                chronosOpenAIEnabled: chronosOpenAIEnabled,
                chronosLocalLLMEnabled: chronosLocalLLMEnabled,
                chronosLocalLLMProvider: chronosLocalLLMProvider,
                chronosLocalLLMBaseURL: chronosLocalLLMBaseURL,
                chronosLocalLLMModel: chronosLocalLLMModel,
                chronosLocalLLMMaxContext: localLLMMaxContextValue,
                chronosLocalLLMAllowedTasks: chronosLocalLLMAllowedTasks
            )
            let response: SuccessResponse = try await api.put("/api/settings", body: body)
            serverConfigMessage = response.message
            await loadServerSettings()
        } catch {
            self.error = error.localizedDescription
        }

        isSavingServerConfig = false
    }

    func loadAll() async {
        await checkServer()
        await loadPlaudStatus()
        await loadNotionStatus()
        await loadSystemStatus()
        await loadServerSettings()
    }

    private func applyPlaudStatusValidation(_ result: PlaudStatusValidationResult) {
        plaudStatus = result.status
        plaudValidationDiagnostics = result.diagnostics
    }

    private func plaudHardFailureMessage(for error: Error) -> String {
        if case APIError.httpError(let status, _) = error {
            switch status {
            case 401, 403:
                return "Plaud session is no longer valid on the server. Reconnect Plaud."
            case 404:
                return "This backend does not expose Plaud auth status yet."
            default:
                return "Plaud validation could not be confirmed. Reconnect Plaud to refresh the server-side token."
            }
        }

        return "Plaud validation could not be confirmed. Reconnect Plaud to refresh the server-side token."
    }

    private func isNotFound(_ error: Error) -> Bool {
        guard case APIError.httpError(let status, _) = error else {
            return false
        }
        return status == 404
    }

}

// MARK: - ASWebAuthenticationSession coordinator
// Keeps the session and presentation-context provider alive for the OAuth round-trip.
private final class PlaudWebAuthCoordinator: NSObject, ASWebAuthenticationPresentationContextProviding {
    let session: ASWebAuthenticationSession
    private let anchor: ASPresentationAnchor

    init(
        url: URL,
        callbackScheme: String,
        anchor: ASPresentationAnchor,
        completionHandler: @escaping ASWebAuthenticationSession.CompletionHandler
    ) {
        self.anchor = anchor
        self.session = ASWebAuthenticationSession(
            url: url,
            callbackURLScheme: callbackScheme,
            completionHandler: completionHandler
        )
        super.init()
        session.presentationContextProvider = self
        session.prefersEphemeralWebBrowserSession = true
    }

    func presentationAnchor(for session: ASWebAuthenticationSession) -> ASPresentationAnchor {
        anchor
    }
}
