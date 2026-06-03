import Foundation
import AuthenticationServices
import Observation

@MainActor
@Observable
final class NotionViewModel: NSObject {
    var authStatus: TokenStatus?
    var status: NotionStatus?
    var databases: [NotionDatabase] = []
    var recordings: [NotionRecording] = []
    var importProgress: NotionImportProgress?
    var coverage: NotionCoverageResponse?
    var isLoading = false
    var isLoadingMore = false
    var isAuthorizing = false
    var importPreview: NotionImportPreview?
    var matchReview: NotionMatchReview?
    var supportsSafeBatchImport = true
    var isImporting = false
    var isRunningImportSweep = false
    var selectedDatabaseId: String?
    var error: String?
    var lastMessage: String?
    var totalRecordings = 0
    var hasMoreRecordings = false

    private let pageSize = 100

    private let api: APIClient
    @ObservationIgnored private var importMonitorTask: Task<Void, Never>?
    @ObservationIgnored private var stableImportPollCount = 0
    @ObservationIgnored private var lastImportSignature = ""
    /// Anchor for ASWebAuthenticationSession presentation
    @ObservationIgnored private var presentationAnchor: ASPresentationAnchor?

    init(api: APIClient) {
        self.api = api
        super.init()
    }

    deinit {
        importMonitorTask?.cancel()
    }

    var isAuthenticated: Bool {
        authStatus?.isAuthenticated == true || status?.isConnected == true
    }

    var workspaceName: String? {
        authStatus?.workspaceName ?? authStatus?.extra?["workspace_name"]?.stringValue
    }

    var authMode: String {
        authStatus?.extra?["auth_mode"]?.stringValue ?? "none"
    }

    var usesOAuth: Bool {
        authMode == "oauth"
    }

    var usesIntegrationToken: Bool {
        authMode == "integration_token"
    }

    var hasSelectedDatabase: Bool {
        status?.isConnected == true
    }

    var unmatchedCount: Int {
        recordings.filter { !$0.isImportedToChronos }.count
    }

    var importedCount: Int {
        recordings.filter(\.isImportedToChronos).count
    }

    var serverPendingImport: Int {
        importPreview?.pendingImport ?? unmatchedCount
    }

    var totalKnownPages: Int {
        max(importPreview?.totalPages ?? 0, status?.totalPages ?? 0, totalRecordings)
    }

    var matchedOrImportedCount: Int {
        importPreview?.matchedToExisting ?? importedCount
    }

    var serverPendingImportRaw: Int {
        importPreview?.pendingImportRaw ?? serverPendingImport
    }

    var duplicatePagesCollapsed: Int {
        importPreview?.duplicatePagesCollapsed ?? 0
    }

    var matchedToExistingCount: Int {
        importPreview?.matchedToExisting ?? importedCount
    }

    var safeBatchLimit: Int {
        max(1, importPreview?.blindImportLimit ?? 25)
    }

    var duplicateGroups: [NotionDuplicateGroup] {
        matchReview?.duplicateGroups ?? []
    }

    var highConfidenceTranscriptAliases: [NotionTranscriptAliasCandidate] {
        matchReview?.highConfidenceTranscriptAliases ?? []
    }

    var manualOverrides: [String: String] {
        matchReview?.manualOverrides ?? [:]
    }

    var manualOverrideCount: Int {
        matchReview?.manualOverrideCount ?? manualOverrides.count
    }

    var duplicateGroupCount: Int {
        matchReview?.duplicateGroupCount ?? duplicateGroups.count
    }

    var shouldShowDatabasePicker: Bool {
        isAuthenticated && !hasSelectedDatabase
    }

    var notionStatusSummary: String {
        if let status, status.isConnected {
            return "\(status.totalPages) page\(status.totalPages == 1 ? "" : "s") in \(status.databaseTitle ?? "selected database")"
        }
        if let workspaceName {
            return workspaceName
        }
        return "Not connected"
    }

    var authModeLabel: String {
        switch authMode {
        case "oauth":
            return "Connected via OAuth"
        case "integration_token":
            return "Using server integration token"
        default:
            return "Not connected"
        }
    }

    func loadAuthStatus() async {
        do {
            authStatus = try await api.get("/api/auth/notion/status")
        } catch {
            self.error = error.localizedDescription
        }
    }

    func loadImportPreview() async {
        do {
            importPreview = try await api.get("/api/notion/import/preview")
            supportsSafeBatchImport = true
        } catch {
            if isNotFound(error) {
                supportsSafeBatchImport = false
            }
            importPreview = nil
        }
    }

    func loadMatchReview(limit: Int = 10) async {
        do {
            matchReview = try await api.get(
                "/api/notion/match/review",
                query: ["limit": "\(limit)"]
            )
        } catch {
            matchReview = nil
        }
    }

    func startImport(process: Bool = true, index: Bool = true, batchSize: Int? = nil) async -> Bool {
        Haptics.impact()
        isImporting = true
        error = nil
        do {
            let body = NotionImportRequest(
                process: process,
                index: index,
                batchSize: batchSize ?? safeBatchLimit,
                force: nil
            )
            let response: SuccessResponse = try await api.post("/api/notion/import/next-batch", body: body)
            supportsSafeBatchImport = true
            lastMessage = response.message
            await loadImportProgress()
            await loadImportPreview()
            startImportMonitoring()
            return response.success
        } catch {
            if isNotFound(error) {
                supportsSafeBatchImport = false
                return await startLegacyImport(process: process, index: index)
            }
            self.error = error.localizedDescription
            isImporting = false
            return false
        }
    }

    func importAllPending(process: Bool = true, index: Bool = true) async -> Bool {
        guard !isRunningImportSweep else { return false }

        isRunningImportSweep = true
        defer { isRunningImportSweep = false }

        error = nil

        await loadImportPreview()

        guard supportsSafeBatchImport else {
            return await startLegacyImport(process: process, index: index)
        }

        let initialPending = serverPendingImport
        guard initialPending > 0 else {
            lastMessage = "No pending Notion pages remain to import."
            return true
        }

        let estimatedBatchCount = Int(ceil(Double(initialPending) / Double(max(safeBatchLimit, 1))))
        let maxBatches = max(1, min(estimatedBatchCount + 2, 50))

        var previousPending = initialPending
        var completedBatches = 0

        while completedBatches < maxBatches, previousPending > 0 {
            let started = await startImport(process: process, index: index, batchSize: safeBatchLimit)
            guard started else { return false }

            let settled = await waitForImportToSettle()
            guard settled else {
                error = "Timed out waiting for Notion import batch \(completedBatches + 1) to finish."
                return false
            }

            await refreshReconciliationState()

            let newPending = serverPendingImport
            completedBatches += 1

            if newPending <= 0 {
                lastMessage = "Imported all pending Notion pages in \(completedBatches) batch\(completedBatches == 1 ? "" : "es")."
                return true
            }

            if newPending >= previousPending {
                lastMessage = "Import sweep stopped after \(completedBatches) batch\(completedBatches == 1 ? "" : "es"); pending pages did not decrease further."
                return false
            }

            previousPending = newPending
        }

        if serverPendingImport > 0 {
            lastMessage = "Import sweep stopped with \(serverPendingImport) pending Notion pages still remaining."
            return false
        }

        return true
    }

    func applyManualOverride(pageId: String, recordingId: String) async -> Bool {
        let trimmed = recordingId.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            error = "Recording ID is required"
            return false
        }

        do {
            let body = NotionMatchOverrideRequest(pageId: pageId, recordingId: trimmed, clear: false)
            let response: SuccessResponse = try await api.post("/api/notion/match/override", body: body)
            lastMessage = response.message
            await refreshReconciliationState()
            return response.success
        } catch {
            self.error = error.localizedDescription
            return false
        }
    }

    func clearManualOverride(pageId: String) async -> Bool {
        do {
            let body = NotionMatchOverrideRequest(pageId: pageId, recordingId: nil, clear: true)
            let response: SuccessResponse = try await api.post("/api/notion/match/override", body: body)
            lastMessage = response.message
            await refreshReconciliationState()
            return response.success
        } catch {
            self.error = error.localizedDescription
            return false
        }
    }

    func applyBulkOverride(pageIds: [String], recordingId: String) async -> Bool {
        let trimmed = recordingId.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            error = "Recording ID is required"
            return false
        }

        do {
            let overrides = pageIds.map {
                NotionMatchOverrideRequest(pageId: $0, recordingId: trimmed, clear: false)
            }
            let body = NotionBulkMatchOverrideRequest(overrides: overrides)
            let response: NotionBulkMatchOverrideResponse = try await api.post("/api/notion/match/override/bulk", body: body)
            if response.failed > 0 {
                lastMessage = "Applied \(response.applied) overrides, \(response.failed) failed"
                error = response.results.first(where: { !$0.ok })?.message
            } else {
                lastMessage = "Applied \(response.applied) overrides"
            }
            await refreshReconciliationState()
            return response.failed == 0
        } catch {
            self.error = error.localizedDescription
            return false
        }
    }

    func loadStatus() async {
        do {
            status = try await api.get("/api/notion/status")
        } catch {
            self.error = error.localizedDescription
        }
    }

    func loadDatabases() async {
        do {
            databases = try await api.get("/api/notion/databases")
            await loadImportPreview()
            await loadMatchReview()
        } catch {
            self.error = error.localizedDescription
        }
    }


    func selectDatabase(dbId: String) async -> Bool {
        do {
            selectedDatabaseId = dbId
            let body = NotionDatabaseSelectRequest(dbId: dbId)
            let _: SuccessResponse = try await api.post("/api/notion/databases/select", body: body)
            await loadStatus()
            await loadRecordings()
            lastMessage = "Notion database selected"
            return true
        } catch {
            self.error = error.localizedDescription
            return false
        }
    }

    func loadRecordings() async {
        do {
            let response: NotionRecordingsResponse = try await api.get(
                "/api/notion/recordings",
                query: ["limit": "\(pageSize)", "offset": "0"]
            )
            recordings = response.recordings
            totalRecordings = response.total
            hasMoreRecordings = response.hasMore
        } catch {
            self.error = error.localizedDescription
        }
    }

    func loadMoreRecordings() async {
        guard !isLoadingMore, hasMoreRecordings else { return }
        isLoadingMore = true
        defer { isLoadingMore = false }
        do {
            let response: NotionRecordingsResponse = try await api.get(
                "/api/notion/recordings",
                query: ["limit": "\(pageSize)", "offset": "\(recordings.count)"]
            )
            recordings.append(contentsOf: response.recordings)
            totalRecordings = response.total
            hasMoreRecordings = response.hasMore
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func startLegacyImport(process: Bool = true, index: Bool = true) async -> Bool {
        Haptics.impact()
        do {
            let body = NotionImportRequest(process: process, index: index)
            let response: SuccessResponse = try await api.post("/api/notion/import", body: body)
            lastMessage = response.message
            await loadImportProgress()
            startImportMonitoring()
            return response.success
        } catch {
            self.error = error.localizedDescription
            isImporting = false
            return false
        }
    }

    func loadImportProgress() async {
        do {
            importProgress = try await api.get("/api/notion/import/progress")
            updateImportingState()
        } catch {
            importProgress = nil
            updateImportingState()
        }
    }

    func loadCoverage() async {
        do {
            coverage = try await api.get("/api/notion/coverage")
        } catch {
            coverage = nil
        }
    }

    func loadAll() async {
        isLoading = true
        error = nil
        await loadAuthStatus()
        await loadStatus()
        if shouldShowDatabasePicker {
            await loadDatabases()
        }
        if hasSelectedDatabase {
            await loadRecordings()
            await loadImportProgress()
            await loadImportPreview()
            await loadMatchReview()
            await loadCoverage()
        }
        isLoading = false
    }

    func startOAuthFlow(anchor: ASPresentationAnchor) async {
        isAuthorizing = true
        error = nil
        defer { isAuthorizing = false }

        do {
            // 1. Get the authorize URL from backend (mobile mode → redirect goes through our callback)
            let response: AuthURLResponse = try await api.get("/api/auth/notion/authorize", query: ["mobile": "true"])
            guard let authURL = URL(string: response.authUrl) else {
                error = "Invalid authorization URL"
                return
            }

            // 2. Open ASWebAuthenticationSession — it catches the plaudblender:// redirect
            presentationAnchor = anchor
            let callbackURL = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<URL, Error>) in
                let session = ASWebAuthenticationSession(
                    url: authURL,
                    callbackURLScheme: "plaudblender"
                ) { url, error in
                    if let error {
                        continuation.resume(throwing: error)
                    } else if let url {
                        continuation.resume(returning: url)
                    } else {
                        continuation.resume(throwing: URLError(.cancelled))
                    }
                }
                session.presentationContextProvider = self
                session.prefersEphemeralWebBrowserSession = false
                session.start()
            }

            // 3. Parse the callback — the backend already exchanged the code
            let components = URLComponents(url: callbackURL, resolvingAgainstBaseURL: false)
            let queryItems = (components?.queryItems ?? []).compactMap { item -> (String, String)? in
                guard let value = item.value else { return nil }
                return (item.name, value)
            }
            let params: [String: String] = Dictionary(uniqueKeysWithValues: queryItems)

            if let callbackError = params["error"] {
                error = "Notion authorization failed: \(callbackError)"
            } else {
                // Success — backend exchanged the token, refresh our status
                lastMessage = "Notion connected!"
                await loadAll()
            }
        } catch let err as ASWebAuthenticationSessionError where err.code == .canceledLogin {
            // User cancelled — not an error
        } catch {
            self.error = error.localizedDescription
        }
    }

    func refreshAfterAuthorization() async {
        await loadAll()
    }

    private func updateImportingState() {
        let status = importProgress?.status.lowercased() ?? "idle"
        isImporting = status == "running" || status == "paused"
        if isImporting {
            startImportMonitoring()
        } else {
            stopImportMonitoring()
        }
    }

    private func startImportMonitoring() {
        guard importMonitorTask == nil else { return }
        stableImportPollCount = 0
        lastImportSignature = importProgressSignature
        importMonitorTask = Task { [weak self] in
            guard let self else { return }
            await self.monitorImportProgress()
        }
    }

    private func stopImportMonitoring() {
        importMonitorTask?.cancel()
        importMonitorTask = nil
        stableImportPollCount = 0
        lastImportSignature = ""
    }

    private var importProgressSignature: String {
        [
            importProgress?.status ?? "-",
            "\(importProgress?.total ?? -1)",
            "\(importProgress?.completed ?? -1)",
            "\(importProgress?.failed ?? -1)",
            "\(importProgress?.skipped ?? -1)",
            importProgress?.currentTitle ?? "-"
        ].joined(separator: "|")
    }

    private func nextImportPollDelayNanoseconds() -> UInt64 {
        switch stableImportPollCount {
        case 0:
            return 750_000_000
        case 1:
            return 1_500_000_000
        default:
            return 3_000_000_000
        }
    }

    private func waitForImportToSettle(timeoutNanoseconds: UInt64 = 240_000_000_000) async -> Bool {
        let start = DispatchTime.now().uptimeNanoseconds

        while DispatchTime.now().uptimeNanoseconds - start < timeoutNanoseconds {
            await loadImportProgress()

            if !isImporting {
                return true
            }

            try? await Task.sleep(nanoseconds: nextImportPollDelayNanoseconds())
        }

        return !isImporting
    }

    private func monitorImportProgress() async {
        defer { importMonitorTask = nil }

        while !Task.isCancelled {
            await loadImportProgress()

            if isImporting {
                let signature = importProgressSignature
                if signature == lastImportSignature {
                    stableImportPollCount = min(stableImportPollCount + 1, 2)
                } else {
                    stableImportPollCount = 0
                    lastImportSignature = signature
                }

                try? await Task.sleep(nanoseconds: nextImportPollDelayNanoseconds())
            } else {
                await loadRecordings()
                await loadImportPreview()
                await loadMatchReview()
                break
            }
        }
    }

    private func refreshReconciliationState() async {
        await loadRecordings()
        await loadImportPreview()
        await loadMatchReview()
    }

    private func isNotFound(_ error: Error) -> Bool {
        guard case APIError.httpError(let status, _) = error else {
            return false
        }
        return status == 404
    }
}

// MARK: - ASWebAuthenticationPresentationContextProviding

extension NotionViewModel: ASWebAuthenticationPresentationContextProviding {
    nonisolated func presentationAnchor(for session: ASWebAuthenticationSession) -> ASPresentationAnchor {
        MainActor.assumeIsolated {
            if let anchor = presentationAnchor { return anchor }
            guard let anchor = currentPresentationAnchor() else {
                preconditionFailure("No presentation anchor available for Notion OAuth")
            }
            return anchor
        }
    }
}
