import Foundation
import Observation

@MainActor
@Observable
final class SyncViewModel {
    struct BackfillFailureDetails: Equatable {
        let statusCode: Int
        let summary: String
        let indicatesUnsupportedStage: Bool
    }

    var pipelineStatus: PipelineStatus?
    var dbStats: RecordingDbStats?
    var workflowStats: WorkflowStats?
    var uploadCandidates: [UploadCandidate] = []
    var syncFailures: SyncFailureSummary?
    var systemStatus: SystemStatus?
    var stackControl: StackControlResponse?
    var backups: [AdminBackupInfo] = []
    var downloadedBackupURL: URL?
    var supportsBackfillStage = true
    var supportsSyncFailuresEndpoint = true
    var supportsAdminEndpoints = true
    var supportsUploadProcessEndpoint = true
    var isLoadingSystemStatus = false
    var isLoading = false
    var isRunning = false
    var isRunningStackAction = false
    var isUploadingCandidates = false
    var isCreatingBackup = false
    var hasStaleData = false
    var error: String?
    var lastMessage: String?
    var lastUpdated: Date?
    var activeTraceRunId: String?
    var hasBootstrapped = false
    var showBatchWorkflowSheet = false

    private let api: APIClient
    @ObservationIgnored private var monitorTask: Task<Void, Never>?
    @ObservationIgnored private var refreshTask: Task<Void, Never>?
    @ObservationIgnored private var stablePollCount = 0
    @ObservationIgnored private var lastMonitorSignature = ""

    init(api: APIClient) {
        self.api = api
    }

    static func backfillFailureDetails(from error: Error) -> BackfillFailureDetails? {
        guard case APIError.httpError(let status, let body) = error else {
            return nil
        }

        let summary = summarizeBackfillErrorBody(body)
        let normalizedBody = collapseWhitespace(body).lowercased()
        let normalizedSummary = summary.lowercased()
        let mentionsStage = normalizedBody.contains("stage") || normalizedSummary.contains("stage")
        let mentionsBackfillCandidate = ["backfill", "all_history", "full_history"].contains {
            normalizedBody.contains($0) || normalizedSummary.contains($0)
        }
        let hasUnsupportedMarker = [
            "input should",
            "unexpected value",
            "not a valid",
            "unsupported",
            "unknown",
            "invalid"
        ].contains {
            normalizedBody.contains($0) || normalizedSummary.contains($0)
        }
        let hasEnumMarker = normalizedBody.contains("enum") || normalizedSummary.contains("enum")
        let indicatesUnsupportedStage = [400, 404, 422].contains(status)
            && mentionsStage
            && mentionsBackfillCandidate
            && (hasUnsupportedMarker || hasEnumMarker)

        return BackfillFailureDetails(
            statusCode: status,
            summary: summary,
            indicatesUnsupportedStage: indicatesUnsupportedStage
        )
    }

    deinit {
        monitorTask?.cancel()
    }

    var currentStageTitle: String {
        let stage = pipelineStatus?.currentPhase ?? pipelineStatus?.activePhase?.name ?? pipelineStatus?.status ?? "idle"
        return stage
            .replacingOccurrences(of: "_", with: " ")
            .capitalized
    }

    var progressEntries: [(key: String, value: String)] {
        guard let status = pipelineStatus, let activePhase = status.activePhase else {
            return []
        }

        var entries: [(key: String, value: String)] = []

        if activePhase.totalItems > 0 {
            entries.append(("Completed", "\(activePhase.completedItems)/\(activePhase.totalItems)"))
        }

        if !activePhase.currentStep.isEmpty {
            entries.append(("Step", activePhase.currentStep))
        }

        if !activePhase.currentItem.isEmpty {
            entries.append(("Item", activePhase.currentItem))
        }

        if activePhase.elapsedSeconds > 0 {
            entries.append(("Phase Time", TimeInterval(activePhase.elapsedSeconds).durationFormatted))
        }

        if let elapsedSeconds = status.elapsedSeconds, elapsedSeconds > 0 {
            entries.append(("Run Time", TimeInterval(elapsedSeconds).durationFormatted))
        }

        if let trigger = status.trigger, !trigger.isEmpty {
            entries.append(("Trigger", trigger.capitalized))
        }

        return entries
    }

    var globalBannerTitle: String {
        isRunning ? "Pipeline In Progress" : "Sync Idle"
    }

    var globalBannerDetail: String {
        if isRunning {
            if let dbStats {
                return "\(currentStageTitle) · \(dbStats.processing) running · \(dbStats.pending) pending · \(dbStats.completed) done"
            }
            if let lead = progressEntries.first {
                return "\(currentStageTitle) · \(lead.key): \(lead.value)"
            }
            return currentStageTitle
        }

        if let updated = lastUpdated {
            return "Last updated \(updated.relativeString)"
        }

        return lastMessage ?? "No active sync"
    }

    var shouldShowGlobalBanner: Bool {
        isRunning || hasVisibleWorkInProgress
    }

    /// Number of healthy/safely-disabled services out of the core set.
    var healthyServiceCount: Int {
        guard let sys = systemStatus else { return 0 }
        var count = 0
        if sys.database?.ok == true { count += 1 }
        if sys.qdrant?.ok == true { count += 1 }
        if sys.gemini?.isSafeOrOperational == true { count += 1 }
        if sys.openai?.isSafeOrOperational == true { count += 1 }
        if sys.plaud?.isUp == true { count += 1 }
        if sys.notion?.isUp == true { count += 1 }
        if sys.localLLM?.isUp == true { count += 1 }
        return count
    }

    /// True when all pipeline-critical services (Database, Qdrant, at least one AI) are healthy
    var pipelineReady: Bool {
        guard let sys = systemStatus else { return false }
        let dbOk = sys.database?.ok == true
        let qdrantOk = sys.qdrant?.ok == true
        let aiOk = (sys.gemini?.isUp == true) || (sys.localLLM?.isUp == true) || (sys.openai?.isOperational == true)
        return dbOk && qdrantOk && aiOk
    }

    var hasVisibleWorkInProgress: Bool {
        guard isRunning else { return false }
        return (dbStats?.processing ?? 0) > 0
            || (dbStats?.pending ?? 0) > 0
            || !progressEntries.isEmpty
    }

    var actionableFailureCount: Int {
        if let syncFailures {
            return syncFailures.actionableCount
        }
        return dbStats?.failed ?? 0
    }

    var hasFailureBreakdown: Bool {
        syncFailures != nil
    }

    var shouldShowRetryableFailures: Bool {
        actionableFailureCount > 0
    }

    var plaudSyncNotices: [PlaudSyncNotice] {
        PlaudSyncNoticeBuilder.build(
            status: pipelineStatus,
            dbStats: dbStats,
            failures: syncFailures
        )
    }

    func bootstrap() async {
        hasBootstrapped = true
        await loadAll()
    }

    func bootstrapIfNeeded() async {
        guard !hasBootstrapped else { return }
        await bootstrap()
    }

    func loadStatus(showLoading: Bool = true) async {
        if showLoading {
            isLoading = true
        }
        do {
            let status: PipelineStatus = try await api.get("/api/sync/status")
            pipelineStatus = status
            lastUpdated = Date()
            hasStaleData = false
            updateRunningState(from: status)
        } catch is CancellationError {
            // Task was cancelled — do not surface as an error or kill monitoring
            return
        } catch let urlError as URLError where urlError.code == .cancelled {
            return
        } catch {
            // Keep last known pipelineStatus — mark as stale rather than hard-failing.
            if pipelineStatus != nil {
                hasStaleData = true
            } else {
                self.error = error.localizedDescription
            }
            if showLoading {
                isLoading = false
            }
            stopMonitoring()
            return
        }

        if showLoading {
            isLoading = false
        }
    }

    func loadDbStats() async {
        do {
            dbStats = try await api.get("/api/sync/db-stats")
            hasStaleData = false
        } catch is CancellationError {
            return
        } catch let urlError as URLError where urlError.code == .cancelled {
            return
        } catch {
            if dbStats == nil {
                self.error = error.localizedDescription
            } else {
                hasStaleData = true
            }
        }
    }

    func loadWorkflowStats() async {
        do {
            workflowStats = try await api.get("/api/stats/workflows")
        } catch {
            workflowStats = nil
        }
    }

    func runPipeline(stage: PipelineStage = .full) async {
        Haptics.impact()
        isRunning = true
        error = nil
        lastMessage = stage == .full
            ? "Starting smart sync..."
            : "Starting \(stage.displayName.lowercased())..."
        do {
            let response = try await startPipeline(for: stage)
            activeTraceRunId = response.runId ?? activeTraceRunId
            lastMessage = pipelineResponseMessage(response, fallbackStage: stage)
            // loadAll → loadStatus → updateRunningState will start monitoring
            await loadAll()
        } catch {
            if Task.isCancelled { return }
            if stage == .backfill, handleUnsupportedBackfill(error) {
                return
            }
            self.error = error.localizedDescription
            isRunning = false
            stopMonitoring()
        }
    }

    func resetStuck() async {
        do {
            let response: SuccessResponse = try await api.post("/api/sync/reset-stuck")
            lastMessage = response.message
            await loadAll()
        } catch {
            self.error = error.localizedDescription
        }
    }

    func submitWorkflows(daysBack: Int = 7, limit: Int = 3, templateId: String? = nil, model: String = "openai") async {
        do {
            let body = WorkflowSubmitRequest(
                daysBack: daysBack,
                limit: limit,
                templateId: templateId,
                model: model
            )
            let response: SuccessResponse = try await api.post(
                "/api/sync/workflows/submit",
                body: body
            )
            lastMessage = response.message
        } catch {
            self.error = error.localizedDescription
        }
    }

    func refreshWorkflows() async {
        do {
            let body = WorkflowRefreshRequest()
            let response: SuccessResponse = try await api.post(
                "/api/sync/workflows/refresh",
                body: body
            )
            lastMessage = response.message
        } catch {
            self.error = error.localizedDescription
        }
    }

    func loadUploadCandidates() async {
        do {
            let candidates: [UploadCandidate] = try await api.get("/api/sync/upload-candidates")
            uploadCandidates = candidates
        } catch {
            uploadCandidates = []
        }
    }

    func loadSyncFailures() async {
        guard supportsSyncFailuresEndpoint else {
            syncFailures = nil
            return
        }
        do {
            syncFailures = try await api.get("/api/sync/failures")
        } catch {
            if isNotFound(error) {
                supportsSyncFailuresEndpoint = false
            }
            syncFailures = nil
        }
    }

    func loadBackups() async {
        guard supportsAdminEndpoints else {
            backups = []
            return
        }
        do {
            backups = try await api.get("/api/admin/backups")
        } catch {
            if isNotFound(error) {
                supportsAdminEndpoints = false
            }
            backups = []
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

    func refreshCache() async {
        do {
            let response: SuccessResponse = try await api.post("/api/sync/refresh-cache")
            lastMessage = response.message
        } catch {
            self.error = error.localizedDescription
        }
    }

    func uploadAllCandidates(templateId: String? = nil, model: String = "openai") async {
        guard supportsUploadProcessEndpoint else {
            lastMessage = "Upload and process requires the newer backend API"
            return
        }

        isUploadingCandidates = true
        defer { isUploadingCandidates = false }

        do {
            let body = UploadProcessRequest(filePaths: nil, templateId: templateId, model: model)
            let response: UploadProcessResult = try await api.post("/api/sync/upload/process", body: body)
            lastMessage = "Uploaded \(response.uploadedCount) file(s), \(response.errorCount) error(s)"
            await loadUploadCandidates()
            await loadWorkflowStats()
        } catch {
            if isNotFound(error) {
                supportsUploadProcessEndpoint = false
                lastMessage = "Upload and process requires the newer backend API"
                return
            }
            self.error = error.localizedDescription
        }
    }

    func runStackAction(_ action: String) async {
        guard supportsAdminEndpoints else {
            lastMessage = "Admin controls require the newer backend API"
            return
        }

        isRunningStackAction = true
        defer { isRunningStackAction = false }
        error = nil

        do {
            let response: StackControlResponse = try await api.post("/api/admin/stack/\(action)")
            stackControl = response
            if response.status.lowercased() == "ok" {
                lastMessage = response.message
            } else {
                lastMessage = nil
                error = response.message.isEmpty
                    ? "Stack action \(response.action) reported \(response.status)."
                    : response.message
            }

            if action != "restart-public" {
                await loadSystemStatus()
            }

            if action != "restart-public" {
                await loadBackups()
                await loadUploadCandidates()
            }
        } catch {
            if isNotFound(error) {
                supportsAdminEndpoints = false
                lastMessage = "Admin controls require the newer backend API"
                return
            }
            self.error = error.localizedDescription
        }
    }

    func createBackup() async {
        guard supportsAdminEndpoints else {
            lastMessage = "Backups require the newer backend API"
            return
        }

        isCreatingBackup = true
        defer { isCreatingBackup = false }

        do {
            let backup: AdminBackupInfo = try await api.post("/api/admin/backups")
            backups.removeAll { $0.filename == backup.filename }
            backups.insert(backup, at: 0)
            lastMessage = backup.message.isEmpty ? "Backup created" : backup.message
        } catch {
            if isNotFound(error) {
                supportsAdminEndpoints = false
                lastMessage = "Backups require the newer backend API"
                return
            }
            self.error = error.localizedDescription
        }
    }

    func downloadBackup(_ backup: AdminBackupInfo) async {
        guard supportsAdminEndpoints else {
            lastMessage = "Backups require the newer backend API"
            return
        }

        do {
            downloadedBackupURL = try await api.downloadFile(backup.downloadPath)
            lastMessage = "Downloaded \(backup.filename)"
        } catch {
            if isNotFound(error) {
                supportsAdminEndpoints = false
                lastMessage = "Backups require the newer backend API"
                return
            }
            self.error = error.localizedDescription
        }
    }

    func loadAll() async {
        isLoading = true
        error = nil
        await loadStatus(showLoading: false)
        await loadDbStats()
        await loadWorkflowStats()
        await loadUploadCandidates()
        await loadSyncFailures()
        await loadBackups()
        await loadSystemStatus()
        // If all key data arrived successfully, clear any previous stale flag.
        if pipelineStatus != nil && dbStats != nil {
            hasStaleData = false
        }
        isLoading = false
    }

    func refresh() async {
        // Cancel any older in-flight refresh; only the newest pull-to-refresh wins.
        refreshTask?.cancel()
        let task = Task { [weak self] in
            guard let self else { return }
            await self.loadAll()
        }
        refreshTask = task
        await task.value
    }

    private func updateRunningState(from status: PipelineStatus) {
        let normalizedStatus = status.status.lowercased()
        if ["idle", "completed", "failed"].contains(normalizedStatus) {
            isRunning = false
            stopMonitoring()
            return
        }

        let running = status.running
            || normalizedStatus.contains("running")
            || normalizedStatus.contains("processing")
            || normalizedStatus.contains("queued")
            || status.activePhase?.status.lowercased() == "running"

        isRunning = running

        if running {
            startMonitoring()
        } else {
            stopMonitoring()
        }
    }

    private func isNotFound(_ error: Error) -> Bool {
        guard case APIError.httpError(let status, _) = error else {
            return false
        }
        return status == 404
    }

    private func startPipeline(for stage: PipelineStage) async throws -> PipelineRunResponse {
        var lastError: Error?

        for candidate in stage.requestStageCandidates {
            do {
                let body = PipelineRunRequest(stage: candidate)
                let response: PipelineRunResponse = try await api.post("/api/sync/run", body: body)
                if stage == .backfill {
                    supportsBackfillStage = true
                }
                return response
            } catch {
                lastError = error
                guard stage == .backfill,
                      isUnsupportedBackfill(error),
                      candidate != stage.requestStageCandidates.last else {
                    throw error
                }
            }
        }

        throw lastError ?? APIError.invalidResponse
    }

    private func pipelineResponseMessage(_ response: PipelineRunResponse, fallbackStage: PipelineStage) -> String {
        let fragments = [response.message, response.detail, response.warning] + (response.warnings ?? [])
        var unique: [String] = []
        var seen = Set<String>()

        for fragment in fragments {
            guard let fragment,
                  !fragment.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  seen.insert(fragment).inserted else {
                continue
            }
            unique.append(fragment)
        }

        if unique.isEmpty {
            return "Started \(fallbackStage.displayName.lowercased())."
        }

        return unique.joined(separator: " ")
    }

    private func isUnsupportedBackfill(_ error: Error) -> Bool {
        Self.backfillFailureDetails(from: error)?.indicatesUnsupportedStage == true
    }

    private func handleUnsupportedBackfill(_ error: Error) -> Bool {
        guard let details = Self.backfillFailureDetails(from: error),
              details.indicatesUnsupportedStage else {
            return false
        }

        supportsBackfillStage = false
        lastMessage = nil
        self.error = "Full History is not supported by this backend yet (HTTP \(details.statusCode)). \(details.summary)"
        isRunning = false
        stopMonitoring()
        return true
    }

    private static func summarizeBackfillErrorBody(_ body: String) -> String {
        let collapsed = collapseWhitespace(body)
        guard !collapsed.isEmpty else {
            return "Empty response body."
        }

        guard let data = collapsed.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) else {
            return String(collapsed.prefix(220))
        }

        if let summary = summarizeBackfillJSON(object) {
            return String(collapseWhitespace(summary).prefix(220))
        }

        return String(collapsed.prefix(220))
    }

    private static func summarizeBackfillJSON(_ object: Any) -> String? {
        if let string = object as? String {
            let collapsed = collapseWhitespace(string)
            return collapsed.isEmpty ? nil : collapsed
        }

        if let dictionary = object as? [String: Any] {
            if let detail = dictionary["detail"],
               let summary = summarizeBackfillJSON(detail) {
                return summary
            }

            for key in ["message", "error", "warning"] {
                if let value = dictionary[key] as? String {
                    let collapsed = collapseWhitespace(value)
                    if !collapsed.isEmpty {
                        return collapsed
                    }
                }
            }
        }

        if let array = object as? [Any] {
            let summaries = array.compactMap { entry -> String? in
                if let validation = entry as? [String: Any],
                   let summary = summarizeBackfillValidation(validation) {
                    return summary
                }

                return summarizeBackfillJSON(entry)
            }

            guard !summaries.isEmpty else {
                return nil
            }

            return summaries.prefix(2).joined(separator: "; ")
        }

        return nil
    }

    private static func summarizeBackfillValidation(_ entry: [String: Any]) -> String? {
        let location = (entry["loc"] as? [Any])?
            .compactMap { $0 as? String }
            .filter { $0 != "body" }
            .joined(separator: ".")
        let message = collapseWhitespace((entry["msg"] as? String) ?? (entry["message"] as? String) ?? "")
        let input = stringifyBackfillJSON(entry["input"])

        guard !message.isEmpty || (location?.isEmpty == false) else {
            return nil
        }

        var summary = ""
        if let location, !location.isEmpty {
            summary += "\(location): "
        }
        summary += message

        if let input, !input.isEmpty, input != "null" {
            summary += " (input: \(String(input.prefix(40))))"
        }

        return summary
    }

    private static func stringifyBackfillJSON(_ value: Any?) -> String? {
        guard let value else {
            return nil
        }

        if value is NSNull {
            return "null"
        }

        if let string = value as? String {
            let collapsed = collapseWhitespace(string)
            return collapsed.isEmpty ? nil : collapsed
        }

        if let number = value as? NSNumber {
            return number.stringValue
        }

        if JSONSerialization.isValidJSONObject(value),
           let data = try? JSONSerialization.data(withJSONObject: value, options: []),
           let string = String(data: data, encoding: .utf8) {
            let collapsed = collapseWhitespace(string)
            return collapsed.isEmpty ? nil : collapsed
        }

        return String(describing: value)
    }

    private static func collapseWhitespace(_ string: String) -> String {
        string
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    private func startMonitoring() {
        guard monitorTask == nil else { return }
        stablePollCount = 0
        lastMonitorSignature = monitorSignature

        monitorTask = Task { [weak self] in
            guard let self else { return }
            await self.monitorLoop()
        }
    }

    private func stopMonitoring() {
        monitorTask?.cancel()
        monitorTask = nil
        stablePollCount = 0
        lastMonitorSignature = ""
    }

    private var monitorSignature: String {
        let activePhase = pipelineStatus?.activePhase
        let runId = pipelineStatus?.runId ?? "-"
        let status = pipelineStatus?.status ?? "-"
        let currentPhase = pipelineStatus?.currentPhase ?? "-"
        let activePhaseStatus = activePhase?.status ?? "-"
        let completedItems = String(activePhase?.completedItems ?? -1)
        let currentStep = activePhase?.currentStep ?? "-"
        let currentItem = activePhase?.currentItem ?? "-"
        let pendingCount = String(dbStats?.pending ?? -1)
        let processingCount = String(dbStats?.processing ?? -1)
        let completedCount = String(dbStats?.completed ?? -1)
        let failedCount = String(dbStats?.failed ?? -1)

        let signatureParts = [
            runId,
            status,
            currentPhase,
            activePhaseStatus,
            completedItems,
            currentStep,
            currentItem,
            pendingCount,
            processingCount,
            completedCount,
            failedCount
        ]

        return signatureParts.joined(separator: "|")
    }

    private func nextMonitorDelayNanoseconds() -> UInt64 {
        switch stablePollCount {
        case 0:
            return 750_000_000
        case 1:
            return 1_500_000_000
        case 2:
            return 3_000_000_000
        default:
            return 5_000_000_000
        }
    }

    private func monitorLoop() async {
        defer { monitorTask = nil }

        // Brief delay on first iteration to avoid racing with the
        // loadAll() call that triggered monitoring in the first place.
        try? await Task.sleep(nanoseconds: 750_000_000)

        while !Task.isCancelled {
            await loadStatus(showLoading: false)

            if isRunning {
                await loadDbStats()
            } else {
                break
            }

            let signature = monitorSignature
            if signature == lastMonitorSignature {
                stablePollCount = min(stablePollCount + 1, 3)
            } else {
                stablePollCount = 0
                lastMonitorSignature = signature
            }

            if Task.isCancelled {
                break
            }

            try? await Task.sleep(nanoseconds: nextMonitorDelayNanoseconds())
        }
    }
}
