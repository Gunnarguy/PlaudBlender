import Foundation
import Observation

@MainActor
@Observable
final class StatsViewModel {
    var stats: Stats?
    var workflowStats: WorkflowStats?
    var sessionCost: SessionCost?
    var costHistory: CostHistory?
    var modelPricing: ModelPricing?
    var isLoading = false
    var hasBootstrapped = false
    var hasStaleData = false
    var error: String?

    private(set) var api: APIClient
    @ObservationIgnored private var refreshTask: Task<Void, Never>?

    init(api: APIClient) {
        self.api = api
    }

    func loadStats() async {
        do {
            stats = try await api.get("/api/stats", timeoutInterval: 30)
            hasStaleData = false
        } catch is CancellationError {
            return
        } catch let urlError as URLError where urlError.code == .cancelled {
            return
        } catch {
            if stats != nil {
                hasStaleData = true
            } else {
                self.error = error.localizedDescription
            }
        }
    }

    func loadWorkflowStats() async {
        do {
            workflowStats = try await api.get("/api/stats/workflows")
        } catch is CancellationError {
            return
        } catch let urlError as URLError where urlError.code == .cancelled {
            return
        } catch {
            if workflowStats == nil && self.error == nil {
                self.error = error.localizedDescription
            }
        }
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
            let query = ["days": "\(days)"]
            costHistory = try await api.get("/api/costs/history", query: query)
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

    func loadAll() async {
        isLoading = true
        error = nil

        async let statsLoad: Void = loadStats()
        async let workflowLoad: Void = loadWorkflowStats()
        async let sessionCostLoad: Void = loadSessionCost()
        async let costHistoryLoad: Void = loadCostHistory()
        async let pricingLoad: Void = loadPricing()

        _ = await (statsLoad, workflowLoad, sessionCostLoad, costHistoryLoad, pricingLoad)

        isLoading = false
    }

    func bootstrapIfNeeded() async {
        guard !hasBootstrapped else { return }
        hasBootstrapped = true
        await loadAll()
    }

    func refresh() async {
        refreshTask?.cancel()
        let task = Task { [weak self] in
            guard let self else { return }
            await self.loadAll()
        }
        refreshTask = task
        await task.value
    }
}
