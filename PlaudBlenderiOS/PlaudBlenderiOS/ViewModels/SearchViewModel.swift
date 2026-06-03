import Foundation
import Observation

@Observable
final class SearchViewModel {
    var query = ""
    var results: [SearchResult] = []
    var aiAnswer: AIAnswer?
    var askConversation: [AskConversationTurn] = []
    var total: Int = 0
    var isSearching = false
    var isAskingAI = false
    var error: String?

    var selectedCategories: Set<String> = []
    var startDate: Date?
    var endDate: Date?
    var askSettings: AskChronosSettings {
        didSet {
            let normalized = askSettings.normalizedForModelCapabilities()
            if normalized != askSettings {
                askSettings = normalized
                return
            }
            persistAskSettings()
        }
    }

    private let api: APIClient
    private static let askSettingsDefaultsKey = "search.askChronosSettings"

    init(api: APIClient) {
        self.api = api
        self.askSettings = Self.loadAskSettings().normalizedForModelCapabilities()
    }

    var latestAskTurn: AskConversationTurn? {
        askConversation.last
    }

    var hasAskConversation: Bool {
        !askConversation.isEmpty
    }

    private var previousResponseId: String? {
        latestAskTurn?.answer.responseId
    }

    @discardableResult
    func search(query queryOverride: String? = nil, preserveAIAnswer: AIAnswer? = nil) async -> Bool {
        let activeQuery = (queryOverride ?? query).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !activeQuery.isEmpty else { return false }
        query = activeQuery
        if preserveAIAnswer == nil {
            Haptics.tap()
        }
        isSearching = true
        error = nil
        do {
            var request = SearchRequest(query: activeQuery)
            if !selectedCategories.isEmpty {
                request.categories = Array(selectedCategories)
            }
            if let startDate {
                request.startDate = startDate.iso8601String
            }
            if let endDate {
                request.endDate = endDate.iso8601String
            }
            let response: SearchResponse = try await api.post("/api/search", body: request)
            results = response.results
            aiAnswer = preserveAIAnswer ?? response.aiAnswer
            total = response.total
            isSearching = false
            return true
        } catch {
            self.error = error.localizedDescription
            if let preserveAIAnswer {
                aiAnswer = preserveAIAnswer
            }
        }
        isSearching = false
        return false
    }

    @discardableResult
    func askAI(question: String, loadSupportingResults: Bool = true) async -> Bool {
        let trimmedQuestion = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedQuestion.isEmpty else { return false }
        Haptics.impact()
        isAskingAI = true
        error = nil
        results = []
        total = 0
        do {
            let request = AskRequest(
                question: trimmedQuestion,
                settings: askSettings,
                previousResponseId: previousResponseId
            )
            let response: AIAnswer = try await api.post("/api/search/ask", body: request)
            let enrichedResponse = response.applyingFallback(settings: askSettings)
            aiAnswer = enrichedResponse
            query = trimmedQuestion
            var supportingMoments: Int?
            if loadSupportingResults {
                let didLoadSupportingResults = await search(
                    query: trimmedQuestion,
                    preserveAIAnswer: enrichedResponse
                )
                if didLoadSupportingResults {
                    supportingMoments = total
                }
            }
            askConversation.append(
                AskConversationTurn(
                    question: trimmedQuestion,
                    answer: enrichedResponse,
                    supportingMoments: supportingMoments
                )
            )
            isAskingAI = false
            return true
        } catch {
            self.error = error.localizedDescription
        }
        isAskingAI = false
        return false
    }

    func clear() {
        query = ""
        results = []
        aiAnswer = nil
        askConversation = []
        total = 0
        selectedCategories = []
        startDate = nil
        endDate = nil
    }

    func resetAskConversation() {
        askConversation = []
        aiAnswer = nil
        results = []
        total = 0
        error = nil
        query = ""
    }

    private static func loadAskSettings() -> AskChronosSettings {
        guard let data = UserDefaults.standard.data(forKey: askSettingsDefaultsKey),
              let decoded = try? JSONDecoder().decode(AskChronosSettings.self, from: data) else {
            return AskChronosSettings()
        }
        return decoded
    }

    private func persistAskSettings() {
        guard let data = try? JSONEncoder().encode(askSettings) else { return }
        UserDefaults.standard.set(data, forKey: Self.askSettingsDefaultsKey)
    }
}
