import Foundation
import Observation

@Observable
final class TopicsViewModel {
    var topics: [Topic] = []
    var selectedTimeline: TopicTimeline?
    var isLoading = false
    var error: String?

    private let api: APIClient

    init(api: APIClient) {
        self.api = api
    }

    func loadTopics(limit: Int = 50) async {
        isLoading = true
        error = nil
        do {
            let query = ["limit": "\(limit)"]
            topics = try await api.get("/api/topics", query: query)
        } catch {
            self.error = error.localizedDescription
        }
        isLoading = false
    }

    func loadTimeline(topic: String) async {
        do {
            let encoded = topic.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? topic
            selectedTimeline = try await api.get("/api/topics/\(encoded)")
        } catch {
            self.error = error.localizedDescription
        }
    }

    func refresh() async {
        await loadTopics()
    }
}
