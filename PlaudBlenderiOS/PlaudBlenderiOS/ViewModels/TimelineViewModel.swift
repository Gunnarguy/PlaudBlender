import Foundation
import Observation

@Observable
final class TimelineViewModel {
    var days: [DaySummary] = []
    var selectedDate: String?
    var isLoading = false
    var isLoadingMore = false
    var hasMorePages = true
    var error: String?

    private let api: APIClient
    private let pageSize = 50
    private var nextRawOffset = 0

    private struct TimelinePage {
        let days: [DaySummary]
        let nextRawOffset: Int
        let hasMorePages: Bool
    }

    init(api: APIClient) {
        self.api = api
    }

    func loadDays(limit: Int = 50, offset: Int = 0) async {
        isLoading = true
        error = nil
        do {
            let page = try await fetchTimelinePage(limit: limit, startingOffset: offset)
            nextRawOffset = page.nextRawOffset
            if offset == 0 {
                days = page.days.sorted(by: DaySummary.sortMostRecentFirst)
            } else {
                appendUniqueDays(page.days)
            }
            hasMorePages = page.hasMorePages
        } catch {
            self.error = error.localizedDescription
        }
        isLoading = false
    }

    func loadMore() async {
        guard !isLoadingMore, !isLoading, hasMorePages else { return }
        isLoadingMore = true
        do {
            let page = try await fetchTimelinePage(limit: pageSize, startingOffset: nextRawOffset)
            nextRawOffset = page.nextRawOffset
            appendUniqueDays(page.days)
            hasMorePages = page.hasMorePages
        } catch {
            self.error = error.localizedDescription
        }
        isLoadingMore = false
    }

    func loadFilledDays(startDate: String, endDate: String) async {
        isLoading = true
        error = nil
        do {
            let query = ["start_date": startDate, "end_date": endDate]
            let response: DaysResponse = try await api.get("/api/timeline/days-filled", query: query)
            days = response.days
                .filter(\.shouldDisplayInTimeline)
                .sorted(by: DaySummary.sortMostRecentFirst)
            nextRawOffset = response.days.count
            hasMorePages = false
        } catch {
            self.error = error.localizedDescription
        }
        isLoading = false
    }

    func refresh() async {
        await loadDays()
    }

    private func appendUniqueDays(_ newDays: [DaySummary]) {
        guard !newDays.isEmpty else { return }
        let existingIds = Set(days.map(\.id))
        days.append(contentsOf: newDays.filter { !existingIds.contains($0.id) })
        days.sort(by: DaySummary.sortMostRecentFirst)
    }

    private func fetchTimelinePage(limit: Int, startingOffset: Int) async throws -> TimelinePage {
        var rawOffset = startingOffset
        var visibleDays: [DaySummary] = []

        while true {
            let query = ["limit": "\(limit)", "offset": "\(rawOffset)"]
            let response: DaysResponse = try await api.get("/api/timeline/days-filled", query: query)
            let batch = response.days

            rawOffset += batch.count
            visibleDays.append(contentsOf: batch.filter(\.shouldDisplayInTimeline))

            let reachedEnd = batch.count < limit
            if !visibleDays.isEmpty || reachedEnd || batch.isEmpty {
                return TimelinePage(
                    days: visibleDays,
                    nextRawOffset: rawOffset,
                    hasMorePages: !reachedEnd
                )
            }
        }
    }
}
