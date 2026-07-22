import Foundation
import Observation

enum TimelineRange: String, CaseIterable, Identifiable {
    case day
    case week
    case month
    case year
    case all

    var id: Self { self }
    var title: String { rawValue.capitalized }

    fileprivate var calendarComponent: Calendar.Component? {
        switch self {
        case .day: .day
        case .week: .weekOfYear
        case .month: .month
        case .year: .year
        case .all: nil
        }
    }
}

@Observable
final class TimelineViewModel {
    var days: [DaySummary] = []
    var selectedDate: String?
    var selectedRange: TimelineRange = .all
    var anchorDate = Date()
    var isLoading = false
    var isLoadingMore = false
    var hasMorePages = true
    var error: String?

    let api: APIClient
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

    var rangeTitle: String {
        let formatter = DateFormatter()
        formatter.locale = .current

        switch selectedRange {
        case .day:
            formatter.dateFormat = "EEEE, MMM d"
        case .week:
            guard let interval = Calendar.current.dateInterval(of: .weekOfYear, for: anchorDate) else {
                return "This Week"
            }
            let end = Calendar.current.date(byAdding: .day, value: -1, to: interval.end) ?? interval.end
            formatter.dateFormat = "MMM d"
            return "\(formatter.string(from: interval.start)) – \(formatter.string(from: end))"
        case .month:
            formatter.dateFormat = "MMMM yyyy"
        case .year:
            formatter.dateFormat = "yyyy"
        case .all:
            return "All Time"
        }
        return formatter.string(from: anchorDate)
    }

    func bootstrapIfNeeded() async {
        guard days.isEmpty, !isLoading else { return }
        await reloadSelectedRange()
    }

    func selectRange(_ range: TimelineRange) async {
        guard selectedRange != range else { return }
        selectedRange = range
        anchorDate = Date()
        days = []
        await reloadSelectedRange()
    }

    func moveRange(by amount: Int) async {
        guard let component = selectedRange.calendarComponent,
              let movedDate = Calendar.current.date(byAdding: component, value: amount, to: anchorDate) else {
            return
        }
        anchorDate = movedDate
        days = []
        await reloadSelectedRange()
    }

    func showMonth(year: Int, month: Int) async {
        guard let date = Calendar.current.date(from: DateComponents(year: year, month: month, day: 1)) else {
            return
        }
        selectedRange = .month
        anchorDate = date
        days = []
        await reloadSelectedRange()
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
        await reloadSelectedRange()
    }

    private func reloadSelectedRange() async {
        guard let component = selectedRange.calendarComponent else {
            nextRawOffset = 0
            hasMorePages = true
            await loadDays()
            return
        }

        let calendar = Calendar.current
        guard let interval = calendar.dateInterval(of: component, for: anchorDate) else { return }
        let inclusiveEnd = calendar.date(byAdding: .day, value: -1, to: interval.end) ?? interval.end
        let formatter = DateFormatter()
        formatter.calendar = calendar
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd"
        await loadFilledDays(
            startDate: formatter.string(from: interval.start),
            endDate: formatter.string(from: inclusiveEnd)
        )
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
