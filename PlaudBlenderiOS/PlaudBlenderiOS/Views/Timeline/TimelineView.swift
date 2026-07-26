import SwiftUI

// MARK: - Timeline View

struct TimelineView: View {
    let viewModel: TimelineViewModel

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                rangeControls

                Group {
                    if viewModel.isLoading && viewModel.days.isEmpty {
                        LoadingView(message: "Loading timeline...")
                    } else if viewModel.days.isEmpty && viewModel.selectedRange != .year {
                        EmptyStateView(
                            icon: "calendar",
                            title: "Nothing in This Range",
                            message: "Try another time range or refresh your recordings.",
                            actionTitle: "Refresh",
                            action: { Task { await viewModel.refresh() } }
                        )
                    } else if viewModel.selectedRange == .year {
                        YearOverviewView(
                            days: viewModel.days,
                            year: Calendar.current.component(.year, from: viewModel.anchorDate),
                            errorMessage: viewModel.error,
                            onSelectMonth: { month in
                                Task {
                                    await viewModel.showMonth(
                                        year: Calendar.current.component(.year, from: viewModel.anchorDate),
                                        month: month
                                    )
                                }
                            }
                        )
                    } else {
                        daysList
                    }
                }
            }
            .navigationTitle("Timeline")
            .refreshable { await viewModel.refresh() }
            .task { await viewModel.bootstrapIfNeeded() }
            .onChange(of: viewModel.api.isServerReachable) { _, reachable in
                if reachable && viewModel.days.isEmpty {
                    Task { await viewModel.refresh() }
                }
            }
            .navigationDestination(for: String.self) { recordingId in
                RecordingDetailContainerView(recordingId: recordingId)
            }
        }
    }

    private var rangeControls: some View {
        VStack(spacing: 10) {
            HStack(spacing: 6) {
                ForEach(TimelineRange.allCases) { range in
                    Button {
                        Task { await viewModel.selectRange(range) }
                    } label: {
                        Text(range.title)
                            .font(.caption.weight(.semibold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 7)
                            .background(
                                viewModel.selectedRange == range
                                    ? Color.accentColor
                                    : Color.secondary.opacity(0.12)
                            )
                            .foregroundStyle(viewModel.selectedRange == range ? Color.white : Color.primary)
                            .clipShape(Capsule())
                    }
                    .buttonStyle(.plain)
                    .accessibilityAddTraits(viewModel.selectedRange == range ? .isSelected : [])
                }
            }

            if viewModel.selectedRange != .all {
                HStack {
                    Button {
                        Task { await viewModel.moveRange(by: -1) }
                    } label: {
                        Image(systemName: "chevron.left")
                            .frame(width: 36, height: 30)
                    }
                    .accessibilityLabel("Previous \(viewModel.selectedRange.title.lowercased())")

                    Spacer()
                    Text(viewModel.rangeTitle)
                        .font(.subheadline.weight(.semibold))
                    Spacer()

                    Button {
                        Task { await viewModel.moveRange(by: 1) }
                    } label: {
                        Image(systemName: "chevron.right")
                            .frame(width: 36, height: 30)
                    }
                    .accessibilityLabel("Next \(viewModel.selectedRange.title.lowercased())")
                }
                .foregroundStyle(.primary)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
        .background(.bar)
    }

    private var daysList: some View {
        ScrollView {
            LazyVStack(spacing: 12) {
                if let error = viewModel.error {
                    ErrorBanner(message: error)
                }

                ForEach(viewModel.days) { day in
                    DayCardView(day: day)
                        .onAppear {
                            if day.id == viewModel.days.last?.id {
                                Task { await viewModel.loadMore() }
                            }
                        }
                }

                if viewModel.isLoadingMore {
                    ProgressView()
                        .padding()
                }
            }
            .padding()
        }
    }
}

// MARK: - Year Overview

private struct YearOverviewView: View {
    let days: [DaySummary]
    let year: Int
    let errorMessage: String?
    let onSelectMonth: (Int) -> Void

    private let columns = [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10)]

    private var yearDays: [DaySummary] {
        days.filter { $0.timelineDateComponents?.year == year }
    }
    private var annualRecordings: Int { yearDays.reduce(0) { $0 + $1.recordingCount } }
    private var annualEvents: Int { yearDays.reduce(0) { $0 + $1.eventCount } }
    private var annualDuration: Double { yearDays.reduce(0) { $0 + $1.totalDurationSeconds } }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let errorMessage {
                    ErrorBanner(message: errorMessage)
                }

                HStack(spacing: 0) {
                    YearMetric(value: "\(annualRecordings)", label: "Recordings")
                    Divider().frame(height: 34)
                    YearMetric(value: "\(annualEvents)", label: "Events")
                    Divider().frame(height: 34)
                    YearMetric(value: TimeInterval(annualDuration).durationFormatted, label: "Recorded")
                }
                .padding(.vertical, 12)
                .background(.regularMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 14))

                Text("Tap a month to see its days")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                LazyVGrid(columns: columns, spacing: 10) {
                    ForEach(1...12, id: \.self) { month in
                        Button {
                            onSelectMonth(month)
                        } label: {
                            YearMonthCard(year: year, month: month, days: daysForMonth(month))
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .padding()
        }
    }

    private func daysForMonth(_ month: Int) -> [DaySummary] {
        yearDays.filter { $0.timelineDateComponents?.month == month }
    }
}

private struct YearMetric: View {
    let value: String
    let label: String

    var body: some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.headline)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

private struct YearMonthCard: View {
    let year: Int
    let month: Int
    let days: [DaySummary]

    private let heatmapColumns = Array(repeating: GridItem(.flexible(), spacing: 3), count: 7)
    private var recordingCount: Int { days.reduce(0) { $0 + $1.recordingCount } }
    private var eventCount: Int { days.reduce(0) { $0 + $1.eventCount } }
    private var maximumEvents: Int { max(1, days.map(\.eventCount).max() ?? 1) }

    private var monthName: String {
        Calendar.current.monthSymbols[month - 1]
    }

    private var calendarCells: [Int?] {
        let calendar = Calendar.current
        guard let first = calendar.date(from: DateComponents(year: year, month: month, day: 1)),
              let dayRange = calendar.range(of: .day, in: .month, for: first) else { return [] }
        let offset = calendar.component(.weekday, from: first) - 1
        return Array(repeating: nil, count: offset) + dayRange.map { Optional($0) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack {
                Text(monthName)
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.tertiary)
            }

            LazyVGrid(columns: heatmapColumns, spacing: 3) {
                ForEach(Array(calendarCells.enumerated()), id: \.offset) { _, dayNumber in
                    RoundedRectangle(cornerRadius: 2)
                        .fill(color(for: dayNumber))
                        .aspectRatio(1, contentMode: .fit)
                }
            }

            Text(recordingCount == 0
                 ? "No activity"
                 : "\(recordingCount) recording\(recordingCount == 1 ? "" : "s") · \(eventCount) events")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.75)
        }
        .padding(12)
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 13))
        .contentShape(RoundedRectangle(cornerRadius: 13))
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(monthName), \(recordingCount) recordings, \(eventCount) events")
    }

    private func color(for dayNumber: Int?) -> Color {
        guard let dayNumber else { return .clear }
        guard let day = days.first(where: { $0.timelineDateComponents?.day == dayNumber }), day.eventCount > 0 else {
            return Color.secondary.opacity(0.12)
        }
        let intensity = Double(day.eventCount) / Double(maximumEvents)
        return Color.accentColor.opacity(0.28 + (intensity * 0.72))
    }
}

private extension DaySummary {
    var timelineDateComponents: DateComponents? {
        let pieces = date.split(separator: "-").compactMap { Int($0) }
        guard pieces.count == 3 else { return nil }
        return DateComponents(year: pieces[0], month: pieces[1], day: pieces[2])
    }
}

// MARK: - Day Card

struct DayCardView: View {
    let day: DaySummary
    @State private var isDayExpanded = false
    @State private var expandedRecordingId: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Header — tappable to expand/collapse the day
            Button {
                withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                    isDayExpanded.toggle()
                    if !isDayExpanded { expandedRecordingId = nil }
                }
            } label: {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(day.dateDisplay ?? day.date)
                            .font(.headline)
                        Text("\(day.recordingCount) recording\(day.recordingCount == 1 ? "" : "s") · \(day.eventCount) events")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    if let topCategory = day.topCategory {
                        CategoryPill(category: topCategory)
                    }
                    Image(systemName: isDayExpanded ? "chevron.up" : "chevron.down")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            // Collapsed preview — always visible
            if !isDayExpanded {
                // AI Summary snippet
                if let summary = day.aiSummary, !summary.isEmpty {
                    Text(summary.markdownPlainPreview)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }

                // Duration + category bar
                HStack(spacing: 4) {
                    Image(systemName: "clock")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text(TimeInterval(day.totalDurationSeconds).durationFormatted)
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Spacer()

                    if let pcts = day.categoryPercentages, !pcts.isEmpty {
                        HStack(spacing: 0) {
                            ForEach(pcts.sorted(by: { $0.value > $1.value }), id: \.key) { cat, pct in
                                Rectangle()
                                    .fill(Color.forCategory(cat))
                                    .frame(width: max(2, CGFloat(pct) * 100), height: 4)
                            }
                        }
                        .clipShape(Capsule())
                        .frame(width: 100)
                    }
                }
            }

            // Expanded content — full day details + recordings
            if isDayExpanded {
                // Full AI Summary
                if let summary = day.aiSummary, !summary.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Label("Day Summary", systemImage: "sparkles")
                            .font(.caption2)
                            .fontWeight(.semibold)
                            .foregroundStyle(.purple)
                        MarkdownText(summary, baseFont: .subheadline)
                            .foregroundStyle(.secondary)
                    }
                }

                // Duration + category bar (full width)
                HStack(spacing: 4) {
                    Image(systemName: "clock")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text(TimeInterval(day.totalDurationSeconds).durationFormatted)
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Spacer()

                    if let keywords = day.topKeywords, !keywords.isEmpty {
                        HStack(spacing: 4) {
                            ForEach(keywords.prefix(3), id: \.self) { keyword in
                                Text(keyword)
                                    .font(.system(size: 9))
                                    .padding(.horizontal, 5)
                                    .padding(.vertical, 2)
                                    .background(.ultraThinMaterial)
                                    .clipShape(Capsule())
                            }
                        }
                    }
                }

                // Category breakdown bar
                if let pcts = day.categoryPercentages, !pcts.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        GeometryReader { geo in
                            HStack(spacing: 0) {
                                ForEach(pcts.sorted(by: { $0.value > $1.value }), id: \.key) { cat, pct in
                                    Rectangle()
                                        .fill(Color.forCategory(cat))
                                        .frame(width: max(2, geo.size.width * pct))
                                }
                            }
                            .clipShape(Capsule())
                        }
                        .frame(height: 6)

                        HStack(spacing: 6) {
                            ForEach(pcts.sorted(by: { $0.value > $1.value }).prefix(5), id: \.key) { cat, pct in
                                HStack(spacing: 3) {
                                    Circle().fill(Color.forCategory(cat)).frame(width: 5, height: 5)
                                    Text("\(cat.capitalized) \(Int(pct * 100))%")
                                        .font(.system(size: 9))
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }

                Divider()

                // Recordings list
                if let recordings = day.recordings {
                    ForEach(recordings) { rec in
                        VStack(spacing: 0) {
                            Button {
                                withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                                    expandedRecordingId = expandedRecordingId == rec.recordingId ? nil : rec.recordingId
                                }
                            } label: {
                                RecordingCardView(
                                    recording: rec,
                                    isExpanded: expandedRecordingId == rec.recordingId
                                )
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)

                            if expandedRecordingId == rec.recordingId {
                                RecordingQuickDetailView(recording: rec)
                                    .transition(.opacity.combined(with: .move(edge: .top)))
                            }
                        }
                    }
                } else {
                    Text("No recordings for this day.")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.vertical, 8)
                }
            }
        }
        .padding()
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - Recording Card (mini)

struct RecordingCardView: View {
    let recording: RecordingSummary
    var isExpanded: Bool = false

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Circle()
                .fill(Color.forCategory(recording.topCategory))
                .frame(width: 8, height: 8)
                .padding(.top, 6)

            VStack(alignment: .leading, spacing: 6) {
                Text(recording.title ?? recording.timeRangeFormatted ?? "Recording")
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .lineLimit(1)

                HStack(spacing: 8) {
                    if let timeRange = recording.timeRangeFormatted {
                        Label(timeRange, systemImage: "clock")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    Label("\(recording.eventCount) events", systemImage: "list.bullet")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                RecordingSystemStrip(summary: recording)
            }

            Spacer(minLength: 8)

            Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .padding(.top, 4)
                .animation(.easeInOut(duration: 0.2), value: isExpanded)
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Recording Quick Detail (expanded inline preview)

struct RecordingQuickDetailView: View {
    let recording: RecordingSummary
    @Environment(APIClient.self) private var api
    @State private var detail: RecordingDetail?
    @State private var isLoading = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            RecordingSystemStrip(summary: detail?.summary ?? recording, workflowStatus: detail?.workflowStatus, detail: true)

            // Plaud AI Summary — prefer the full detail payload once loaded
            if let plaudSummary = resolvedPlaudAISummary, !plaudSummary.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Label("Plaud AI Summary", systemImage: "brain")
                        .font(.caption2)
                        .fontWeight(.semibold)
                        .foregroundStyle(.purple)
                    MarkdownText(plaudSummary, baseFont: .caption)
                        .foregroundStyle(.secondary)
                }
            }

            // Rich detail (fetched on expand)
            if isLoading {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Loading details…")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                .frame(maxWidth: .infinity, alignment: .center)
                .padding(.vertical, 4)
            } else if let detail {
                if let fullSummary = detail.aiSummary, !fullSummary.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Label("Chronos Summary", systemImage: "sparkles")
                            .font(.caption2)
                            .fontWeight(.semibold)
                            .foregroundStyle(.blue)
                        Text(fullSummary)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                // Category breakdown
                if let pcts = detail.categoryPercentages, !pcts.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        // Bar
                        GeometryReader { geo in
                            HStack(spacing: 0) {
                                ForEach(pcts.sorted(by: { $0.value > $1.value }), id: \.key) { cat, pct in
                                    Rectangle()
                                        .fill(Color.forCategory(cat))
                                        .frame(width: max(2, geo.size.width * pct))
                                }
                            }
                            .clipShape(Capsule())
                        }
                        .frame(height: 5)

                        // Legend
                        HStack(spacing: 6) {
                            ForEach(pcts.sorted(by: { $0.value > $1.value }).prefix(4), id: \.key) { cat, pct in
                                HStack(spacing: 3) {
                                    Circle().fill(Color.forCategory(cat)).frame(width: 5, height: 5)
                                    Text("\(cat.capitalized) \(Int(pct * 100))%")
                                        .font(.system(size: 9))
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }

                // Key moments — top 3 events by confidence
                if !detail.events.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Label("Key Moments", systemImage: "star.fill")
                            .font(.caption2)
                            .fontWeight(.semibold)
                            .foregroundStyle(.yellow)

                        let topEvents = Array(detail.events
                            .sorted { ($0.categoryConfidence ?? 0) > ($1.categoryConfidence ?? 0) }
                            .prefix(3))

                        ForEach(topEvents) { event in
                            HStack(alignment: .top, spacing: 6) {
                                Circle()
                                    .fill(Color.forCategory(event.category))
                                    .frame(width: 5, height: 5)
                                    .padding(.top, 4)
                                VStack(alignment: .leading, spacing: 1) {
                                    Text(event.cleanText)
                                        .font(.caption2)
                                        .lineLimit(2)
                                    HStack(spacing: 6) {
                                        CategoryPill(category: event.category)
                                        if let conf = event.categoryConfidence {
                                            ConfidenceBadge(confidence: conf)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Full local transcript
                if let transcript = detail.transcript, !transcript.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Label("Transcript", systemImage: "text.quote")
                            .font(.caption2)
                            .fontWeight(.semibold)
                            .foregroundStyle(.secondary)
                        Text(transcript)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }

                // Full Plaud AI transcript
                if let plaudTranscript = detail.plaudTranscript, !plaudTranscript.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Label("Plaud AI Transcript", systemImage: "waveform")
                            .font(.caption2)
                            .fontWeight(.semibold)
                            .foregroundStyle(.secondary)
                        Text(plaudTranscript)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            // Metadata row
            HStack(spacing: 10) {
                Label(
                    recording.durationFormatted ?? "\(recording.durationSeconds)s",
                    systemImage: "timer"
                )
                .font(.caption2)
                .foregroundStyle(.secondary)

                if recording.timeIsEstimated == true {
                    Label("Estimated", systemImage: "questionmark.circle")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                }

                Spacer()

                CategoryPill(category: recording.topCategory)
            }

            // View Full button
            NavigationLink(value: recording.recordingId) {
                HStack(spacing: 6) {
                    Text("View Full Detail")
                        .font(.caption.weight(.semibold))
                    Image(systemName: "arrow.right")
                        .font(.caption2)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 8)
                .background(.ultraThinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }
            .buttonStyle(.plain)
        }
        .padding(10)
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .padding(.top, 2)
        .task {
            await loadDetail()
        }
    }

    private func loadDetail() async {
        guard detail == nil, !isLoading else { return }
        isLoading = true
        do {
            detail = try await RecordingDetailViewModel.loadFully(api: api, recordingId: recording.recordingId)
        } catch {
            // Silently fail — the preview still shows summary-level data
        }
        isLoading = false
    }

    private var resolvedPlaudAISummary: String? {
        if let summary = detail?.summary.plaudAiSummary, !summary.isEmpty {
            return summary
        }
        if let summary = recording.plaudAiSummary, !summary.isEmpty {
            return summary
        }
        return nil
    }
}
