import SwiftUI
import Charts

struct StatsView: View {
    let viewModel: StatsViewModel

    @State private var showPricing = false
    @State private var showCostQuickLook = false

    var body: some View {
        NavigationStack {
            Group {
                if viewModel.isLoading && viewModel.stats == nil {
                    LoadingView(message: "Loading stats...")
                } else if let stats = viewModel.stats {
                    statsContent(stats)
                } else {
                    statsUnavailableState
                }
            }
            .navigationTitle("Stats")
            .toolbar {
                ToolbarItem(placement: platformTrailingToolbarPlacement) {
                    Button {
                        showCostQuickLook = true
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "dollarsign.circle.fill")
                                .font(.caption.weight(.semibold))
                            Text(costToolbarLabel)
                                .font(.caption.weight(.semibold))
                                .monospacedDigit()
                        }
                        .foregroundStyle(costToolbarTint)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(costToolbarTint.opacity(0.12))
                        .clipShape(Capsule())
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Open AI spend quick look")
                }
            }
            .sheet(isPresented: $showCostQuickLook) {
                CostQuickLookSheet(
                    sessionCost: viewModel.sessionCost,
                    costHistory: viewModel.costHistory,
                    pricing: viewModel.modelPricing,
                    errorMessage: viewModel.error,
                    onRefresh: {
                        async let sessionLoad: Void = viewModel.loadSessionCost()
                        async let historyLoad: Void = viewModel.loadCostHistory()
                        async let pricingLoad: Void = viewModel.loadPricing()
                        _ = await (sessionLoad, historyLoad, pricingLoad)
                    }
                )
                    .presentationDetents([.medium, .large])
                    .presentationDragIndicator(.visible)
            }
            .refreshable { await viewModel.refresh() }
            .task { await viewModel.bootstrapIfNeeded() }
        }
    }

    private var costToolbarLabel: String {
        if let total = viewModel.sessionCost?.totalCostUsd {
            return formatCost(total, zeroAsFree: false)
        }
        if let total = viewModel.costHistory?.totalCostUsd {
            return formatCost(total, zeroAsFree: false)
        }
        return "Costs"
    }

    private var costToolbarTint: Color {
        let amount = viewModel.sessionCost?.totalCostUsd ?? viewModel.costHistory?.totalCostUsd ?? 0
        switch amount {
        case ..<0.01:
            return .secondary
        case ..<0.1:
            return .accentGreen
        case ..<1:
            return .accentOrange
        default:
            return .accentRed
        }
    }

    private var statsUnavailableState: some View {
        VStack(spacing: 16) {
            if let error = viewModel.error {
                ErrorBanner(message: error)
                    .padding(.horizontal)
            }

            EmptyStateView(
                icon: "chart.bar",
                title: "Stats Are Taking Too Long",
                actionTitle: "Refresh",
                action: { Task { await viewModel.refresh() } }
            )
        }
    }

    private func statsContent(_ stats: Stats) -> some View {
        let hourly = stats.effectiveEventsByHour

        return ScrollView {
            VStack(spacing: 16) {
                if viewModel.hasStaleData {
                    HStack(spacing: 8) {
                        Image(systemName: "clock.arrow.circlepath")
                            .foregroundStyle(.orange)
                        Text("Showing last known stats while the server catches up.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                    }
                    .padding(10)
                    .background(.orange.opacity(0.08))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .padding(.horizontal)
                }

                if let error = viewModel.error {
                    ErrorBanner(message: error)
                        .padding(.horizontal)
                }

                overviewSection(stats)
                densitySection(stats)
                insightRadarSection(stats)

                if !stats.recentDayStats.isEmpty {
                    recentDaysSection(stats.recentDayStats)
                }

                if let workflowStats = viewModel.workflowStats {
                    workflowSection(workflowStats)
                }

                if let dist = stats.sentimentDistribution, !dist.isEmpty {
                    sentimentDistributionSection(dist)
                }

                if !stats.categories.isEmpty {
                    categoriesSection(stats)
                }

                if stats.categoriesByHour != nil {
                    heatmapSection(stats)
                }

                if !hourly.isEmpty {
                    activityByHourSection(hourly)
                }

                if let dow = stats.eventsByDayOfWeek, !dow.isEmpty {
                    dayOfWeekSection(dow)
                }

                if let keywords = stats.topKeywords, !keywords.isEmpty {
                    keywordsSection(keywords)
                }

                if let cloud = stats.plaudCloudStats, !cloud.isEmpty {
                    plaudCloudSection(cloud)
                }

                if viewModel.sessionCost != nil || viewModel.costHistory != nil {
                    costOverviewSection
                }

                if let history = viewModel.costHistory, let byDay = history.byDay, !byDay.isEmpty {
                    costHistorySection(byDay, totalCost: history.totalCostUsd, totalCalls: history.totalCalls, days: history.days)
                }

                if let cost = viewModel.sessionCost {
                    modelCostSection(cost)
                }

                if let pricing = viewModel.modelPricing {
                    modelPricingSection(pricing)
                }

                topicsLink
            }
            .padding(.vertical)
        }
    }

    private func overviewSection(_ stats: Stats) -> some View {
        StatsSectionCard(
            title: "Overview",
            subtitle: "The broad shape of your recordings and events.",
            systemImage: "chart.bar.xaxis"
        ) {
            LazyVGrid(columns: statsGridColumns, spacing: 10) {
                MetricTile(title: "Recordings", value: "\(stats.totalRecordings)", footnote: densityFootnote(stats.averageRecordingsPerDay, suffix: "/day"), icon: "waveform", tint: .accentPrimary)
                MetricTile(title: "Events", value: "\(stats.totalEvents)", footnote: densityFootnote(stats.averageEventsPerDay, suffix: "/day"), icon: "list.bullet", tint: .accentGreen)
                MetricTile(title: "Tracked Days", value: "\(stats.totalDays)", footnote: "Active span", icon: "calendar", tint: .accentOrange)
                MetricTile(title: "Recorded Hours", value: String(format: "%.1f", stats.totalDurationHours), footnote: densityFootnote(stats.averageHoursPerDay, suffix: "h/day"), icon: "clock", tint: .accentPurple)
                MetricTile(title: "Avg Events/Rec", value: String(format: "%.1f", stats.effectiveAvgEvents), footnote: "Per recording", icon: "number", tint: .accentCyan)
                MetricTile(title: "Avg Duration", value: String(format: "%.0fm", stats.effectiveAvgDuration), footnote: "Per recording", icon: "timer", tint: .accentPink)
            }
        }
    }

    private func densitySection(_ stats: Stats) -> some View {
        StatsSectionCard(
            title: "Density And Coverage",
            subtitle: "How concentrated the data is across categories, sentiment, and completion.",
            systemImage: "scope"
        ) {
            LazyVGrid(columns: statsGridColumns, spacing: 10) {
                if let dominant = stats.dominantCategory {
                    MetricTile(
                        title: "Dominant Category",
                        value: dominant.name.replacingOccurrences(of: "_", with: " ").capitalized,
                        footnote: "\(percentString(dominant.share)) of categorized events",
                        icon: "square.grid.2x2",
                        tint: Color.forCategory(dominant.name)
                    )
                }

                MetricTile(
                    title: "Categories",
                    value: "\(stats.categoryDiversity)",
                    footnote: "Distinct active categories",
                    icon: "square.stack.3d.up",
                    tint: .accentPrimary
                )

                if let sentiment = stats.sentimentAvg {
                    MetricTile(
                        title: "Avg Sentiment",
                        value: String(format: "%+.2f", sentiment),
                        footnote: sentiment > 0.2 ? "Mostly positive" : sentiment < -0.2 ? "Skews negative" : "Mostly neutral",
                        icon: sentiment > 0.2 ? "face.smiling" : sentiment < -0.2 ? "face.dashed.fill" : "face.dashed",
                        tint: sentiment > 0.2 ? .accentGreen : sentiment < -0.2 ? .accentRed : .accentOrange
                    )
                }

                if let rate = stats.pipelineCompletionRate {
                    MetricTile(
                        title: "Pipeline Completion",
                        value: percentString(rate),
                        footnote: rate >= 0.9 ? "Healthy" : "Needs cleanup",
                        icon: "checkmark.circle",
                        tint: rate >= 0.9 ? .accentGreen : .accentOrange
                    )
                }

                if let positiveShare = stats.positiveShare {
                    MetricTile(
                        title: "Positive Share",
                        value: percentString(positiveShare),
                        footnote: "Positive sentiment events",
                        icon: "sun.max",
                        tint: .accentGreen
                    )
                }

                if let negativeShare = stats.negativeShare {
                    MetricTile(
                        title: "Negative Share",
                        value: percentString(negativeShare),
                        footnote: "Negative sentiment events",
                        icon: "cloud.rain",
                        tint: .accentRed
                    )
                }
            }
        }
    }

    private func insightRadarSection(_ stats: Stats) -> some View {
        let radar = radarInsights(stats)
        guard !radar.isEmpty else { return AnyView(EmptyView()) }

        return AnyView(
            StatsSectionCard(
                title: "Pattern Radar",
                subtitle: "The most interesting high-signal takeaways from the current aggregate.",
                systemImage: "sparkles"
            ) {
                LazyVGrid(columns: statsGridColumns, spacing: 10) {
                    ForEach(radar) { insight in
                        InsightMetricTile(insight: insight)
                    }
                }
            }
        )
    }

    private func recentDaysSection(_ recentDays: [RecentDayStat]) -> some View {
        let recent = Array(recentDays.suffix(10))

        return StatsSectionCard(
            title: "Recent Days",
            subtitle: "A compact look at the latest day-level activity the API exposed.",
            systemImage: "calendar.badge.clock"
        ) {
            if !recent.isEmpty {
                Chart(recent) { day in
                    BarMark(
                        x: .value("Day", day.displayLabel),
                        y: .value("Events", day.events)
                    )
                    .foregroundStyle(Color.accentPrimary.gradient)
                }
                .chartYAxis(.hidden)
                .frame(height: 120)
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(recent.reversed()) { day in
                        VStack(alignment: .leading, spacing: 6) {
                            Text(day.displayLabel)
                                .font(.caption.weight(.semibold))
                            Text("\(day.events) events")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                            if let recordings = day.recordings {
                                Text("\(recordings) rec")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                            if let hours = day.durationHours {
                                Text(String(format: "%.1fh", hours))
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                            if let summary = day.summary, !summary.isEmpty {
                                Text(summary)
                                    .font(.caption2)
                                    .foregroundStyle(.tertiary)
                                    .lineLimit(2)
                            }
                        }
                        .frame(width: 110, alignment: .leading)
                        .padding(10)
                        .background(Color.accentPrimary.opacity(0.06))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    }
                }
            }
        }
    }

    private func workflowSection(_ workflow: WorkflowStats) -> some View {
        StatsSectionCard(
            title: "Plaud Workflow Coverage",
            subtitle: "Workflow backlog, completion, and AI summary coverage from the dedicated workflow endpoint.",
            systemImage: "brain"
        ) {
            LazyVGrid(columns: statsGridColumns, spacing: 10) {
                MetricTile(title: "Recent Recordings", value: "\(workflow.recentRecordings ?? 0)", footnote: "Workflow window", icon: "waveform.path.ecg", tint: .accentPrimary)
                MetricTile(title: "AI Summaries", value: "\(workflow.withAiSummary ?? 0)", footnote: "Already enriched", icon: "sparkles.rectangle.stack", tint: .accentGreen)
                MetricTile(title: "Ready", value: "\(workflow.readyForEnrichment ?? 0)", footnote: "Ready for enrichment", icon: "wand.and.stars", tint: .accentOrange)
                MetricTile(title: "Pending", value: "\(workflow.workflowPending ?? 0)", footnote: "Queued or running", icon: "hourglass", tint: .accentPurple)
                MetricTile(title: "Failed", value: "\(workflow.workflowFailed ?? 0)", footnote: "Need retry", icon: "exclamationmark.triangle", tint: .accentRed)
                MetricTile(title: "Succeeded", value: "\(workflow.workflowSuccess ?? 0)", footnote: workflow.lastSubmittedAt ?? "Completed workflows", icon: "checkmark.seal", tint: .accentGreen)
            }
        }
    }

    private func sentimentDistributionSection(_ dist: [String: Int]) -> some View {
        let segments = [
            SentimentSegment(label: "positive", count: dist["positive"] ?? 0, color: .accentGreen),
            SentimentSegment(label: "neutral", count: dist["neutral"] ?? 0, color: .secondary),
            SentimentSegment(label: "negative", count: dist["negative"] ?? 0, color: .accentRed)
        ]
        let total = max(segments.map(\.count).reduce(0, +), 1)

        return StatsSectionCard(
            title: "Sentiment Split",
            subtitle: "A quick breakdown of positive, neutral, and negative event tone.",
            systemImage: "face.smiling.inverse"
        ) {
            GeometryReader { geo in
                HStack(spacing: 0) {
                    ForEach(segments) { segment in
                        if segment.count > 0 {
                            Rectangle()
                                .fill(segment.color)
                                .frame(width: geo.size.width * CGFloat(segment.count) / CGFloat(total))
                        }
                    }
                }
                .clipShape(Capsule())
            }
            .frame(height: 20)

            LazyVGrid(columns: statsGridColumns, spacing: 10) {
                ForEach(segments) { segment in
                    MetricTile(
                        title: segment.label.capitalized,
                        value: "\(segment.count)",
                        footnote: percentString(Double(segment.count) / Double(total)),
                        icon: "circle.fill",
                        tint: segment.color
                    )
                }
            }
        }
    }

    private func categoriesSection(_ stats: Stats) -> some View {
        let topCategories = stats.categories.sorted { $0.value > $1.value }.prefix(8)
        let total = max(stats.categories.values.reduce(0, +), 1)

        return StatsSectionCard(
            title: "Categories",
            subtitle: "Where your event volume is concentrated.",
            systemImage: "square.grid.2x2"
        ) {
            categoryChart(stats.categories)

            VStack(spacing: 10) {
                ForEach(Array(topCategories), id: \.key) { category in
                    CategoryShareRow(
                        name: category.key,
                        count: category.value,
                        share: Double(category.value) / Double(total),
                        color: Color.forCategory(category.key)
                    )
                }
            }
        }
    }

    private func heatmapSection(_ stats: Stats) -> some View {
        let data = stats.heatmapData
        guard !data.isEmpty else { return AnyView(EmptyView()) }

        var categoryTotals: [String: Int] = [:]
        for point in data {
            categoryTotals[point.category, default: 0] += point.count
        }
        let categories = categoryTotals.sorted { $0.value > $1.value }.prefix(6).map(\.key)
        let filtered = data.filter { categories.contains($0.category) }
        let maxCount = max(filtered.map(\.count).max() ?? 1, 1)
        let hourTotals = Dictionary(grouping: filtered, by: \.hour).mapValues { $0.map(\.count).reduce(0, +) }
        let busiestHour = hourTotals.max(by: { $0.value < $1.value })

        return AnyView(
            StatsSectionCard(
                title: "Activity Heat Map",
                subtitle: busiestHour.map { "Hour \($0.key) is the densest block with \($0.value) events." } ?? "Hour of day by category.",
                systemImage: "square.grid.3x3.fill"
            ) {
                HStack(spacing: 10) {
                    Text("Quiet")
                        .font(.caption2)
                        .foregroundStyle(.secondary)

                    HStack(spacing: 4) {
                        ForEach(0..<4, id: \.self) { index in
                            RoundedRectangle(cornerRadius: 3)
                                .fill(Color.accentPrimary.opacity(0.15 + Double(index) * 0.2))
                                .frame(width: 14, height: 14)
                        }
                    }

                    Text("Dense")
                        .font(.caption2)
                        .foregroundStyle(.secondary)

                    Spacer()

                    Text("Top 6 categories")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                ScrollView(.horizontal, showsIndicators: false) {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 4) {
                            Text("")
                                .frame(width: 84, alignment: .leading)
                            ForEach(0..<24, id: \.self) { hour in
                                Text(hourLabelShort(hour))
                                    .font(.system(size: 8, weight: .medium))
                                    .foregroundStyle(hour % 3 == 0 ? .secondary : .tertiary)
                                    .frame(width: 16)
                            }
                            Text("Sum")
                                .font(.caption2.weight(.medium))
                                .foregroundStyle(.secondary)
                                .frame(width: 34, alignment: .trailing)
                        }

                        ForEach(categories, id: \.self) { category in
                            let total = categoryTotals[category] ?? 0
                            HStack(spacing: 4) {
                                Text(category.replacingOccurrences(of: "_", with: " ").capitalized)
                                    .font(.caption2)
                                    .lineLimit(1)
                                    .frame(width: 84, alignment: .leading)

                                ForEach(0..<24, id: \.self) { hour in
                                    let count = filtered.first(where: { $0.hour == hour && $0.category == category })?.count ?? 0
                                    let intensity = Double(count) / Double(maxCount)

                                    RoundedRectangle(cornerRadius: 3)
                                        .fill(Color.forCategory(category).opacity(count == 0 ? 0.05 : 0.18 + intensity * 0.82))
                                        .overlay {
                                            if count > 0 && intensity > 0.65 {
                                                Text("\(count)")
                                                    .font(.system(size: 6, weight: .bold))
                                                    .foregroundStyle(.white)
                                            }
                                        }
                                        .frame(width: 16, height: 16)
                                }

                                Text("\(total)")
                                    .font(.caption2.monospacedDigit())
                                    .foregroundStyle(.secondary)
                                    .frame(width: 34, alignment: .trailing)
                            }
                        }

                        Divider()

                        HStack(spacing: 4) {
                            Text("Total")
                                .font(.caption2.weight(.semibold))
                                .frame(width: 84, alignment: .leading)

                            ForEach(0..<24, id: \.self) { hour in
                                let count = hourTotals[hour] ?? 0
                                let intensity = Double(count) / Double(maxCount)
                                RoundedRectangle(cornerRadius: 3)
                                    .fill(Color.accentPrimary.opacity(count == 0 ? 0.05 : 0.18 + intensity * 0.82))
                                    .frame(width: 16, height: 16)
                            }

                            Text("\(hourTotals.values.reduce(0, +))")
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.secondary)
                                .frame(width: 34, alignment: .trailing)
                        }
                    }
                }
            }
        )
    }

    private func activityByHourSection(_ hourly: [Int: Int]) -> some View {
        let maxVal = hourly.values.max() ?? 1
        let entries = (0..<24).map { hour in
            HourEntry(hour: hour, count: hourly[hour] ?? 0)
        }

        return StatsSectionCard(
            title: "Activity By Hour",
            subtitle: "When events cluster across the day.",
            systemImage: "clock.badge"
        ) {
            Chart(entries) { entry in
                BarMark(
                    x: .value("Hour", entry.hour),
                    y: .value("Events", entry.count)
                )
                .foregroundStyle(Color.accentPrimary.opacity(entry.count > 0 ? 0.3 + 0.7 * Double(entry.count) / Double(maxVal) : 0.08))
            }
            .chartXAxis {
                AxisMarks(values: .stride(by: 3)) { value in
                    AxisValueLabel {
                        if let hour = value.as(Int.self) {
                            Text(hourLabelShort(hour))
                                .font(.caption2)
                        }
                    }
                }
            }
            .chartYAxis {
                AxisMarks { value in
                    AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5))
                    AxisValueLabel {
                        if let count = value.as(Int.self), count > 0 {
                            Text("\(count)")
                                .font(.caption2)
                        }
                    }
                }
            }
            .frame(height: 160)
        }
    }

    private func dayOfWeekSection(_ dow: [String: Int]) -> some View {
        let order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        let short = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        let entries = order.enumerated().map { index, day in
            DayEntry(
                label: short[index],
                count: dow[day] ?? dow[day.lowercased()] ?? dow[short[index]] ?? dow[short[index].lowercased()] ?? 0,
                index: index
            )
        }

        return StatsSectionCard(
            title: "Activity By Day",
            subtitle: "Weekly rhythm across the event set.",
            systemImage: "calendar.day.timeline.leading"
        ) {
            Chart(entries) { entry in
                BarMark(
                    x: .value("Day", entry.label),
                    y: .value("Events", entry.count)
                )
                .foregroundStyle(Color.accentPurple.opacity(0.75))
                .annotation(position: .top) {
                    if entry.count > 0 {
                        Text("\(entry.count)")
                            .font(.system(size: 9))
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .chartYAxis(.hidden)
            .frame(height: 150)
        }
    }

    private func keywordsSection(_ keywords: [[String: JSONValue]]) -> some View {
        let parsed: [(word: String, count: Int)] = keywords.compactMap { dict in
            guard let word = dict["keyword"]?.stringValue,
                  let count = dict["count"]?.intValue else { return nil }
            return (word, count)
        }
        .prefix(24)
        .map { $0 }

        let maxCount = max(parsed.map(\.count).max() ?? 1, 1)

        return StatsSectionCard(
            title: "Top Keywords",
            subtitle: "Repeated terms that cut across your recordings.",
            systemImage: "textformat.abc"
        ) {
            FlowLayout(spacing: 6) {
                ForEach(parsed, id: \.word) { item in
                    let scale = 0.75 + 0.55 * Double(item.count) / Double(maxCount)
                    VStack(spacing: 2) {
                        Text(item.word)
                            .font(.system(size: 13 * scale, weight: scale > 1.0 ? .semibold : .regular))
                        Text("\(item.count)")
                            .font(.system(size: 9))
                            .foregroundStyle(.secondary)
                    }
                    .padding(.horizontal, 9)
                    .padding(.vertical, 6)
                    .background(Color.accentPrimary.opacity(0.08 + 0.14 * Double(item.count) / Double(maxCount)))
                    .clipShape(Capsule())
                    .foregroundStyle(Color.accentPrimary)
                }
            }
        }
    }

    private func plaudCloudSection(_ cloud: [String: JSONValue]) -> some View {
        let count = cloud["total_count"]?.intValue ?? 0
        let hours = cloud["total_duration_hours"]?.doubleValue ?? 0
        let avgMin = cloud["avg_duration_minutes"]?.doubleValue ?? 0
        let earliest = cloud["date_range"]?.objectValue?["earliest"]?.stringValue
        let latest = cloud["date_range"]?.objectValue?["latest"]?.stringValue

        return StatsSectionCard(
            title: "Plaud Cloud Coverage",
            subtitle: "What the cloud-side recording set looks like.",
            systemImage: "cloud.fill"
        ) {
            LazyVGrid(columns: statsGridColumns, spacing: 10) {
                MetricTile(title: "Cloud Recordings", value: "\(count)", footnote: "Remote library", icon: "cloud", tint: .accentPrimary)
                MetricTile(title: "Cloud Hours", value: String(format: "%.1f", hours), footnote: "Total duration", icon: "clock.arrow.circlepath", tint: .accentCyan)
                MetricTile(title: "Avg Cloud Duration", value: String(format: "%.0fm", avgMin), footnote: "Per cloud recording", icon: "timer", tint: .accentOrange)
                MetricTile(title: "Cloud Share", value: statsCloudShare(count), footnote: "Vs local recordings", icon: "externaldrive.badge.icloud", tint: .accentPurple)
            }

            if let earliest, let latest {
                Text("\(earliest) – \(latest)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var costOverviewSection: some View {
        StatsSectionCard(
            title: "Cost Overview",
            subtitle: "Costs, token volume, and daily burn compressed into cleaner summary tiles.",
            systemImage: "dollarsign.circle"
        ) {
            LazyVGrid(columns: statsGridColumns, spacing: 10) {
                if let cost = viewModel.sessionCost {
                    MetricTile(title: "Total Cost", value: String(format: "$%.4f", cost.totalCostUsd), footnote: "Current session", icon: "dollarsign.circle", tint: .accentGreen)
                    MetricTile(title: "API Calls", value: "\(cost.totalCalls)", footnote: String(format: "%.0fm session", cost.sessionMinutes), icon: "arrow.left.arrow.right.circle", tint: .accentPrimary)
                    MetricTile(title: "Input Tokens", value: formatTokens(cost.totalInputTokens), footnote: "Prompt volume", icon: "arrow.down.circle", tint: .accentOrange)
                    MetricTile(title: "Output Tokens", value: formatTokens(cost.totalOutputTokens), footnote: "Response volume", icon: "arrow.up.circle", tint: .accentPurple)
                }

                if let history = viewModel.costHistory {
                    let avg = history.days > 0 ? history.totalCostUsd / Double(history.days) : 0
                    MetricTile(title: "Avg Daily Cost", value: String(format: "$%.4f", avg), footnote: "\(history.days)-day window", icon: "chart.line.uptrend.xyaxis", tint: .accentCyan)
                    MetricTile(title: "Total Calls Window", value: "\(history.totalCalls)", footnote: "Cost history window", icon: "calendar.badge.clock", tint: .accentPink)
                }
            }
        }
    }

    private func costHistorySection(_ byDay: [[String: JSONValue]], totalCost: Double, totalCalls: Int, days: Int) -> some View {
        let entries = byDay.compactMap { dict -> CostDayEntry? in
            guard let dateStr = dict["date"]?.stringValue,
                  let cost = dict["cost"]?.doubleValue ?? dict["cost_usd"]?.doubleValue ?? dict["total_cost_usd"]?.doubleValue else {
                return nil
            }
            let calls = dict["calls"]?.intValue ?? dict["total_calls"]?.intValue ?? 0
            return CostDayEntry(index: 0, date: dateStr, cost: cost, calls: calls)
        }

        let indexed = entries.enumerated().map { index, entry in
            CostDayEntry(index: index, date: entry.date, cost: entry.cost, calls: entry.calls)
        }
        let avgCost = indexed.isEmpty ? 0 : indexed.map(\.cost).reduce(0, +) / Double(indexed.count)
        let peak = indexed.max(by: { $0.cost < $1.cost })
        let width = max(CGFloat(indexed.count) * 24, 320)
        let stride = max(indexed.count / 4, 1)

        return StatsSectionCard(
            title: "Cost History",
            subtitle: "A wider, calmer chart with the average line and the peak day called out.",
            systemImage: "chart.line.uptrend.xyaxis.circle"
        ) {
            LazyVGrid(columns: statsGridColumns, spacing: 10) {
                MetricTile(title: "Window Total", value: String(format: "$%.4f", totalCost), footnote: "\(days)-day total", icon: "sum", tint: .accentGreen)
                MetricTile(title: "Avg / Day", value: String(format: "$%.4f", avgCost), footnote: "Daily burn", icon: "function", tint: .accentPrimary)
                if let peak {
                    MetricTile(title: "Peak Day", value: String(format: "$%.4f", peak.cost), footnote: peak.displayLabel, icon: "arrow.up.right.circle", tint: .accentRed)
                }
                MetricTile(title: "Calls", value: "\(totalCalls)", footnote: "Window total", icon: "arrow.up.arrow.down.circle", tint: .accentPurple)
            }

            ScrollView(.horizontal, showsIndicators: false) {
                Chart(indexed) { entry in
                    BarMark(
                        x: .value("Index", entry.index),
                        y: .value("Cost", entry.cost)
                    )
                    .foregroundStyle(Color.accentPrimary.gradient)

                    RuleMark(y: .value("Average", avgCost))
                        .foregroundStyle(Color.accentOrange.opacity(0.8))
                        .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 4]))
                }
                .chartXAxis {
                    AxisMarks(values: indexed.compactMap { entry in
                        entry.index == 0 || entry.index == indexed.count - 1 || entry.index % stride == 0 ? entry.index : nil
                    }) { value in
                        AxisValueLabel {
                            if let index = value.as(Int.self), indexed.indices.contains(index) {
                                Text(indexed[index].displayLabel)
                                    .font(.caption2)
                            }
                        }
                    }
                }
                .chartYAxis {
                    AxisMarks(position: .leading) { value in
                        AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5))
                        AxisValueLabel {
                            if let raw = value.as(Double.self) {
                                Text(String(format: "$%.3f", raw))
                                    .font(.caption2)
                            }
                        }
                    }
                }
                .frame(width: width, height: 190)
            }

            if let peak {
                Text("Peak spend landed on \(peak.displayLabel) with \(peak.calls) calls.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func modelCostSection(_ cost: SessionCost) -> some View {
        let models: [(name: String, calls: Int, cost: Double)] = cost.byModel.compactMap { key, value in
            guard let object = value.objectValue,
                  let calls = object["calls"]?.intValue,
                  let modelCost = object["cost_usd"]?.doubleValue else {
                return nil
            }
            return (key, calls, modelCost)
        }
        .sorted { $0.cost > $1.cost }

        let maxCost = max(models.map(\.cost).max() ?? 1, 1)

        return StatsSectionCard(
            title: "Cost By Model",
            subtitle: "Which models are actually consuming budget.",
            systemImage: "cpu"
        ) {
            ForEach(models, id: \.name) { model in
                VStack(alignment: .leading, spacing: 6) {
                    HStack(alignment: .firstTextBaseline) {
                        Text(model.name)
                            .font(.caption.weight(.medium))
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        Text("\(model.calls) calls")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                        Text(String(format: "$%.4f", model.cost))
                            .font(.caption.weight(.medium))
                            .monospacedDigit()
                    }

                    GeometryReader { geo in
                        Capsule()
                            .fill(Color.secondary.opacity(0.12))
                            .overlay(alignment: .leading) {
                                Capsule()
                                    .fill(Color.accentPrimary.gradient)
                                    .frame(width: geo.size.width * CGFloat(model.cost / maxCost))
                            }
                    }
                    .frame(height: 8)
                }

                if model.name != models.last?.name {
                    Divider()
                }
            }
        }
    }

    private func modelPricingSection(_ pricing: ModelPricing) -> some View {
        StatsSectionCard(
            title: "Model Pricing",
            subtitle: "Reference pricing from the server-side cost tracker.",
            systemImage: "tag.circle"
        ) {
            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    showPricing.toggle()
                }
            } label: {
                HStack {
                    Text(showPricing ? "Hide Pricing Table" : "Show Pricing Table")
                        .font(.subheadline.weight(.semibold))
                    Spacer()
                    Image(systemName: showPricing ? "chevron.up" : "chevron.down")
                        .foregroundStyle(.secondary)
                }
            }
            .buttonStyle(.plain)

            if showPricing {
                ForEach(pricing.models.indices, id: \.self) { index in
                    let model = pricing.models[index]
                    let name = model["label"]?.stringValue ?? model["model"]?.stringValue ?? model["name"]?.stringValue ?? "Unknown"
                    let provider = model["provider"]?.stringValue
                    let inputPerMillion = model["input_per_mtok"]?.doubleValue
                    let outputPerMillion = model["output_per_mtok"]?.doubleValue
                    let inputLegacy = model["input_cost_per_1k"]?.doubleValue ?? model["input_cost"]?.doubleValue
                    let outputLegacy = model["output_cost_per_1k"]?.doubleValue ?? model["output_cost"]?.doubleValue
                    let tier = model["tier"]?.stringValue

                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text(name)
                                .font(.caption.weight(.medium))
                                .lineLimit(1)
                            Spacer()
                            HStack(spacing: 6) {
                                if let tier {
                                    Text(tier.capitalized)
                                        .font(.caption2)
                                        .foregroundStyle(tier == "free" ? Color.accentGreen : .secondary)
                                        .padding(.horizontal, 6)
                                        .padding(.vertical, 2)
                                        .background((tier == "free" ? Color.accentGreen : Color.secondary).opacity(0.1))
                                        .clipShape(Capsule())
                                }

                                if let provider {
                                    Text(provider)
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                        .padding(.horizontal, 6)
                                        .padding(.vertical, 2)
                                        .background(.secondary.opacity(0.1))
                                        .clipShape(Capsule())
                                }
                            }
                        }

                        HStack(spacing: 16) {
                            if let inputPerMillion {
                                Text("In: $\(String(format: "%.2f", inputPerMillion))/1M")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .monospacedDigit()
                            } else if let inputLegacy {
                                Text("In: $\(String(format: "%.4f", inputLegacy))/1K")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .monospacedDigit()
                            }

                            if let outputPerMillion {
                                Text("Out: $\(String(format: "%.2f", outputPerMillion))/1M")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .monospacedDigit()
                            } else if let outputLegacy {
                                Text("Out: $\(String(format: "%.4f", outputLegacy))/1K")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .monospacedDigit()
                            }
                        }
                    }
                    .padding(.vertical, 4)

                    if index < pricing.models.count - 1 {
                        Divider()
                    }
                }
            }
        }
    }

    private var topicsLink: some View {
        NavigationLink {
            TopicsGridView(viewModel: TopicsViewModel(api: viewModel.api))
        } label: {
            HStack(spacing: 12) {
                Image(systemName: "tag")
                    .font(.title3)
                    .frame(width: 32)
                    .foregroundStyle(Color.accentPrimary)

                VStack(alignment: .leading, spacing: 4) {
                    Text("Topics")
                        .font(.headline)
                        .foregroundStyle(.primary)
                    Text("Clusters and recurring themes across your recordings.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Image(systemName: "chevron.right")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            .padding()
            .background(.regularMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
        .buttonStyle(.plain)
        .padding(.horizontal)
    }

    private func categoryChart(_ categories: [String: Int]) -> some View {
        let sorted = categories.sorted { $0.value > $1.value }

        return Chart {
            ForEach(sorted, id: \.key) { category, count in
                BarMark(
                    x: .value("Count", count),
                    y: .value("Category", category.replacingOccurrences(of: "_", with: " ").capitalized)
                )
                .foregroundStyle(Color.forCategory(category))
                .annotation(position: .trailing) {
                    Text("\(count)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .chartXAxis(.hidden)
        .frame(height: CGFloat(sorted.count * 34))
    }

    private func radarInsights(_ stats: Stats) -> [RadarInsight] {
        var insights: [RadarInsight] = []

        if let bestDay = stats.mostProductiveDay {
            insights.append(RadarInsight(title: "Best Day", value: bestDay, footnote: "Highest aggregate day", icon: "star.fill", tint: .accentYellow))
        }

        if let peakHour = stats.effectivePeakHour {
            insights.append(RadarInsight(title: "Peak Hour", value: hourLabelLong(peakHour), footnote: "Most active hour block", icon: "bolt.fill", tint: .accentOrange))
        }

        if let longest = stats.longestRecordingMin {
            insights.append(RadarInsight(title: "Longest Recording", value: String(format: "%.0fm", longest), footnote: "Single longest capture", icon: "record.circle", tint: .accentRed))
        }

        if let dominant = stats.dominantCategory {
            insights.append(RadarInsight(title: "Dominant Category", value: dominant.name.replacingOccurrences(of: "_", with: " ").capitalized, footnote: percentString(dominant.share), icon: "square.stack", tint: Color.forCategory(dominant.name)))
        }

        insights.append(RadarInsight(title: "Rec / Day", value: String(format: "%.1f", stats.averageRecordingsPerDay), footnote: "Average recordings per day", icon: "calendar.badge.plus", tint: .accentPrimary))
        insights.append(RadarInsight(title: "Events / Day", value: String(format: "%.1f", stats.averageEventsPerDay), footnote: "Average events per day", icon: "chart.bar.doc.horizontal", tint: .accentGreen))

        if let positiveShare = stats.positiveShare {
            insights.append(RadarInsight(title: "Positive Tone", value: percentString(positiveShare), footnote: "Positive event share", icon: "sun.max.fill", tint: .accentGreen))
        }

        if let negativeShare = stats.negativeShare {
            insights.append(RadarInsight(title: "Negative Tone", value: percentString(negativeShare), footnote: "Negative event share", icon: "cloud.bolt.fill", tint: .accentRed))
        }

        return Array(insights.prefix(8))
    }

    private func statsCloudShare(_ count: Int) -> String {
        guard let stats = viewModel.stats, stats.totalRecordings > 0 else {
            return "—"
        }
        return percentString(Double(count) / Double(stats.totalRecordings))
    }

    private func densityFootnote(_ value: Double, suffix: String) -> String {
        String(format: "%.1f%@", value, suffix)
    }

    private func percentString(_ value: Double) -> String {
        String(format: "%.0f%%", value * 100)
    }

    private func hourLabelShort(_ hour: Int) -> String {
        switch hour {
        case 0: return "12a"
        case 12: return "12p"
        case 1..<12: return "\(hour)a"
        default: return "\(hour - 12)p"
        }
    }

    private func hourLabelLong(_ hour: Int) -> String {
        switch hour {
        case 0: return "12 AM"
        case 12: return "12 PM"
        case 1..<12: return "\(hour) AM"
        default: return "\(hour - 12) PM"
        }
    }

    private func formatTokens(_ count: Int) -> String {
        if count >= 1_000_000 { return String(format: "%.1fM", Double(count) / 1_000_000) }
        if count >= 1_000 { return String(format: "%.1fK", Double(count) / 1_000) }
        return "\(count)"
    }

    private func formatCost(_ usd: Double, zeroAsFree: Bool = true) -> String {
        if usd == 0 {
            return zeroAsFree ? "FREE" : "$0.00"
        }
        if usd < 0.01 {
            return String(format: "$%.4f", usd)
        }
        if usd < 1 {
            return String(format: "$%.3f", usd)
        }
        return String(format: "$%.2f", usd)
    }

    private var statsGridColumns: [GridItem] {
        [GridItem(.flexible()), GridItem(.flexible())]
    }
}

struct CostQuickLookSheet: View {
    let sessionCost: SessionCost?
    let costHistory: CostHistory?
    let pricing: ModelPricing?
    let errorMessage: String?
    let onRefresh: @Sendable () async -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var isRefreshing = false

    private var topModels: [(name: String, calls: Int, cost: Double)] {
        guard let sessionCost else { return [] }
        return sessionCost.byModel.compactMap { key, value in
            guard let object = value.objectValue else { return nil }
            return (
                name: key,
                calls: object["calls"]?.intValue ?? 0,
                cost: object["cost_usd"]?.doubleValue ?? 0
            )
        }
        .sorted { $0.cost > $1.cost }
    }

    private var typeBreakdown: [(name: String, calls: Int, cost: Double)] {
        guard let sessionCost else { return [] }
        return sessionCost.byType.compactMap { key, value in
            guard let object = value.objectValue else { return nil }
            return (
                name: key.replacingOccurrences(of: "_", with: " ").capitalized,
                calls: object["calls"]?.intValue ?? 0,
                cost: object["cost_usd"]?.doubleValue ?? 0
            )
        }
        .sorted { $0.cost > $1.cost }
    }

    private var recentCalls: [CostRecentCall] {
        sessionCost?.recent ?? []
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    if let errorMessage, sessionCost == nil, costHistory == nil {
                        ErrorBanner(message: errorMessage)
                            .padding(.horizontal)
                    }

                    StatsSectionCard(
                        title: "AI Spend Quick Look",
                        subtitle: "Server-tracked usage from PlaudBlender. This is app spend telemetry, not your remaining OpenAI account balance.",
                        systemImage: "dollarsign.gauge.chart.lefthalf.righthalf"
                    ) {
                        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                            MetricTile(
                                title: "Session",
                                value: formatCost(sessionCost?.totalCostUsd ?? 0, zeroAsFree: false),
                                footnote: sessionCost.map { String(format: "%.0fm live window", $0.sessionMinutes) } ?? "Current app session",
                                icon: "dollarsign.circle",
                                tint: .accentGreen
                            )
                            MetricTile(
                                title: "30-Day",
                                value: formatCost(costHistory?.totalCostUsd ?? 0, zeroAsFree: false),
                                footnote: costHistory.map { "\($0.days)-day window" } ?? "Historical window",
                                icon: "calendar.badge.clock",
                                tint: .accentPrimary
                            )
                            MetricTile(
                                title: "Calls",
                                value: "\(sessionCost?.totalCalls ?? 0)",
                                footnote: "Current session calls",
                                icon: "arrow.left.arrow.right.circle",
                                tint: .accentOrange
                            )
                            MetricTile(
                                title: "Avg / Day",
                                value: formatCost(averageDailyCost, zeroAsFree: false),
                                footnote: "Burn rate",
                                icon: "chart.line.uptrend.xyaxis",
                                tint: .accentPurple
                            )
                        }
                    }

                    if !typeBreakdown.isEmpty {
                        StatsSectionCard(
                            title: "By Call Type",
                            subtitle: "Where the spend is landing right now.",
                            systemImage: "square.stack.3d.up"
                        ) {
                            ForEach(Array(typeBreakdown.prefix(4)), id: \.name) { item in
                                quickRow(
                                    title: item.name,
                                    subtitle: "\(item.calls) calls",
                                    trailing: formatCost(item.cost, zeroAsFree: false),
                                    tint: item.cost == 0 ? .secondary : .accentPrimary
                                )

                                if item.name != typeBreakdown.prefix(4).last?.name {
                                    Divider()
                                }
                            }
                        }
                    }

                    if !topModels.isEmpty {
                        StatsSectionCard(
                            title: "Top Models",
                            subtitle: "The models currently consuming the most budget.",
                            systemImage: "cpu"
                        ) {
                            ForEach(Array(topModels.prefix(5)), id: \.name) { item in
                                VStack(alignment: .leading, spacing: 6) {
                                    quickRow(
                                        title: item.name,
                                        subtitle: "\(item.calls) calls",
                                        trailing: formatCost(item.cost, zeroAsFree: false),
                                        tint: item.cost == 0 ? .secondary : .accentGreen
                                    )

                                    GeometryReader { geo in
                                        Capsule()
                                            .fill(Color.secondary.opacity(0.12))
                                            .overlay(alignment: .leading) {
                                                Capsule()
                                                    .fill(Color.accentPrimary.gradient)
                                                    .frame(width: geo.size.width * CGFloat(item.cost / max(topModels.first?.cost ?? 1, 1)))
                                            }
                                    }
                                    .frame(height: 8)
                                }

                                if item.name != topModels.prefix(5).last?.name {
                                    Divider()
                                }
                            }
                        }
                    }

                    if !recentCalls.isEmpty {
                        StatsSectionCard(
                            title: "Recent Calls",
                            subtitle: "Latest model usage recorded by the server-side ledger.",
                            systemImage: "clock.arrow.circlepath"
                        ) {
                            ForEach(Array(recentCalls.prefix(8))) { call in
                                VStack(alignment: .leading, spacing: 8) {
                                    HStack(alignment: .firstTextBaseline) {
                                        VStack(alignment: .leading, spacing: 2) {
                                            Text(call.model)
                                                .font(.caption.weight(.semibold))
                                                .lineLimit(1)
                                                .truncationMode(.middle)
                                            Text(call.type.replacingOccurrences(of: "_", with: " ").capitalized)
                                                .font(.caption2)
                                                .foregroundStyle(.secondary)
                                        }

                                        Spacer()

                                        VStack(alignment: .trailing, spacing: 2) {
                                            Text(formatCost(call.costUsd, zeroAsFree: false))
                                                .font(.caption.weight(.semibold))
                                                .monospacedDigit()
                                            Text(agoLabel(call.agoS))
                                                .font(.caption2)
                                                .foregroundStyle(.secondary)
                                        }
                                    }

                                    HStack(spacing: 10) {
                                        smallMetaPill("In \(formatTokens(call.inputTokens))")
                                        smallMetaPill("Out \(formatTokens(call.outputTokens))")
                                    }
                                }

                                if call.id != recentCalls.prefix(8).last?.id {
                                    Divider()
                                }
                            }
                        }
                    }

                    if let pricing, !pricing.models.isEmpty {
                        StatsSectionCard(
                            title: "Pricing Snapshot",
                            subtitle: "Server-advertised model pricing reference.",
                            systemImage: "tag.circle"
                        ) {
                            ForEach(Array(pricing.models.prefix(6).enumerated()), id: \.offset) { _, model in
                                let name = model["label"]?.stringValue ?? model["model"]?.stringValue ?? model["name"]?.stringValue ?? "Unknown"
                                let provider = model["provider"]?.stringValue?.capitalized ?? "Provider"
                                let input = model["input_per_mtok"]?.doubleValue
                                let output = model["output_per_mtok"]?.doubleValue
                                let tier = model["tier"]?.stringValue?.capitalized ?? "Tier"

                                quickRow(
                                    title: name,
                                    subtitle: "\(provider) · \(tier)",
                                    trailing: pricingString(input: input, output: output),
                                    tint: tier == "Free" ? .accentGreen : .secondary
                                )

                                if name != (pricing.models.prefix(6).last?["label"]?.stringValue ?? pricing.models.prefix(6).last?["model"]?.stringValue ?? pricing.models.prefix(6).last?["name"]?.stringValue) {
                                    Divider()
                                }
                            }
                        }
                    }
                }
                .padding(.vertical)
            }
            .navigationTitle("AI Spend")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") {
                        dismiss()
                    }
                }

                ToolbarItem(placement: platformTrailingToolbarPlacement) {
                    Button {
                        Task {
                            isRefreshing = true
                            await onRefresh()
                            isRefreshing = false
                        }
                    } label: {
                        if isRefreshing {
                            ProgressView()
                        } else {
                            Image(systemName: "arrow.clockwise")
                        }
                    }
                    .disabled(isRefreshing)
                }
            }
        }
    }

    private var averageDailyCost: Double {
        guard let costHistory, costHistory.days > 0 else { return 0 }
        return costHistory.totalCostUsd / Double(costHistory.days)
    }

    private func quickRow(title: String, subtitle: String, trailing: String, tint: Color) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.caption.weight(.semibold))
                Text(subtitle)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Text(trailing)
                .font(.caption.weight(.semibold))
                .foregroundStyle(tint)
                .monospacedDigit()
        }
    }

    private func smallMetaPill(_ text: String) -> some View {
        Text(text)
            .font(.caption2)
            .foregroundStyle(.secondary)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color.secondary.opacity(0.08))
            .clipShape(Capsule())
    }

    private func pricingString(input: Double?, output: Double?) -> String {
        let inText = input.map { String(format: "In $%.2f", $0) } ?? "In —"
        let outText = output.map { String(format: "Out $%.2f", $0) } ?? "Out —"
        return "\(inText) · \(outText) /1M"
    }

    private func agoLabel(_ seconds: Double) -> String {
        if seconds < 60 {
            return String(format: "%.0fs ago", seconds)
        }
        if seconds < 3600 {
            return String(format: "%.0fm ago", seconds / 60)
        }
        return String(format: "%.1fh ago", seconds / 3600)
    }

    private func formatTokens(_ count: Int) -> String {
        if count >= 1_000_000 { return String(format: "%.1fM", Double(count) / 1_000_000) }
        if count >= 1_000 { return String(format: "%.1fK", Double(count) / 1_000) }
        return "\(count)"
    }

    private func formatCost(_ usd: Double, zeroAsFree: Bool = true) -> String {
        if usd == 0 {
            return zeroAsFree ? "FREE" : "$0.00"
        }
        if usd < 0.01 {
            return String(format: "$%.4f", usd)
        }
        if usd < 1 {
            return String(format: "$%.3f", usd)
        }
        return String(format: "$%.2f", usd)
    }
}

private struct StatsSectionCard<Content: View>: View {
    let title: String
    let subtitle: String?
    let systemImage: String
    let content: Content

    init(title: String, subtitle: String? = nil, systemImage: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.subtitle = subtitle
        self.systemImage = systemImage
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: systemImage)
                    .font(.headline)
                    .foregroundStyle(Color.accentPrimary)
                    .frame(width: 24)

                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .font(.headline)
                    if let subtitle {
                        Text(subtitle)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            content
        }
        .padding()
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }
}

private struct MetricTile: View {
    let title: String
    let value: String
    let footnote: String?
    let icon: String
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .foregroundStyle(tint)
                Spacer()
            }

            Text(value)
                .font(.title3.weight(.bold))
                .monospacedDigit()
                .lineLimit(1)
                .minimumScaleFactor(0.7)

            Text(title)
                .font(.caption.weight(.medium))
                .foregroundStyle(.primary)

            if let footnote {
                Text(footnote)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(tint.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

private struct InsightMetricTile: View {
    let insight: RadarInsight

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Image(systemName: insight.icon)
                .foregroundStyle(insight.tint)

            Text(insight.value)
                .font(.subheadline.weight(.semibold))
                .lineLimit(2)

            Text(insight.title)
                .font(.caption.weight(.medium))
                .foregroundStyle(.primary)

            Text(insight.footnote)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(insight.tint.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

private struct CategoryShareRow: View {
    let name: String
    let count: Int
    let share: Double
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(name.replacingOccurrences(of: "_", with: " ").capitalized)
                    .font(.caption.weight(.medium))
                Spacer()
                Text("\(count)")
                    .font(.caption.monospacedDigit())
                Text(String(format: "%.0f%%", share * 100))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(width: 34, alignment: .trailing)
            }

            GeometryReader { geo in
                Capsule()
                    .fill(Color.secondary.opacity(0.12))
                    .overlay(alignment: .leading) {
                        Capsule()
                            .fill(color)
                            .frame(width: geo.size.width * CGFloat(share))
                    }
            }
            .frame(height: 8)
        }
    }
}

private struct RadarInsight: Identifiable {
    let id = UUID()
    let title: String
    let value: String
    let footnote: String
    let icon: String
    let tint: Color
}

private struct SentimentSegment: Identifiable {
    let id = UUID()
    let label: String
    let count: Int
    let color: Color
}

private struct HourEntry: Identifiable {
    let hour: Int
    let count: Int
    var id: Int { hour }
}

private struct DayEntry: Identifiable {
    let label: String
    let count: Int
    let index: Int
    var id: Int { index }
}

struct FlowLayout: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let result = arrange(proposal: proposal, subviews: subviews)
        return result.size
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let result = arrange(proposal: proposal, subviews: subviews)
        for (index, position) in result.positions.enumerated() {
            subviews[index].place(at: CGPoint(x: bounds.minX + position.x, y: bounds.minY + position.y), proposal: .unspecified)
        }
    }

    private func arrange(proposal: ProposedViewSize, subviews: Subviews) -> (positions: [CGPoint], size: CGSize) {
        let maxWidth = proposal.width ?? .infinity
        var positions: [CGPoint] = []
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > maxWidth && x > 0 {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            positions.append(CGPoint(x: x, y: y))
            rowHeight = max(rowHeight, size.height)
            x += size.width + spacing
        }

        return (positions, CGSize(width: maxWidth, height: y + rowHeight))
    }
}

private struct CostDayEntry: Identifiable {
    let index: Int
    let date: String
    let cost: Double
    let calls: Int

    var id: Int { index }

    var displayLabel: String {
        if let parsed = date.iso8601Date {
            return parsed.shortDateString
        }

        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd"
        if let parsed = formatter.date(from: date) {
            return parsed.shortDateString
        }

        return date
    }
}
