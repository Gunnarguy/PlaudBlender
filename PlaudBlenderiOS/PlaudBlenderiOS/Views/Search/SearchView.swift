import SwiftUI

enum SearchSurfaceMode: String, CaseIterable, Identifiable {
    case findMoments
    case askChronos

    var id: String { rawValue }

    var title: String {
        switch self {
        case .findMoments:
            return "Find Moments"
        case .askChronos:
            return "Ask Chronos"
        }
    }

    var subtitle: String {
        switch self {
        case .findMoments:
            return "Search events and recordings"
        case .askChronos:
            return "Ask questions and inspect the evidence"
        }
    }

    var systemImage: String {
        switch self {
        case .findMoments:
            return "magnifyingglass"
        case .askChronos:
            return "brain"
        }
    }
}

struct SearchView: View {
    @Bindable var viewModel: SearchViewModel
    @FocusState private var isSearchFocused: Bool
    @State private var isShowingDateFilter = false
    @State private var selectedMode: SearchSurfaceMode = .findMoments
    @State private var askQuestion = ""
    @State private var isShowingAskSettings = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                modeSelector

                if selectedMode == .findMoments {
                    findMomentsScrollView
                } else {
                    askChronosLayout
                }
            }
            .navigationTitle("Search")
            .toolbar {
                if viewModel.aiAnswer != nil || viewModel.hasAskConversation || !viewModel.query.isEmpty || !askQuestion.isEmpty {
                    ToolbarItem(placement: platformTrailingToolbarPlacement) {
                        Button("Reset") {
                            viewModel.clear()
                            askQuestion = ""
                            selectedMode = .findMoments
                        }
                    }
                }
            }
            .sheet(isPresented: $isShowingDateFilter) {
                DateFilterSheet(
                    startDate: $viewModel.startDate,
                    endDate: $viewModel.endDate,
                    onApply: {
                        isShowingDateFilter = false
                        if !viewModel.query.isEmpty {
                            Task { await viewModel.search() }
                        }
                    }
                )
                .presentationDetents([.medium])
            }
            .sheet(isPresented: $isShowingAskSettings) {
                AskChronosSettingsSheetView(viewModel: viewModel)
                    .presentationDetents([.medium, .large])
            }
            .onAppear {
                if askQuestion.isEmpty {
                    askQuestion = viewModel.query
                }
            }
            .onChange(of: selectedMode) { _, newMode in
                isSearchFocused = false
                if newMode == .askChronos && askQuestion.isEmpty {
                    askQuestion = viewModel.query
                }
            }
        }
    }

    private var findMomentsScrollView: some View {
        ScrollView {
            findMomentsContent
                .padding(.bottom, 12)
        }
        .scrollDismissesKeyboard(.interactively)
    }

    private var findMomentsContent: some View {
        VStack(spacing: 0) {
            searchBar
            filterSection

            if viewModel.startDate != nil || viewModel.endDate != nil {
                dateRangeIndicator
            }

            if let error = viewModel.error {
                errorBanner(error)
            }

            if viewModel.isSearching {
                LoadingView(message: "Searching...")
            } else if !viewModel.results.isEmpty {
                resultsContent
            } else if !viewModel.query.isEmpty {
                EmptyStateView(
                    icon: "magnifyingglass",
                    title: "No Results",
                    message: "Try different search terms or adjust filters."
                )
            } else {
                EmptyStateView(
                    icon: "magnifyingglass",
                    title: "Search Your Timeline",
                    message: "Semantic search across all your recordings and events."
                )
            }
        }
    }

    private var askChronosLayout: some View {
        VStack(spacing: 0) {
            if let error = viewModel.error {
                errorBanner(error)
            }

            askConversationPane

            AskChronosComposerView(
                viewModel: viewModel,
                question: $askQuestion,
                onAsk: askChronos,
                onStartNewThread: startNewAskThread,
                onOpenSettings: { isShowingAskSettings = true }
            )
            .background(.ultraThinMaterial)
            .overlay(alignment: .top) {
                Divider()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
    }

    private var askConversationPane: some View {
        ScrollViewReader { proxy in
            Group {
                if viewModel.hasAskConversation || viewModel.isAskingAI {
                    ScrollView {
                        VStack(spacing: 0) {
                            askConversationContent
                                .padding(.top, 12)
                                .padding(.bottom, 8)

                            if viewModel.isAskingAI {
                                LoadingView(message: "Chronos is answering...")
                                    .padding(.horizontal)
                                    .padding(.bottom, 12)
                            }

                            Color.clear
                                .frame(height: 1)
                                .id("conversation-bottom")
                        }
                    }
                    .scrollDismissesKeyboard(.interactively)
                } else {
                    AskChronosWelcomeView(
                        settings: viewModel.askSettings,
                        promptSuggestions: askPromptSuggestions,
                        onSelectPrompt: { askQuestion = $0 },
                        onOpenSettings: { isShowingAskSettings = true }
                    )
                }
            }
            .onChange(of: viewModel.askConversation.count) { _, _ in
                guard selectedMode == .askChronos else { return }
                withAnimation(.easeOut(duration: 0.25)) {
                    proxy.scrollTo("conversation-bottom", anchor: .bottom)
                }
            }
        }
    }

    private var modeSelector: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Choose how you want to explore your timeline")
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack(spacing: 10) {
                ForEach(SearchSurfaceMode.allCases) { mode in
                    Button {
                        selectedMode = mode
                    } label: {
                        VStack(alignment: .leading, spacing: 6) {
                            Label(mode.title, systemImage: mode.systemImage)
                                .font(.subheadline.weight(.semibold))
                            Text(mode.subtitle)
                                .font(.caption)
                                .foregroundStyle(selectedMode == mode ? Color.primary.opacity(0.85) : Color.secondary)
                                .multilineTextAlignment(.leading)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(14)
                        .background(modeBackground(for: mode))
                        .overlay(
                            RoundedRectangle(cornerRadius: 16)
                                .stroke(modeBorderColor(for: mode), lineWidth: 1)
                        )
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .padding(.horizontal)
        .padding(.top)
        .padding(.bottom, 8)
    }

    private func modeBackground(for mode: SearchSurfaceMode) -> LinearGradient {
        let isSelected = selectedMode == mode
        switch mode {
        case .findMoments:
            return LinearGradient(
                colors: isSelected
                    ? [Color.accentPrimary.opacity(0.14), Color.accentCyan.opacity(0.10)]
                    : [Color.accentPrimary.opacity(0.05), Color.clear],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        case .askChronos:
            return LinearGradient(
                colors: isSelected
                    ? [Color.accentCyan.opacity(0.14), Color.accentPrimary.opacity(0.10)]
                    : [Color.accentCyan.opacity(0.05), Color.clear],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        }
    }

    private func modeBorderColor(for mode: SearchSurfaceMode) -> Color {
        selectedMode == mode ? Color.accentPrimary.opacity(0.22) : Color.primary.opacity(0.08)
    }

    private var searchBar: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Find exact moments")
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                TextField("Search recordings...", text: $viewModel.query)
                    .textFieldStyle(.plain)
                    .focused($isSearchFocused)
                    .submitLabel(.search)
                    .onSubmit { Task { await viewModel.search() } }
                if !viewModel.query.isEmpty {
                    Button {
                        viewModel.clear()
                        askQuestion = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(12)
            .background(.regularMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
        .padding(.horizontal)
        .padding(.bottom, 8)
    }

    private var filterSection: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                // Date filter button
                Button { isShowingDateFilter = true } label: {
                    let hasDate = viewModel.startDate != nil || viewModel.endDate != nil
                    Label("Dates", systemImage: "calendar")
                        .font(.caption)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(hasDate ? Color.accentPrimary.opacity(0.3) : Color.clear)
                        .overlay(Capsule().stroke(Color.accentPrimary, lineWidth: 1))
                        .clipShape(Capsule())
                }
                .buttonStyle(.plain)

                ForEach(EventCategory.allCases, id: \.rawValue) { cat in
                    let isSelected = viewModel.selectedCategories.contains(cat.rawValue)
                    Button {
                        if isSelected {
                            viewModel.selectedCategories.remove(cat.rawValue)
                        } else {
                            viewModel.selectedCategories.insert(cat.rawValue)
                        }
                    } label: {
                        Text(cat.rawValue.capitalized)
                            .font(.caption)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 5)
                            .background(isSelected ? Color.forCategory(cat.rawValue).opacity(0.3) : Color.clear)
                            .overlay(Capsule().stroke(Color.forCategory(cat.rawValue), lineWidth: 1))
                            .clipShape(Capsule())
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal)
        }
    }

    private var dateRangeIndicator: some View {
        HStack(spacing: 6) {
            Image(systemName: "calendar")
                .font(.caption2)
            if let start = viewModel.startDate {
                Text(start.shortDateString)
                    .font(.caption2)
            }
            if viewModel.startDate != nil && viewModel.endDate != nil {
                Text("–")
                    .font(.caption2)
            }
            if let end = viewModel.endDate {
                Text(end.shortDateString)
                    .font(.caption2)
            }
            Spacer()
            Button {
                viewModel.startDate = nil
                viewModel.endDate = nil
                if !viewModel.query.isEmpty {
                    Task { await viewModel.search() }
                }
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.caption2)
            }
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal)
        .padding(.vertical, 4)
    }

    private var resultsContent: some View {
        LazyVStack(spacing: 8) {
            Text("\(viewModel.total) results")
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal)

            ForEach(viewModel.results) { result in
                SearchResultCardView(result: result)
            }
        }
        .padding()
    }

    private var askConversationContent: some View {
        LazyVStack(spacing: 12) {
            ForEach(viewModel.askConversation) { turn in
                AskQuestionBubbleView(
                    question: turn.question,
                    isLatest: turn.id == viewModel.latestAskTurn?.id
                )
                .id("question-\(turn.id)")

                AIAnswerCardView(answer: turn.answer)
                    .id("answer-\(turn.id)")

                if let supportingMoments = turn.supportingMoments {
                    Text(
                        turn.id == viewModel.latestAskTurn?.id
                            ? "\(supportingMoments) supporting moments for this turn"
                            : "Used \(supportingMoments) supporting moments"
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 2)
                }

                if turn.id == viewModel.latestAskTurn?.id, !viewModel.results.isEmpty {
                    Label("Evidence", systemImage: "doc.text.magnifyingglass")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.top, 2)

                    ForEach(viewModel.results) { result in
                        SearchResultCardView(result: result)
                    }
                }
            }
        }
        .padding()
    }

    private var askPromptSuggestions: [String] {
        [
            "What dominated my last three days?",
            "When was the last time I mentioned this project?",
            "Summarize the main meetings from today.",
            "What did I keep circling back to this week?"
        ]
    }

    private func errorBanner(_ message: String) -> some View {
        Text(message)
            .font(.caption)
            .foregroundStyle(.red)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal)
            .padding(.bottom, 8)
    }

    private func askChronos() {
        Task {
            selectedMode = .askChronos
            let didAsk = await viewModel.askAI(question: askQuestion)
            if didAsk {
                askQuestion = ""
            }
        }
    }

    private func startNewAskThread() {
        viewModel.resetAskConversation()
        askQuestion = ""
    }
}

struct AskChronosWelcomeView: View {
    let settings: AskChronosSettings
    let promptSuggestions: [String]
    let onSelectPrompt: (String) -> Void
    let onOpenSettings: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 8) {
                    Label("Ask Chronos", systemImage: "brain")
                        .font(.title3.weight(.semibold))
                        .foregroundStyle(Color.accentPrimary)
                    Text("Ask a question, get an answer, then inspect the supporting moments and continue with follow-ups.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .padding(18)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    LinearGradient(
                        colors: [Color.accentPrimary.opacity(0.10), Color.accentCyan.opacity(0.08)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .clipShape(RoundedRectangle(cornerRadius: 18))

                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Text("Current response settings")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button(action: onOpenSettings) {
                            Label("Change", systemImage: "slider.horizontal.3")
                                .font(.caption.weight(.semibold))
                        }
                        .buttonStyle(.plain)
                    }

                    if !settings.summaryChips.isEmpty {
                        AskChipRow(chips: settings.summaryChips)
                    }
                }

                VStack(alignment: .leading, spacing: 10) {
                    Text("Try one of these")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    ForEach(promptSuggestions, id: \.self) { suggestion in
                        Button {
                            onSelectPrompt(suggestion)
                        } label: {
                            HStack(spacing: 10) {
                                Text(suggestion)
                                    .font(.subheadline)
                                    .multilineTextAlignment(.leading)
                                Spacer(minLength: 12)
                                Image(systemName: "arrow.up.left.and.arrow.down.right.circle")
                                    .foregroundStyle(Color.accentPrimary)
                            }
                            .padding(14)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(.regularMaterial)
                            .clipShape(RoundedRectangle(cornerRadius: 14))
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .scrollDismissesKeyboard(.interactively)
    }
}

// MARK: - Ask Chronos Composer

struct AskChronosComposerView: View {
    @Bindable var viewModel: SearchViewModel
    @Binding var question: String
    let onAsk: () -> Void
    let onStartNewThread: () -> Void
    let onOpenSettings: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 0) {
                TextField(
                    viewModel.hasAskConversation
                        ? "Ask a follow-up about the last answer..."
                        : "Ask Chronos anything...",
                    text: $question,
                    axis: .vertical
                )
                .lineLimit(1...5)
                .textFieldStyle(.plain)
                .padding(.horizontal, 14)
                .padding(.top, 14)
                .padding(.bottom, 12)

                Divider()
                    .padding(.horizontal, 8)

                HStack(alignment: .center, spacing: 10) {
                    Button(action: onOpenSettings) {
                        Label("Settings", systemImage: "slider.horizontal.3")
                    }
                    .buttonStyle(.plain)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)

                    if viewModel.hasAskConversation {
                        Button("New thread", action: onStartNewThread)
                            .buttonStyle(.plain)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    if viewModel.hasAskConversation {
                        Label("Follow-up", systemImage: "arrow.triangle.branch")
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.tertiary)
                    }

                    Button(action: onAsk) {
                        if viewModel.isAskingAI {
                            HStack(spacing: 8) {
                                ProgressView()
                                Text("Asking")
                            }
                        } else {
                            Label(
                                viewModel.hasAskConversation ? "Send" : "Ask",
                                systemImage: "arrow.up.circle.fill"
                            )
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || viewModel.isAskingAI)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
            }
            .background(
                RoundedRectangle(cornerRadius: 18)
                    .fill(Color(uiColor: .secondarySystemBackground))
            )
        }
        .padding(.horizontal)
        .padding(.top, 12)
        .padding(.bottom, 12)
    }
}

struct AskChronosSettingsSheetView: View {
    @Bindable var viewModel: SearchViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var showAdvanced = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Response presets")
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 10) {
                                ForEach(AskChronosPreset.allCases) { preset in
                                    Button {
                                        var updated = viewModel.askSettings
                                        preset.apply(to: &updated)
                                        viewModel.askSettings = updated
                                    } label: {
                                        VStack(alignment: .leading, spacing: 4) {
                                            Text(preset.title)
                                                .font(.subheadline.weight(.semibold))
                                            Text(preset.subtitle)
                                                .font(.caption)
                                                .foregroundStyle(.secondary)
                                        }
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .padding(12)
                                        .background(Color.accentPrimary.opacity(0.06))
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 12)
                                                .stroke(Color.accentPrimary.opacity(0.14), lineWidth: 1)
                                        )
                                        .clipShape(RoundedRectangle(cornerRadius: 12))
                                    }
                                    .buttonStyle(.plain)
                                }
                            }
                        }
                    }
                    .padding(16)
                    .background(.regularMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 16))

                    VStack(alignment: .leading, spacing: 12) {
                        HStack(spacing: 12) {
                            AskMenuField(
                                title: "Model",
                                icon: "cpu",
                                selection: $viewModel.askSettings.model,
                                options: AskChronosSettings.modelOptions
                            )
                            AskMenuField(
                                title: "Reasoning",
                                icon: "sparkles",
                                selection: $viewModel.askSettings.reasoning,
                                options: AskChronosSettings.reasoningOptions
                            )
                        }

                        if !viewModel.askSettings.summaryChips.isEmpty {
                            AskChipRow(chips: viewModel.askSettings.summaryChips)
                        }
                    }
                    .padding(16)
                    .background(.regularMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 16))

                    DisclosureGroup("Advanced response controls", isExpanded: $showAdvanced) {
                        VStack(alignment: .leading, spacing: 14) {
                            HStack(spacing: 12) {
                                AskMenuField(
                                    title: "Verbosity",
                                    icon: "text.alignleft",
                                    selection: $viewModel.askSettings.verbosity,
                                    options: AskChronosSettings.verbosityOptions
                                )
                                AskMenuField(
                                    title: "Reasoning summary",
                                    icon: "list.bullet.rectangle",
                                    selection: $viewModel.askSettings.reasoningSummary,
                                    options: AskChronosSettings.reasoningSummaryOptions
                                )
                            }

                            AskMenuField(
                                title: "Service tier",
                                icon: "speedometer",
                                selection: $viewModel.askSettings.serviceTier,
                                options: AskChronosSettings.serviceTierOptions
                            )

                            if viewModel.askSettings.supportsTemperatureControl {
                                OptionalSliderField(
                                    title: "Temperature",
                                    subtitle: "Leave off to use the API default.",
                                    value: $viewModel.askSettings.temperature,
                                    range: 0.0...2.0,
                                    step: 0.1,
                                    defaultValue: 1.0
                                )
                            } else {
                                settingsInfoRow(
                                    title: "Temperature unavailable",
                                    message: "OpenAI’s GPT-5 family guidance focuses on reasoning effort and text verbosity instead of temperature tuning."
                                )
                            }

                            OptionalSliderField(
                                title: "Top-p",
                                subtitle: "Nucleus sampling. Leave off for the server default.",
                                value: $viewModel.askSettings.topP,
                                range: 0.0...1.0,
                                step: 0.05,
                                defaultValue: 1.0
                            )

                            OptionalStepperField(
                                title: "Max output tokens",
                                subtitle: "Caps visible output plus reasoning tokens.",
                                value: $viewModel.askSettings.maxOutputTokens,
                                range: 256...8192,
                                step: 256,
                                defaultValue: 2048
                            )

                            Text("These settings apply to the next Chronos answer you send from the composer.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .padding(.top, 8)
                    }
                    .font(.subheadline.weight(.semibold))
                    .padding(16)
                    .background(.regularMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                }
                .padding()
            }
            .navigationTitle("Chronos Settings")
            .platformNavigationBarTitleDisplayModeInline()
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Reset") {
                        viewModel.askSettings = AskChronosSettings()
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func settingsInfoRow(title: String, message: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.subheadline.weight(.semibold))
            Text(message)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color(uiColor: .secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - Date Filter Sheet

struct DateFilterSheet: View {
    @Binding var startDate: Date?
    @Binding var endDate: Date?
    let onApply: () -> Void

    @State private var localStart: Date = Calendar.current.date(byAdding: .day, value: -7, to: Date()) ?? Date()
    @State private var localEnd: Date = Date()
    @State private var hasStartDate = false
    @State private var hasEndDate = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Toggle("Start Date", isOn: $hasStartDate)
                    if hasStartDate {
                        DatePicker("From", selection: $localStart, displayedComponents: .date)
                    }
                }
                Section {
                    Toggle("End Date", isOn: $hasEndDate)
                    if hasEndDate {
                        DatePicker("To", selection: $localEnd, displayedComponents: .date)
                    }
                }
                Section {
                    HStack {
                        Button("Last 7 Days") {
                            hasStartDate = true
                            hasEndDate = true
                            localStart = Calendar.current.date(byAdding: .day, value: -7, to: Date()) ?? Date()
                            localEnd = Date()
                        }
                        .font(.caption)
                        Spacer()
                        Button("Last 30 Days") {
                            hasStartDate = true
                            hasEndDate = true
                            localStart = Calendar.current.date(byAdding: .day, value: -30, to: Date()) ?? Date()
                            localEnd = Date()
                        }
                        .font(.caption)
                        Spacer()
                        Button("Clear") {
                            hasStartDate = false
                            hasEndDate = false
                        }
                        .font(.caption)
                    }
                }
            }
            .navigationTitle("Date Range")
            .platformNavigationBarTitleDisplayModeInline()
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Apply") {
                        startDate = hasStartDate ? localStart : nil
                        endDate = hasEndDate ? localEnd : nil
                        onApply()
                    }
                }
            }
            .onAppear {
                if let s = startDate {
                    localStart = s
                    hasStartDate = true
                }
                if let e = endDate {
                    localEnd = e
                    hasEndDate = true
                }
            }
        }
    }
}

// MARK: - AI Answer Card

struct AskQuestionBubbleView: View {
    let question: String
    let isLatest: Bool

    var body: some View {
        HStack {
            Spacer(minLength: 48)
            VStack(alignment: .leading, spacing: 6) {
                Text(isLatest ? "Latest question" : "Question")
                    .font(.caption)
                    .foregroundStyle(Color.white.opacity(0.72))
                Text(question)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.white)
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                LinearGradient(
                    colors: [Color.accentPrimary, Color.accentCyan.opacity(0.88)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            )
            .clipShape(RoundedRectangle(cornerRadius: 16))
        }
    }
}

struct AIAnswerCardView: View {
    let answer: AIAnswer

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top) {
                Label("Chronos Answer", systemImage: "brain")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundStyle(Color.accentPrimary)
                Spacer()
                if let responseId = answer.responseId, !responseId.isEmpty {
                    Text(responseId)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }

            if !answer.detailChips.isEmpty {
                AskChipRow(chips: answer.detailChips)
            }

            Text(answer.answer)
                .font(.subheadline)

            if let reasoningSummary = answer.reasoningSummary,
               !reasoningSummary.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                DisclosureGroup("Reasoning summary") {
                    Text(reasoningSummary)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.top, 6)
                }
                .font(.caption.weight(.semibold))
            }
        }
        .padding()
        .background(
            LinearGradient(
                colors: [Color.accentPrimary.opacity(0.08), Color.accentCyan.opacity(0.06)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .stroke(Color.accentPrimary.opacity(0.14), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }
}

struct AskChipRow: View {
    let chips: [AskChip]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(chips) { chip in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(chip.label.uppercased())
                            .font(.system(size: 10, weight: .semibold, design: .rounded))
                            .foregroundStyle(.secondary)
                        Text(chip.value)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.primary)
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(Color.white.opacity(0.65))
                    .overlay(
                        Capsule()
                            .stroke(Color.accentPrimary.opacity(0.12), lineWidth: 1)
                    )
                    .clipShape(Capsule())
                }
            }
        }
    }
}

struct AskMenuField: View {
    let title: String
    let icon: String
    @Binding var selection: String
    let options: [AskChoice]

    var body: some View {
        Menu {
            ForEach(options) { option in
                Button {
                    selection = option.id
                } label: {
                    Label(option.title, systemImage: selection == option.id ? "checkmark.circle.fill" : "circle")
                }
            }
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                Text(title)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                HStack(spacing: 10) {
                    Image(systemName: icon)
                        .foregroundStyle(Color.accentPrimary)
                    Text(selectedTitle)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                    Spacer()
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                if let detail = options.first(where: { $0.id == selection })?.detail {
                    Text(detail)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.regularMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
        .buttonStyle(.plain)
    }

    private var selectedTitle: String {
        options.first(where: { $0.id == selection })?.title ?? selection
    }
}

struct OptionalSliderField: View {
    let title: String
    let subtitle: String
    @Binding var value: Double?
    let range: ClosedRange<Double>
    let step: Double
    let defaultValue: Double

    private var isEnabled: Binding<Bool> {
        Binding(
            get: { value != nil },
            set: { enabled in
                value = enabled ? (value ?? defaultValue) : nil
            }
        )
    }

    private var liveValue: Binding<Double> {
        Binding(
            get: { value ?? defaultValue },
            set: { value = $0 }
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.subheadline.weight(.semibold))
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Toggle("Override", isOn: isEnabled)
                    .labelsHidden()
            }

            if value != nil {
                HStack {
                    Slider(value: liveValue, in: range, step: step)
                    Text(formattedValue)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .frame(width: 40, alignment: .trailing)
                }
            }
        }
    }

    private var formattedValue: String {
        String(format: "%.2f", value ?? defaultValue)
            .replacingOccurrences(of: #"\.?0+$"#, with: "", options: .regularExpression)
    }
}

struct OptionalStepperField: View {
    let title: String
    let subtitle: String
    @Binding var value: Int?
    let range: ClosedRange<Int>
    let step: Int
    let defaultValue: Int

    private var isEnabled: Binding<Bool> {
        Binding(
            get: { value != nil },
            set: { enabled in
                value = enabled ? (value ?? defaultValue) : nil
            }
        )
    }

    private var liveValue: Binding<Int> {
        Binding(
            get: { value ?? defaultValue },
            set: { value = min(max($0, range.lowerBound), range.upperBound) }
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.subheadline.weight(.semibold))
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Toggle("Override", isOn: isEnabled)
                    .labelsHidden()
            }

            if value != nil {
                Stepper(value: liveValue, in: range, step: step) {
                    Text("\(value ?? defaultValue) tokens")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
        }
    }
}

// MARK: - Search Result Card

struct SearchResultCardView: View {
    let result: SearchResult

    var body: some View {
        NavigationLink(value: result.event.recordingId) {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    CategoryPill(category: result.event.category)
                    Spacer()
                    Text(String(format: "%.0f%%", result.score * 100))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
                Text(result.event.cleanText)
                    .font(.subheadline)
                    .lineLimit(3)
                    .multilineTextAlignment(.leading)

                HStack {
                    Text(result.event.startTs.iso8601Date?.shortDateString ?? result.event.startTs)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    if !result.event.keywords.isEmpty {
                        Text("· " + result.event.keywords.prefix(3).joined(separator: ", "))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
            }
            .padding(12)
            .background(.regularMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
    }
}
