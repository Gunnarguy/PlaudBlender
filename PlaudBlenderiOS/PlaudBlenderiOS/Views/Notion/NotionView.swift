import SwiftUI

struct NotionView: View {
    @Environment(NotionViewModel.self) private var viewModel
    @Environment(XRayViewModel.self) private var xray
    @Environment(\.scenePhase) private var scenePhase
    @State private var showDatabaseSwitcher = false
    @State private var showBulkImportConfirmation = false
    @State private var selectedOverrideTarget: NotionRecording?
    @State private var overrideRecordingId = ""
    @State private var selectedDuplicateGroup: NotionDuplicateGroup?
    @State private var groupOverrideRecordingId = ""

    var body: some View {
        NavigationStack {
            Group {
                if viewModel.isLoading && viewModel.authStatus == nil && viewModel.status == nil {
                    LoadingView(message: "Checking Notion...")
                } else if !viewModel.isAuthenticated {
                    disconnectedView
                } else if viewModel.shouldShowDatabasePicker {
                    databaseSelectionView
                } else if let status = viewModel.status, status.isConnected {
                    connectedView(status)
                } else {
                    disconnectedView
                }
            }
            .navigationTitle("Notion")
            .refreshable { await viewModel.loadAll() }
            .task { await viewModel.loadAll() }
            .onChange(of: scenePhase) { _, newPhase in
                guard newPhase == .active else { return }
                Task { await viewModel.refreshAfterAuthorization() }
            }
            .alert(viewModel.supportsSafeBatchImport ? "Start Safe Notion Batch?" : "Start Notion Import?", isPresented: $showBulkImportConfirmation) {
                Button("Cancel", role: .cancel) {}
                Button(viewModel.supportsSafeBatchImport ? "Import Next Safe Batch" : "Import From Notion") {
                    Task { _ = await viewModel.startImport() }
                }
            } message: {
                Text(safeBatchWarningMessage)
            }
            .sheet(isPresented: $showDatabaseSwitcher) {
                databaseSwitcherSheet
            }
            .sheet(item: $selectedOverrideTarget) { recording in
                manualOverrideSheet(for: recording)
            }
            .sheet(item: $selectedDuplicateGroup) { group in
                duplicateGroupOverrideSheet(for: group)
            }
        }
    }

    private var disconnectedView: some View {
        VStack(spacing: 18) {
            EmptyStateView(
                icon: "link.badge.plus",
                title: "Connect Notion",
                message: "Authorize Notion, choose the database you want Chronos to watch, then import unmatched pages."
            )

            VStack(spacing: 10) {
                Button {
                    startOAuthFlow()
                } label: {
                    Label(viewModel.isAuthorizing ? "Authorizing..." : "Connect Notion", systemImage: "person.badge.key")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(viewModel.isAuthorizing)
            }
            .padding(.horizontal)

            if let error = viewModel.error {
                ErrorBanner(message: error)
            }
        }
    }

    private var databaseSelectionView: some View {
        ScrollView {
            VStack(spacing: 16) {
                if let error = viewModel.error {
                    ErrorBanner(message: error)
                }

                VStack(alignment: .leading, spacing: 8) {
                    Label("Workspace", systemImage: "building.columns")
                        .font(.headline)
                    Text(viewModel.workspaceName ?? "Authenticated with Notion")
                        .font(.subheadline)
                    Text(viewModel.authModeLabel)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("Pick the database Chronos should sync from. This is the step that actually connects Notion content into the app.")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Button {
                        startOAuthFlow()
                    } label: {
                        Label(viewModel.usesOAuth ? "Reconnect OAuth" : "Connect with OAuth", systemImage: "person.badge.key")
                    }
                    .buttonStyle(.bordered)
                    .disabled(viewModel.isAuthorizing)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .background(.regularMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .padding(.horizontal)

                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Text("Available Databases")
                            .font(.headline)
                        Spacer()
                        Button("Refresh") {
                            Task { await viewModel.loadDatabases() }
                        }
                        .font(.caption)
                    }

                    if viewModel.databases.isEmpty {
                        Text("No accessible databases found yet.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    ForEach(viewModel.databases) { database in
                        Button {
                            Task { _ = await viewModel.selectDatabase(dbId: database.id) }
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(database.title)
                                        .font(.subheadline.weight(.semibold))
                                    Text(database.id)
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                    if let pageCount = database.pageCount {
                                        Text("\(pageCount) pages")
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                }

                                Spacer()

                                if viewModel.selectedDatabaseId == database.id {
                                    ProgressView()
                                        .controlSize(.small)
                                } else {
                                    Image(systemName: "chevron.right")
                                        .foregroundStyle(.tertiary)
                                }
                            }
                            .padding(12)
                            .background(.regularMaterial)
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding()
                .background(.regularMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .padding(.horizontal)
            }
            .padding(.vertical)
        }
    }

    private func connectedView(_ status: NotionStatus) -> some View {
        ScrollView {
            VStack(spacing: 16) {
                if let error = viewModel.error {
                    ErrorBanner(message: error)
                }

                // Connection info
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                    VStack(alignment: .leading) {
                        Text("Connected")
                            .font(.headline)
                        Text(viewModel.authModeLabel)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        if let db = status.databaseTitle {
                            Text(db)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        if let workspace = viewModel.workspaceName {
                            Text(workspace)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                    Spacer()
                    VStack(alignment: .trailing, spacing: 6) {
                        Text("\(status.totalPages) pages")
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        Button {
                            startOAuthFlow()
                        } label: {
                            Text(viewModel.usesOAuth ? "Reconnect OAuth" : "Connect OAuth")
                                .font(.caption.weight(.semibold))
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .disabled(viewModel.isAuthorizing)

                        Button {
                            Task {
                                await viewModel.loadDatabases()
                                showDatabaseSwitcher = true
                            }
                        } label: {
                            Text("Switch DB")
                                .font(.caption.weight(.semibold))
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                }
                .padding()
                .background(.green.opacity(0.05))
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .padding(.horizontal)

                HStack(spacing: 12) {
                    statCard(
                        title: viewModel.importPreview != nil ? "Server Pending" : "Loaded Unmatched",
                        value: "\(viewModel.importPreview != nil ? viewModel.serverPendingImport : viewModel.unmatchedCount)",
                        color: .orange
                    )
                    statCard(
                        title: viewModel.importPreview != nil ? "Matched" : "Loaded Imported",
                        value: "\(viewModel.importPreview != nil ? viewModel.matchedOrImportedCount : viewModel.importedCount)",
                        color: .green
                    )
                    statCard(title: "Total", value: "\(max(viewModel.totalKnownPages, status.totalPages))")
                }
                .padding(.horizontal)

                notionSnapshotSection(status)
                    .padding(.horizontal)

                // Coverage calendar
                if let coverage = viewModel.coverage, let calendar = coverage.calendar, !calendar.isEmpty {
                    coverageCalendarSection(coverage)
                }

                // Import section
                VStack(alignment: .leading, spacing: 8) {
                    Label("Import", systemImage: "square.and.arrow.down")
                        .font(.headline)

                    if viewModel.supportsSafeBatchImport, viewModel.serverPendingImport > 0 {
                        VStack(alignment: .leading, spacing: 6) {
                            Label("Server Preview", systemImage: "checklist")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.orange)
                            Text("The backend now imports only the next deduped batch of up to \(viewModel.safeBatchLimit) pages. Server-side preview currently shows \(viewModel.serverPendingImport) effective pending pages, down from \(viewModel.serverPendingImportRaw) raw pending pages after collapsing \(viewModel.duplicatePagesCollapsed) exact duplicate pages.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .padding(10)
                        .background(.orange.opacity(0.08))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    }

                    Button {
                        showBulkImportConfirmation = true
                    } label: {
                        Label(importButtonTitle, systemImage: viewModel.isImporting ? "hourglass" : "square.and.arrow.down.on.square")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(viewModel.isImporting || viewModel.isRunningImportSweep || viewModel.serverPendingImport == 0)

                    if viewModel.supportsSafeBatchImport, viewModel.serverPendingImport > viewModel.safeBatchLimit {
                        Button {
                            Task { _ = await viewModel.importAllPending() }
                        } label: {
                            Label(
                                viewModel.isRunningImportSweep ? "Importing Remaining Batches..." : "Import Everything Pending",
                                systemImage: viewModel.isRunningImportSweep ? "hourglass.circle" : "arrow.down.circle"
                            )
                            .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .disabled(viewModel.isImporting || viewModel.isRunningImportSweep)
                    }

                    Text(importSummaryText)
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    if let progress = viewModel.importProgress {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text(progress.status.capitalized)
                                    .font(.subheadline.weight(.semibold))
                                Spacer()
                                Text("\(progress.imported)/\(progress.total)")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }

                            ProgressView(value: progress.total > 0 ? Double(progress.imported) / Double(progress.total) : 0)

                            if let currentTitle = progress.currentTitle {
                                Text("Current: \(currentTitle)")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(2)
                            }

                            HStack {
                                Text("Pending \(progress.pending)")
                                Spacer()
                                Text("Skipped \(progress.skipped)")
                                Spacer()
                                Text("Failed \(progress.failed)")
                                    .foregroundStyle(progress.failed > 0 ? .red : .secondary)
                            }
                            .font(.caption)

                            if let firstError = progress.errors.first {
                                Text(firstError)
                                    .font(.caption2)
                                    .foregroundStyle(.red)
                                    .lineLimit(2)
                            }
                        }
                    }
                }
                .padding()
                .background(.regularMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .padding(.horizontal)

                if let review = viewModel.matchReview {
                    notionReviewSection(review)
                }

                if !xray.notionEvents.isEmpty {
                    notionLiveFeedSection
                }

                // Recordings list
                if !viewModel.recordings.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Notion Pages")
                                .font(.headline)
                            Spacer()
                            Text("\(viewModel.recordings.count) of \(viewModel.totalRecordings)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .padding(.horizontal)
                        ForEach(viewModel.recordings) { rec in
                            VStack(alignment: .leading, spacing: 8) {
                                HStack(alignment: .top) {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(rec.title)
                                            .font(.subheadline.weight(.semibold))
                                        if let date = rec.date ?? rec.createdTime {
                                            Text(date)
                                                .font(.caption2)
                                                .foregroundStyle(.secondary)
                                        }
                                    }
                                    Spacer()
                                    pageStatusBadge(for: rec)
                                }

                                if let summary = rec.summary, !summary.isEmpty {
                                    Text(summary)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(2)
                                }

                                if !rec.isImportedToChronos {
                                    Text(rec.pageId)
                                        .font(.caption2.monospaced())
                                        .foregroundStyle(.tertiary)
                                        .textSelection(.enabled)
                                }

                                HStack(spacing: 10) {
                                    if let category = rec.category {
                                        CategoryPill(category: category)
                                    }
                                    if let duration = rec.duration {
                                        Text(duration)
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                    }
                                    if let urlString = rec.url, let url = URL(string: urlString) {
                                        Spacer()
                                        Link(destination: url) {
                                            Label("Open", systemImage: "arrow.up.right.square")
                                                .font(.caption)
                                        }
                                    }

                                    if !rec.isImportedToChronos, viewModel.matchReview != nil {
                                        Button {
                                            overrideRecordingId = viewModel.manualOverrides[rec.pageId] ?? ""
                                            selectedOverrideTarget = rec
                                        } label: {
                                            Label("Override", systemImage: "slider.horizontal.3")
                                                .font(.caption)
                                        }
                                        .buttonStyle(.bordered)
                                        .controlSize(.small)
                                    }
                                }
                            }
                            .padding(10)
                            .background(.regularMaterial)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                            .onAppear {
                                if rec.id == viewModel.recordings.last?.id {
                                    Task { await viewModel.loadMoreRecordings() }
                                }
                            }
                        }
                        if viewModel.isLoadingMore {
                            ProgressView()
                                .frame(maxWidth: .infinity)
                                .padding()
                        }
                    }
                    .padding(.horizontal)
                }
            }
            .padding(.vertical)
        }
    }

    private func coverageCalendarSection(_ coverage: NotionCoverageResponse) -> some View {
        let days = coverage.calendar ?? []

        return VStack(alignment: .leading, spacing: 10) {
            Label("Coverage Calendar", systemImage: "calendar")
                .font(.headline)

            HStack(spacing: 16) {
                if let cn = coverage.totalChronos {
                    VStack {
                        Text("\(cn)").font(.title3.bold()).monospacedDigit()
                        Text("Chronos").font(.caption2).foregroundStyle(.secondary)
                    }
                }
                if let nn = coverage.totalNotion {
                    VStack {
                        Text("\(nn)").font(.title3.bold()).monospacedDigit()
                        Text("Notion").font(.caption2).foregroundStyle(.secondary)
                    }
                }
            }

            // Legend
            HStack(spacing: 12) {
                legendDot(.green, "Both")
                legendDot(.blue, "Chronos only")
                legendDot(.orange, "Notion only")
                legendDot(.secondary.opacity(0.2), "Neither")
            }
            .font(.caption2)

            // Calendar grid (last 30 days, 7 cols)
            let columns = Array(repeating: GridItem(.flexible(), spacing: 3), count: 7)
            LazyVGrid(columns: columns, spacing: 3) {
                ForEach(days.suffix(35), id: \.date) { day in
                    let color: Color = {
                        if day.hasChronos && day.hasNotion { return .green }
                        if day.hasChronos { return .blue }
                        if day.hasNotion { return .orange }
                        return .secondary.opacity(0.15)
                    }()
                    RoundedRectangle(cornerRadius: 3)
                        .fill(color)
                        .frame(height: 16)
                        .overlay {
                            Text(String(day.date.suffix(2)))
                                .font(.system(size: 7))
                                .foregroundStyle(.white.opacity(0.7))
                        }
                }
            }
        }
        .padding()
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }

    private func legendDot(_ color: Color, _ label: String) -> some View {
        HStack(spacing: 4) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(label)
        }
    }

    private func startOAuthFlow() {
        Task { @MainActor in
            guard let anchor = currentPresentationAnchor() else { return }
            await viewModel.startOAuthFlow(anchor: anchor)
        }
    }

    private var databaseSwitcherSheet: some View {
        NavigationStack {
            List(viewModel.databases) { database in
                Button {
                    Task {
                        _ = await viewModel.selectDatabase(dbId: database.id)
                        showDatabaseSwitcher = false
                    }
                } label: {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(database.title)
                                .font(.subheadline.weight(.semibold))
                            Text(database.id)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                        Spacer()
                        if viewModel.status?.databaseId == database.id {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                        }
                    }
                }
                .buttonStyle(.plain)
            }
            .navigationTitle("Switch Database")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { showDatabaseSwitcher = false }
                }
                ToolbarItem(placement: platformTrailingToolbarPlacement) {
                    Button("Refresh") {
                        Task { await viewModel.loadDatabases() }
                    }
                }
            }
        }
    }

    private func statCard(title: String, value: String, color: Color = .primary) -> some View {
        VStack(spacing: 4) {
            Text(value)
                .font(.title3.bold())
                .monospacedDigit()
                .foregroundStyle(color)
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func pageStatusBadge(for recording: NotionRecording) -> some View {
        Group {
            if viewModel.manualOverrides[recording.pageId] != nil {
                Label("Override", systemImage: "slider.horizontal.3")
                    .foregroundStyle(.blue)
            } else if recording.isImportedToChronos {
                Label("Imported", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
            } else {
                Label("Unmatched", systemImage: "exclamationmark.circle")
                    .foregroundStyle(.orange)
            }
        }
        .font(.caption)
    }

    private func notionSnapshotSection(_ status: NotionStatus) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Server Snapshot")
                .font(.headline)

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                statCard(title: "Workspace", value: viewModel.workspaceName ?? "Unknown")
                statCard(title: "Database", value: status.databaseTitle ?? "Selected")
                statCard(title: "Server Pending", value: "\(viewModel.serverPendingImport)", color: .orange)
                statCard(title: "Matched", value: "\(viewModel.matchedToExistingCount)", color: .green)
                statCard(title: "Raw Pending", value: "\(viewModel.serverPendingImportRaw)")
                statCard(title: "Dupes Collapsed", value: "\(viewModel.duplicatePagesCollapsed)", color: .blue)
                statCard(title: "Overrides", value: "\(viewModel.manualOverrideCount)")
                statCard(title: "Import State", value: viewModel.isImporting ? "Running" : "Idle", color: viewModel.isImporting ? .green : .primary)
            }
        }
    }

    private var safeBatchWarningMessage: String {
        if viewModel.supportsSafeBatchImport {
            return "The backend preview currently shows \(viewModel.serverPendingImport) effective pending Notion pages from \(viewModel.serverPendingImportRaw) raw pending pages. \(viewModel.duplicatePagesCollapsed) exact duplicate pages are already collapsed server-side. Starting this action imports only the next deduped batch of up to \(viewModel.safeBatchLimit) pages."
        }

        return "This backend does not expose safe-batch Notion import yet. Starting this action will use the older import flow for the currently selected Notion database."
    }

    private var importButtonTitle: String {
        if viewModel.isImporting {
            return viewModel.supportsSafeBatchImport ? "Safe Batch Running" : "Import Running"
        }
        return viewModel.supportsSafeBatchImport ? "Import Next Safe Batch" : "Import From Notion"
    }

    private var importSummaryText: String {
        if viewModel.supportsSafeBatchImport {
            return "Loaded here: \(viewModel.recordings.count) of \(viewModel.totalRecordings) pages. Known Notion total: \(viewModel.totalKnownPages). Effective server backlog: \(viewModel.serverPendingImport). Safe batch size: up to \(viewModel.safeBatchLimit) pages per run."
        }

        return "Loaded here: \(viewModel.recordings.count) of \(viewModel.totalRecordings) pages. Known Notion total: \(viewModel.totalKnownPages). This backend is using the older full import path instead of the newer safe-batch flow."
    }

    private func notionReviewSection(_ review: NotionMatchReview) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Review", systemImage: "magnifyingglass.circle")
                .font(.headline)

            Text("The backend has exhausted high-confidence automatic matching. Use this section to review any suggested aliases, see duplicate transcript groups, and apply manual overrides when you know the correct Chronos recording ID.")
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack(spacing: 12) {
                statCard(title: "Pending", value: "\(review.pendingTotal)", color: .orange)
                statCard(title: "Aliases", value: "\(review.highConfidenceTranscriptAliasCount)")
                statCard(title: "Duplicate Groups", value: "\(review.duplicateGroupCount)")
                statCard(title: "Overrides", value: "\(review.manualOverrideCount)", color: .blue)
            }

            if !review.highConfidenceTranscriptAliases.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Suggested Matches")
                        .font(.subheadline.weight(.semibold))

                    ForEach(Array(review.highConfidenceTranscriptAliases.prefix(5))) { alias in
                        VStack(alignment: .leading, spacing: 6) {
                            Text(alias.title)
                                .font(.subheadline.weight(.semibold))
                            if let date = alias.date {
                                Text(date)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                            Text("Candidate: \(alias.candidateTitle)")
                                .font(.caption)
                            Text(alias.candidateRecordingId)
                                .font(.caption2.monospaced())
                                .foregroundStyle(.tertiary)
                                .textSelection(.enabled)

                            HStack {
                                Text("Similarity \(alias.transcriptSimilarity.formatted(.number.precision(.fractionLength(2))))")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                Spacer()
                                Button("Use Candidate") {
                                    Task {
                                        _ = await viewModel.applyManualOverride(
                                            pageId: alias.pageId,
                                            recordingId: alias.candidateRecordingId
                                        )
                                    }
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                            }
                        }
                        .padding(10)
                        .background(.thinMaterial)
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    }
                }
            }

            if !review.duplicateGroups.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Duplicate Transcript Groups")
                        .font(.subheadline.weight(.semibold))

                    ForEach(Array(review.duplicateGroups.prefix(5))) { group in
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text("\(group.groupSize) exact duplicates")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.orange)
                                Spacer()
                                Button("Group Override") {
                                    groupOverrideRecordingId = ""
                                    selectedDuplicateGroup = group
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                            }

                            ForEach(group.pages.prefix(3)) { page in
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(page.title)
                                        .font(.caption.weight(.medium))
                                    HStack {
                                        if let date = page.date {
                                            Text(date)
                                        }
                                        Text(page.pageId)
                                            .font(.caption2.monospaced())
                                            .foregroundStyle(.tertiary)
                                    }
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                }
                            }
                        }
                        .padding(10)
                        .background(.thinMaterial)
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    }
                }
            }
        }
        .padding()
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }

    private func manualOverrideSheet(for recording: NotionRecording) -> some View {
        NavigationStack {
            Form {
                Section("Notion Page") {
                    Text(recording.title)
                    Text(recording.pageId)
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                    if let date = recording.date ?? recording.createdTime {
                        Text(date)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("Chronos Recording") {
                    TextField("Recording ID", text: $overrideRecordingId)
                        .platformTextEntryBehavior()
                    if let currentOverride = viewModel.manualOverrides[recording.pageId], !currentOverride.isEmpty {
                        Text("Current override: \(currentOverride)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                }
            }
            .navigationTitle("Manual Override")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") {
                        selectedOverrideTarget = nil
                    }
                }
                ToolbarItem(placement: platformLeadingToolbarPlacement) {
                    Button("Clear") {
                        Task {
                            if await viewModel.clearManualOverride(pageId: recording.pageId) {
                                selectedOverrideTarget = nil
                            }
                        }
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Apply") {
                        Task {
                            if await viewModel.applyManualOverride(pageId: recording.pageId, recordingId: overrideRecordingId) {
                                selectedOverrideTarget = nil
                            }
                        }
                    }
                    .disabled(overrideRecordingId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
        }
    }

    private func duplicateGroupOverrideSheet(for group: NotionDuplicateGroup) -> some View {
        NavigationStack {
            Form {
                Section("Chronos Recording") {
                    TextField("Recording ID", text: $groupOverrideRecordingId)
                        .platformTextEntryBehavior()
                    Text("Apply the same Chronos recording ID to every Notion page in this exact duplicate transcript group.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("Pages") {
                    ForEach(group.pages) { page in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(page.title)
                            Text(page.pageId)
                                .font(.caption.monospaced())
                                .foregroundStyle(.tertiary)
                                .textSelection(.enabled)
                            if let date = page.date {
                                Text(date)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Group Override")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") {
                        selectedDuplicateGroup = nil
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Apply") {
                        Task {
                            if await viewModel.applyBulkOverride(
                                pageIds: group.pages.map(\.pageId),
                                recordingId: groupOverrideRecordingId
                            ) {
                                selectedDuplicateGroup = nil
                            }
                        }
                    }
                    .disabled(groupOverrideRecordingId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
        }
    }

    private var notionLiveFeedSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("Live Notion Feed", systemImage: "bolt.horizontal.circle")
                    .font(.headline)
                Spacer()
                Circle()
                    .fill(xray.isLiveUpdating ? .green : .secondary)
                    .frame(width: 8, height: 8)
                Text(xray.isLiveUpdating ? "Live" : "Paused")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            Text("Includes recent Notion-related activity pulled from server history, even from before the app was opened.")
                .font(.caption)
                .foregroundStyle(.secondary)

            ForEach(Array(xray.notionEvents.prefix(6))) { event in
                HStack(alignment: .top, spacing: 10) {
                    Text(event.source.uppercased())
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(.secondary)
                        .frame(width: 54, alignment: .leading)

                    VStack(alignment: .leading, spacing: 2) {
                        Text(event.message)
                            .font(.caption)
                            .lineLimit(2)
                        HStack(spacing: 8) {
                            Text(event.op)
                            Spacer()
                            Text(event.date.timeString)
                        }
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    }
                }
                .padding(10)
                .background(.thinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 10))
            }
        }
        .padding()
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }
}
