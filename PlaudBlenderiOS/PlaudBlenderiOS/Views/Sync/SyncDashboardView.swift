import SwiftUI

struct SyncDashboardView: View {
    @Environment(SyncViewModel.self) private var viewModel
    @Environment(XRayViewModel.self) private var xray
    @State private var showConsole = true
    @State private var consoleAutoScroll = true
    @State private var consoleEventLimit = 30
    @State private var showPhaseDetail = true
    @State private var selectedConsoleSourceFilter: String? = nil
    @State private var selectedStage: PipelineStage = .full
    @State private var showBatchWorkflowSheet = false
    @State private var batchDaysBack = 7
    @State private var batchLimit = 5
    @State private var batchTemplateId = ""
    @State private var batchWorkflowModel = "openai"
    @State private var uploadModel = "openai"
    @State private var uploadTemplateId = ""

    private let workflowProviders = ["openai", "gemini", "claude", "local"]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 12) {
                    // 0. Service health bar — unified integration status
                    ServiceStatusBar(
                        systemStatus: viewModel.systemStatus,
                        isLoading: viewModel.isLoadingSystemStatus
                    ) {
                        await viewModel.loadSystemStatus()
                    }
                    .background(.regularMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .padding(.horizontal)

                    // Qdrant / pipeline readiness warning
                    if let sys = viewModel.systemStatus, !viewModel.pipelineReady {
                        pipelineReadinessWarning(sys)
                    }

                    // 1. Hero — status + actions + inline stats
                    heroCard

                    if !viewModel.plaudSyncNotices.isEmpty {
                        plaudNoticesCard(viewModel.plaudSyncNotices)
                    }

                    // 2. X-Ray Pipeline Flow — always visible diagnostic card
                    xrayPipelineCard

                    if let run = activeTraceRun {
                        traceRunCard(run)
                    }

                    // 3. Live console — terminal event stream
                    consoleCard

                    // 4. Stats — recordings + workflows unified
                    statsCard

                    if viewModel.supportsAdminEndpoints {
                        adminCard
                    }

                    if !viewModel.supportsAdminEndpoints || !viewModel.supportsSyncFailuresEndpoint || !viewModel.supportsUploadProcessEndpoint {
                        compatibilityCard
                    }

                          if viewModel.supportsSyncFailuresEndpoint,
                              let failures = viewModel.syncFailures,
                              failures.actionableCount > 0 {
                        failuresCard(failures)
                    }

                    // 5. Upload candidates
                    if !viewModel.uploadCandidates.isEmpty {
                        uploadCandidatesCard
                    }
                }
                .padding(.vertical, 8)
            }
            .navigationTitle("Sync")
            .refreshable { await viewModel.refresh() }
            .task { await viewModel.bootstrapIfNeeded() }
            .task { await xray.bootstrapIfNeeded() }
            .sheet(isPresented: $showBatchWorkflowSheet) {
                batchWorkflowSheet
            }
        }
    }

    // MARK: - 1. Hero Card

    private var heroCard: some View {
        VStack(spacing: 12) {
            // Stale data notice (transient refresh failure, last payload retained)
            if viewModel.hasStaleData {
                HStack(spacing: 6) {
                    Image(systemName: "clock.arrow.circlepath")
                        .foregroundStyle(.orange).font(.caption)
                    Text("Showing last known data — refresh may be delayed")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
                .padding(8)
                .background(.orange.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            // Error / message banner (inline, not a separate card)
            if let error = viewModel.error {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.yellow)
                    Text(error).font(.caption).lineLimit(2)
                    Spacer()
                }
                .padding(10)
                .background(.red.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            if let msg = viewModel.lastMessage, viewModel.error == nil {
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green).font(.caption)
                    Text(msg).font(.caption).foregroundStyle(.secondary)
                    Spacer()
                }
            }

            // Main status row
            HStack(alignment: .center, spacing: 12) {
                // Left: status icon + labels
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 8) {
                        if viewModel.isRunning {
                            PulsingDot()
                        } else {
                            Image(systemName: idleStatusIcon)
                                .font(.caption)
                                .foregroundStyle(idleStatusColor)
                        }
                        Text(viewModel.isRunning ? "Running" : idleStatusLabel)
                            .font(.headline)
                    }

                    if viewModel.isRunning {
                        Text(viewModel.currentStageTitle)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    } else if let status = viewModel.pipelineStatus {
                        // Idle subtitle — last run info
                        VStack(alignment: .leading, spacing: 2) {
                            if let elapsed = status.elapsedSeconds {
                                Text("Last run took \(TimeInterval(elapsed).durationFormatted)")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            if let updated = viewModel.lastUpdated {
                                Text(updated.relativeString)
                                    .font(.caption2)
                                    .foregroundStyle(.tertiary)
                            }
                        }
                    } else if let updated = viewModel.lastUpdated {
                        Text(updated.relativeString)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }

                Spacer()

                // Right: elapsed time when running, or completed count when idle
                if viewModel.isRunning,
                   let elapsed = viewModel.pipelineStatus?.elapsedSeconds {
                    VStack(alignment: .trailing, spacing: 2) {
                        Text(TimeInterval(elapsed).durationFormatted)
                            .font(.title2.weight(.semibold).monospacedDigit())
                        Text("elapsed")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                } else if !viewModel.isRunning, let db = viewModel.dbStats {
                    VStack(alignment: .trailing, spacing: 2) {
                        Text("\(db.completed)")
                            .font(.title2.weight(.semibold).monospacedDigit())
                            .foregroundStyle(.green)
                        Text("synced")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            // Inline DB counters (always visible)
            if let db = viewModel.dbStats {
                HStack(spacing: 0) {
                    counterChip("\(db.completed)", "done", .green)
                    counterChip("\(db.processing)", "running", .blue)
                    counterChip("\(db.pending)", "queued", .yellow)
                    if viewModel.hasFailureBreakdown {
                        if viewModel.shouldShowRetryableFailures {
                            counterChip("\(viewModel.actionableFailureCount)", "retryable", .orange)
                        }
                    } else if db.failed > 0 {
                        counterChip("\(db.failed)", "failed", .red)
                    }
                }
            }

            // Stuck warning (inline)
            if !viewModel.isRunning, let db = viewModel.dbStats, db.processing > 0 {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange).font(.caption)
                    Text("\(db.processing) stuck — pipeline finished but these didn't complete")
                        .font(.caption)
                    Spacer()
                    Button("Fix") { Task { await viewModel.resetStuck() } }
                        .font(.caption.weight(.semibold))
                        .buttonStyle(.bordered)
                        .tint(.orange)
                        .controlSize(.small)
                }
                .padding(8)
                .background(.orange.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            // Action buttons
            HStack(spacing: 10) {
                Button {
                    Task { await viewModel.runPipeline(stage: selectedStage) }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: viewModel.isRunning ? "hourglass" : selectedStage.icon)
                        Text(viewModel.isRunning ? "Syncing…" : selectedStage.displayName)
                    }
                    .font(.subheadline.weight(.semibold))
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.regular)
                .disabled(viewModel.isRunning)

                // Stage picker
                Menu {
                    ForEach(PipelineStage.allCases) { stage in
                        Button {
                            selectedStage = stage
                        } label: {
                            Label {
                                VStack(alignment: .leading) {
                                    Text(stage.displayName)
                                    Text(stage.description)
                                        .font(.caption2)
                                }
                            } icon: {
                                Image(systemName: stage.icon)
                            }
                        }
                    }
                } label: {
                    Image(systemName: "chevron.down.circle")
                        .font(.title3)
                }
                .buttonStyle(.bordered)

                Menu {
                    Button { Task { await viewModel.resetStuck() } } label: {
                        Label("Reset Stuck", systemImage: "arrow.counterclockwise")
                    }
                    Button { Task { await viewModel.refreshWorkflows() } } label: {
                        Label("Refresh Workflows", systemImage: "arrow.clockwise")
                    }
                    Button { Task { await viewModel.refreshCache() } } label: {
                        Label("Refresh Cache", systemImage: "arrow.triangle.2.circlepath")
                    }
                    Divider()
                    Button { showBatchWorkflowSheet = true } label: {
                        Label("Batch Workflows", systemImage: "brain")
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .font(.title3)
                }
                .buttonStyle(.bordered)
            }

            if selectedStage == .backfill {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "clock.arrow.trianglehead.counterclockwise.rotate.90")
                        .foregroundStyle(.orange)
                    Text("Full History paginates through Plaud page-by-page. Smart Sync stays conservative unless you explicitly choose this backfill mode.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
                .padding(10)
                .background(.orange.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }
        }
        .padding()
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .padding(.horizontal)
    }

    private var idleStatusIcon: String {
        let status = viewModel.pipelineStatus?.status.lowercased() ?? "idle"
        switch status {
        case "completed": return "checkmark.circle.fill"
        case "failed": return "xmark.circle.fill"
        default: return "circle.fill"
        }
    }

    private var idleStatusColor: Color {
        let status = viewModel.pipelineStatus?.status.lowercased() ?? "idle"
        switch status {
        case "completed": return .green
        case "failed": return .red
        default: return viewModel.pipelineStatus != nil ? .green : .secondary
        }
    }

    private var idleStatusLabel: String {
        let status = viewModel.pipelineStatus?.status.lowercased() ?? "idle"
        switch status {
        case "completed": return "Completed"
        case "failed": return "Failed"
        default: return "Idle"
        }
    }

    private var activeTraceRun: TraceRun? {
        if let runId = viewModel.activeTraceRunId ?? viewModel.pipelineStatus?.runId,
           let match = xray.runs.first(where: { $0.runId == runId }) {
            return match
        }
        return xray.latestRun
    }

    private func traceRunCard(_ run: TraceRun) -> some View {
        let spans = xray.spans.filter { $0.runId == run.runId }
        let providers = Array(Set(spans.compactMap(\.provider))).sorted()
        let models = Array(Set(spans.compactMap(\.model))).sorted()
        let cost = spans.reduce(0.0) { $0 + ($1.costUsd ?? 0) }

        return VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: run.status.lowercased() == "running" ? "waveform.path.ecg" : "point.3.connected.trianglepath.dotted")
                    .foregroundStyle(run.status.lowercased() == "running" ? .green : .purple)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Execution Trace")
                        .font(.subheadline.weight(.semibold))
                    Text(run.runId)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Text(run.status.capitalized)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(run.status.lowercased() == "failed" ? .red : .green)
            }

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                miniTraceMetric("Spans", "\(spans.count)", .purple)
                miniTraceMetric("Cost", formatSyncCost(cost), cost > 0 ? .orange : .secondary)
                miniTraceMetric("Providers", providers.isEmpty ? "—" : providers.joined(separator: ", "), .blue)
                miniTraceMetric("Models", models.first ?? "—", .green)
            }

            ForEach(spans.prefix(4)) { span in
                HStack(spacing: 6) {
                    Circle()
                        .fill(span.status.lowercased() == "failed" ? .red : (span.status.lowercased() == "running" ? .green : .secondary))
                        .frame(width: 6, height: 6)
                    Text(span.stage ?? span.source ?? "span")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(sourceColor(span.source ?? span.provider ?? ""))
                        .frame(width: 58, alignment: .leading)
                    Text(span.message ?? span.operation)
                        .font(.caption2)
                        .lineLimit(1)
                    Spacer()
                    if let duration = span.durationMs {
                        Text(duration < 1000 ? String(format: "%.0fms", duration) : TimeInterval(duration / 1000).durationFormatted)
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(7)
                .background(.thinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }
        }
        .padding()
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }

    private func miniTraceMetric(_ title: String, _ value: String, _ color: Color) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption.weight(.semibold))
                .foregroundStyle(color)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(8)
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private func formatSyncCost(_ usd: Double) -> String {
        if usd == 0 { return "$0.00" }
        if usd < 0.01 { return String(format: "$%.4f", usd) }
        if usd < 1 { return String(format: "$%.3f", usd) }
        return String(format: "$%.2f", usd)
    }

    private func counterChip(_ value: String, _ label: String, _ color: Color) -> some View {
        VStack(spacing: 1) {
            Text(value)
                .font(.subheadline.weight(.bold).monospacedDigit())
                .foregroundStyle(color)
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
    }

    // MARK: - 2. X-Ray Pipeline Flow

    private struct VisualStage: Identifiable {
        var id: String { name }
        let name: String
        let displayName: String
        let icon: String
        let description: String
        let defaultColor: Color
        var phase: PipelinePhase?
    }

    private enum StageStatus {
        case idle
        case pending
        case running
        case completed
        case failed
    }

    private struct RotatingRing: View {
        @State private var isAnimating = false
        let color: Color
        
        var body: some View {
            Circle()
                .stroke(color, style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round, dash: [4, 3]))
                .frame(width: 34, height: 34)
                .rotationEffect(.degrees(isAnimating ? 360 : 0))
                .animation(.linear(duration: 4).repeatForever(autoreverses: false), value: isAnimating)
                .onAppear { isAnimating = true }
        }
    }

    private func buildVisualStages(from status: PipelineStatus?) -> [VisualStage] {
        let phases = status?.phases ?? []
        
        let ingestPhase = phases.first { $0.name.localizedCaseInsensitiveContains("ingest") }
        let processPhase = phases.first { 
            $0.name.localizedCaseInsensitiveContains("process") || 
            $0.name.localizedCaseInsensitiveContains("workflow") || 
            $0.name.localizedCaseInsensitiveContains("ai") 
        }
        let indexPhase = phases.first { 
            $0.name.localizedCaseInsensitiveContains("index") || 
            $0.name.localizedCaseInsensitiveContains("embed") || 
            $0.name.localizedCaseInsensitiveContains("qdrant") 
        }
        let graphPhase = phases.first { $0.name.localizedCaseInsensitiveContains("graph") }
        let syncPhase = phases.first { 
            $0.name.localizedCaseInsensitiveContains("sync") || 
            $0.name.localizedCaseInsensitiveContains("notion") 
        }
        
        return [
            VisualStage(
                name: "ingest",
                displayName: "Plaud Ingest",
                icon: "arrow.down.doc",
                description: "Fetch and download recordings from Plaud Cloud API",
                defaultColor: .green,
                phase: ingestPhase
            ),
            VisualStage(
                name: "process",
                displayName: "AI Processing",
                icon: "brain",
                description: "Run transcription, categorization and summarization workflows",
                defaultColor: .purple,
                phase: processPhase
            ),
            VisualStage(
                name: "index",
                displayName: "Vector Indexing",
                icon: "magnifyingglass",
                description: "Generate embeddings and upsert vectors into Qdrant",
                defaultColor: .orange,
                phase: indexPhase
            ),
            VisualStage(
                name: "graph",
                displayName: "Knowledge Graph",
                icon: "point.3.connected.trianglepath.dotted",
                description: "Extract entity connections and link semantic nodes",
                defaultColor: .indigo,
                phase: graphPhase
            ),
            VisualStage(
                name: "sync",
                displayName: "Data Sync & Finish",
                icon: "arrow.triangle.2.circlepath",
                description: "Sync summaries with Notion database and wrap up",
                defaultColor: .blue,
                phase: syncPhase
            )
        ]
    }

    private func stageStatus(for stage: VisualStage, overallRunning: Bool) -> StageStatus {
        guard let phase = stage.phase else {
            if overallRunning {
                if let currentPhase = viewModel.pipelineStatus?.currentPhase,
                   currentPhase.localizedCaseInsensitiveContains(stage.name) {
                    return .running
                }
                
                let order = ["ingest", "process", "index", "graph", "sync"]
                if let currentPhase = viewModel.pipelineStatus?.currentPhase?.lowercased() {
                    let currentIdx = order.firstIndex(of: order.first(where: { currentPhase.contains($0) }) ?? "") ?? 0
                    let stageIdx = order.firstIndex(of: stage.name) ?? 0
                    if stageIdx < currentIdx {
                        return .completed
                    } else if stageIdx == currentIdx {
                        return .running
                    } else {
                        return .pending
                    }
                }
                return .pending
            }
            return .idle
        }
        
        switch phase.status.lowercased() {
        case "running": return .running
        case "completed", "success": return .completed
        case "failed", "error": return .failed
        case "pending", "queued": return .pending
        default: return .idle
        }
    }

    private func stageColor(for status: StageStatus, defaultColor: Color) -> Color {
        switch status {
        case .idle: return .secondary.opacity(0.6)
        case .pending: return .orange.opacity(0.7)
        case .running: return defaultColor
        case .completed: return .green
        case .failed: return .red
        }
    }
    
    private func statusBadge(for status: StageStatus) -> some View {
        let text: String
        let color: Color
        switch status {
        case .idle:
            text = "IDLE"
            color = .secondary
        case .pending:
            text = "PENDING"
            color = .orange
        case .running:
            text = "ACTIVE"
            color = .blue
        case .completed:
            text = "DONE"
            color = .green
        case .failed:
            text = "FAILED"
            color = .red
        }
        
        return Text(text)
            .font(.system(size: 8, weight: .bold))
            .padding(.horizontal, 4)
            .padding(.vertical, 2)
            .background(color.opacity(0.12))
            .foregroundStyle(color)
            .clipShape(RoundedRectangle(cornerRadius: 3))
    }

    private var xrayPipelineCard: some View {
        let status = viewModel.pipelineStatus
        let stages = buildVisualStages(from: status)
        let isAnyRunning = viewModel.isRunning || (status?.running ?? false)
        
        return VStack(alignment: .leading, spacing: 14) {
            // Header
            HStack(spacing: 8) {
                Image(systemName: "point.3.connected.trianglepath.dotted")
                    .foregroundStyle(isAnyRunning ? .blue : .secondary)
                    .font(.headline)
                
                Text("X-Ray Pipeline Flow")
                    .font(.subheadline.weight(.semibold))
                
                if isAnyRunning {
                    PulsingDot()
                }
                
                Spacer()
                
                if let runId = status?.runId {
                    Text("Run \(String(runId.prefix(8)))")
                        .font(.caption2.monospaced())
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.secondary.opacity(0.1))
                        .clipShape(Capsule())
                }
                
                Button {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        showPhaseDetail.toggle()
                    }
                } label: {
                    Image(systemName: showPhaseDetail ? "chevron.up" : "chevron.down")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 14)
            .padding(.top, 14)
            
            if showPhaseDetail {
                Divider()
                    .padding(.horizontal, 14)
                
                // Stages layout
                ZStack(alignment: .topLeading) {
                    // Vertical connection track line
                    Rectangle()
                        .fill(Color.secondary.opacity(0.15))
                        .frame(width: 2)
                        .padding(.leading, 30) // Center of the 34pt column plus padding (14 + 16 = 30)
                        .padding(.vertical, 24)
                    
                    VStack(alignment: .leading, spacing: 20) {
                        ForEach(stages) { stage in
                            stageRow(stage, overallRunning: isAnyRunning)
                        }
                    }
                    .padding(.horizontal, 14)
                    .padding(.bottom, 14)
                }
                
                // Progress entries/metadata if running
                if !viewModel.progressEntries.isEmpty {
                    Divider()
                        .padding(.horizontal, 14)
                    
                    LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                        ForEach(viewModel.progressEntries, id: \.key) { entry in
                            VStack(alignment: .leading, spacing: 2) {
                                Text(entry.key)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                Text(entry.value)
                                    .font(.caption.weight(.semibold))
                                    .monospacedDigit()
                                    .lineLimit(1)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    .padding(.horizontal, 14)
                    .padding(.bottom, 14)
                }
            }
        }
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .padding(.horizontal)
    }

    private func stageRow(_ stage: VisualStage, overallRunning: Bool) -> some View {
        let status = stageStatus(for: stage, overallRunning: overallRunning)
        let color = stageColor(for: status, defaultColor: stage.defaultColor)
        
        return HStack(alignment: .top, spacing: 14) {
            // Left: Circle indicator with icon
            ZStack {
                Circle()
                    .fill(status == .completed ? color : Color.clear)
                    .frame(width: 28, height: 28)
                
                Circle()
                    .strokeBorder(color, lineWidth: status == .idle ? 1 : 2)
                    .frame(width: 28, height: 28)
                    .shadow(color: status == .running ? color.opacity(0.4) : Color.clear, radius: 4)
                
                if status == .running {
                    RotatingRing(color: color)
                }
                
                Image(systemName: status == .failed ? "exclamationmark.triangle.fill" : stage.icon)
                    .font(.caption2)
                    .foregroundStyle(status == .completed ? .white : color)
            }
            .frame(width: 34, height: 34)
            
            // Right: Content details
            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .center, spacing: 8) {
                    Text(stage.displayName)
                        .font(.subheadline.weight(.semibold))
                    
                    statusBadge(for: status)
                    
                    Spacer()
                    
                    if let elapsed = stage.phase?.elapsedSeconds, elapsed > 0 {
                        Text(TimeInterval(elapsed).durationFormatted)
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
                
                Text(stage.description)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                
                // If stage has items, show progress
                if let phase = stage.phase, phase.totalItems > 0 {
                    VStack(alignment: .leading, spacing: 4) {
                        ProgressView(value: Double(phase.completedItems), total: Double(max(phase.totalItems, 1)))
                            .tint(color)
                            .scaleEffect(x: 1, y: 0.75, anchor: .center)
                        
                        Text("\(phase.completedItems) of \(phase.totalItems) records")
                            .font(.system(size: 10, weight: .medium, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                    .padding(.top, 2)
                }
                
                // If running, show current step/file
                if status == .running, let phase = stage.phase {
                    VStack(alignment: .leading, spacing: 2) {
                        if !phase.currentStep.isEmpty {
                            HStack(spacing: 4) {
                                Text("Step:")
                                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                                    .foregroundStyle(color.opacity(0.8))
                                Text(phase.currentStep)
                                    .font(.system(size: 9, design: .monospaced))
                                    .foregroundStyle(.white.opacity(0.9))
                            }
                        }
                        if !phase.currentItem.isEmpty {
                            HStack(spacing: 4) {
                                Text("Item:")
                                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                                    .foregroundStyle(color.opacity(0.8))
                                Text(phase.currentItem)
                                    .font(.system(size: 9, design: .monospaced))
                                    .foregroundStyle(.white.opacity(0.7))
                            }
                        }
                    }
                    .padding(6)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.black.opacity(0.8))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .padding(.top, 4)
                }
                
                // If failed, show error detail
                if status == .failed, let phase = stage.phase, !phase.error.isEmpty {
                    HStack(alignment: .top, spacing: 6) {
                        Image(systemName: "exclamationmark.octagon.fill")
                            .font(.caption2)
                            .foregroundStyle(.red)
                            .padding(.top, 1)
                        Text(phase.error)
                            .font(.caption2)
                            .foregroundStyle(.red)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.red.opacity(0.08))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .padding(.top, 4)
                }
            }
        }
    }

    private func metaLabel(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(title).font(.caption2).foregroundStyle(.tertiary)
            Text(value).font(.caption2.weight(.medium)).foregroundStyle(.secondary)
        }
    }

    // MARK: - 3. Live Console

    private var filteredConsoleEvents: [XRayEvent] {
        guard let filter = selectedConsoleSourceFilter else {
            return xray.syncEvents
        }
        
        switch filter {
        case "ingest":
            return xray.events.filter { $0.source.lowercased() == "ingest" }
        case "process":
            return xray.events.filter { ["gemini", "local", "openai", "claude"].contains($0.source.lowercased()) }
        case "index":
            return xray.events.filter { ["embed", "qdrant"].contains($0.source.lowercased()) }
        case "graph":
            return xray.events.filter { $0.source.lowercased() == "graph" }
        case "sync":
            return xray.events.filter { 
                $0.source.lowercased() == "sync" || 
                $0.op.localizedCaseInsensitiveContains("notion") || 
                $0.message.localizedCaseInsensitiveContains("notion") ||
                $0.detail?.localizedCaseInsensitiveContains("notion") == true
            }
        default:
            return xray.syncEvents
        }
    }

    private func filterButton(_ title: String, filter: String?) -> some View {
        let isSelected = selectedConsoleSourceFilter == filter
        return Button {
            withAnimation(.easeInOut(duration: 0.15)) {
                selectedConsoleSourceFilter = filter
            }
        } label: {
            Text(title)
                .font(.system(size: 10, weight: .semibold))
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(isSelected ? Color.blue : Color.white.opacity(0.08))
                .foregroundStyle(isSelected ? .white : .secondary)
                .clipShape(Capsule())
        }
        .buttonStyle(.plain)
    }

    private var consoleCard: some View {
        let consoleEvents = Array(filteredConsoleEvents.prefix(consoleEventLimit))
        let isActive = viewModel.isRunning

        return VStack(alignment: .leading, spacing: 0) {
            // Header
            Button {
                withAnimation(.easeInOut(duration: 0.25)) { showConsole.toggle() }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "terminal")
                        .foregroundStyle(isActive ? .green : .secondary)

                    Text("Console Logs")
                        .font(.subheadline.weight(.semibold))

                    if isActive { PulsingDot() }

                    Spacer()

                    if !filteredConsoleEvents.isEmpty {
                        Text("\(filteredConsoleEvents.count)")
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }

                    Image(systemName: showConsole ? "chevron.up" : "chevron.down")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
            }
            .buttonStyle(.plain)

            if showConsole {
                // Filters Toolbar
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        filterButton("All Logs", filter: nil)
                        filterButton("Ingest", filter: "ingest")
                        filterButton("AI Process", filter: "process")
                        filterButton("Qdrant Index", filter: "index")
                        filterButton("Graph RAG", filter: "graph")
                        filterButton("Sync", filter: "sync")
                    }
                    .padding(.horizontal, 14)
                }
                .padding(.bottom, 8)

                // Toolbar
                HStack(spacing: 10) {
                    Toggle(isOn: $consoleAutoScroll) {
                        Label("Auto-scroll", systemImage: "arrow.down.to.line")
                            .font(.caption2)
                    }
                    .toggleStyle(.button)
                    .controlSize(.mini)

                    Spacer()

                    Picker("Limit", selection: $consoleEventLimit) {
                        Text("30").tag(30)
                        Text("60").tag(60)
                        Text("All").tag(200)
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 130)
                }
                .padding(.horizontal, 12)
                .padding(.bottom, 6)

                // Terminal output
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 0) {
                            ForEach(Array(consoleEvents.reversed().enumerated()), id: \.element.id) { index, event in
                                consoleRow(event, index: index)
                                    .id(event.id)
                            }
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                    }
                    .frame(height: 240)
                    .background(.black.opacity(0.88))
                    .onChange(of: filteredConsoleEvents.first?.seq) { _, _ in
                        if consoleAutoScroll, let latest = consoleEvents.first {
                            withAnimation(.easeOut(duration: 0.15)) {
                                proxy.scrollTo(latest.id, anchor: .bottom)
                            }
                        }
                    }
                }

                // Summary bar
                if isActive, let db = viewModel.dbStats {
                    HStack(spacing: 14) {
                        inlineStat(.blue, "\(db.processing) run")
                        inlineStat(.yellow, "\(db.pending) queued")
                        inlineStat(.green, "\(db.completed) done")
                        Spacer()
                        if let elapsed = viewModel.pipelineStatus?.elapsedSeconds {
                            Text(TimeInterval(elapsed).durationFormatted)
                                .monospacedDigit()
                        }
                    }
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                }
            }
        }
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
        .onAppear {
            if viewModel.isRunning { showConsole = true }
        }
    }

    private func consoleRow(_ event: XRayEvent, index: Int) -> some View {
        HStack(alignment: .top, spacing: 5) {
            Text(event.date.timeString)
                .foregroundStyle(.white.opacity(0.35))
                .frame(width: 50, alignment: .leading)

            Text(event.source.uppercased())
                .foregroundStyle(sourceColor(event.source))
                .fontWeight(.bold)
                .frame(width: 48, alignment: .leading)

            VStack(alignment: .leading, spacing: 1) {
                Text(event.message)
                    .foregroundStyle(event.level == "error" ? .red : .white.opacity(0.88))
                    .lineLimit(2)

                if let detail = event.detail, !detail.isEmpty {
                    Text(detail.prefix(100))
                        .foregroundStyle(.white.opacity(0.35))
                        .lineLimit(1)
                }
            }

            Spacer(minLength: 2)

            if let ms = event.durationMs {
                Text(formattedDuration(ms))
                    .foregroundStyle(durationColor(ms))
                    .frame(width: 44, alignment: .trailing)
            }
        }
        .font(.caption2.monospaced())
        .padding(.vertical, 2)
        .background(index % 2 == 0 ? Color.white.opacity(0.03) : Color.clear)
    }

    private func inlineStat(_ color: Color, _ text: String) -> some View {
        HStack(spacing: 3) {
            Circle().fill(color).frame(width: 5, height: 5)
            Text(text)
        }
    }

    // MARK: - 4. Stats Card

    private var statsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Recordings
            if let db = viewModel.dbStats {
                VStack(alignment: .leading, spacing: 6) {
                    Label("Recordings", systemImage: "cylinder")
                        .font(.subheadline.weight(.semibold))

                    LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 3), spacing: 6) {
                        miniStat("\(db.total)", "total")
                        miniStat("\(db.completed)", "done", .green)
                        miniStat("\(db.pending)", "pending", .yellow)
                        miniStat("\(db.processing)", "running", .blue)
                        if viewModel.hasFailureBreakdown {
                            if viewModel.shouldShowRetryableFailures {
                                miniStat("\(viewModel.actionableFailureCount)", "retryable", .orange)
                            }
                        } else if db.failed > 0 {
                            miniStat("\(db.failed)", "failed", .red)
                        }
                        miniStat("\(db.noTranscript)", "no transcript", .orange)
                    }
                }
            }

            // Workflows
            if let wf = viewModel.workflowStats {
                Divider()

                VStack(alignment: .leading, spacing: 6) {
                    Label("Plaud AI Workflows", systemImage: "brain")
                        .font(.subheadline.weight(.semibold))

                    LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 3), spacing: 6) {
                        miniStat("\(wf.recentRecordings ?? 0)", "recordings")
                        miniStat("\(wf.workflowSuccess ?? 0)", "done", .green)
                        miniStat("\(wf.workflowPending ?? 0)", "pending", .yellow)
                        miniStat("\(wf.workflowFailed ?? 0)", "failed", .red)
                        miniStat("\(wf.withAiSummary ?? 0)", "summaries", .purple)
                    }
                }
            }
        }
        .padding()
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }

    private var adminCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Admin Controls", systemImage: "wrench.and.screwdriver")
                .font(.subheadline.weight(.semibold))

            // Qdrant detail inline when available
            if let qdrant = viewModel.systemStatus?.qdrant {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 8) {
                        Image(systemName: qdrant.ok ? "checkmark.circle.fill" : "xmark.circle.fill")
                            .foregroundStyle(qdrant.ok ? .green : .red)
                            .font(.caption)
                        Text("Qdrant")
                            .font(.caption.weight(.medium))
                        if let collections = qdrant.collections {
                            Text("· \(collections) collection\(collections == 1 ? "" : "s")")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        if let url = qdrant.url {
                            Spacer()
                            Text(url)
                                .font(.caption2.monospaced())
                                .foregroundStyle(.tertiary)
                                .lineLimit(1)
                        }
                    }

                    if let error = qdrant.error, !error.isEmpty {
                        Text(error)
                            .font(.caption2)
                            .foregroundStyle(.red)
                            .lineLimit(3)
                    }
                }
                .padding(8)
                .background(qdrant.ok ? .green.opacity(0.08) : .red.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            HStack(spacing: 8) {
                adminButton("Pi Stack Status") {
                    Task { await viewModel.runStackAction("status") }
                }
                adminButton("Ensure Public") {
                    Task { await viewModel.runStackAction("ensure-public") }
                }
                adminButton("Restart Public") {
                    Task { await viewModel.runStackAction("restart-public") }
                }
            }

            HStack(spacing: 8) {
                adminButton(viewModel.isCreatingBackup ? "Creating…" : "Create Backup") {
                    Task { await viewModel.createBackup() }
                }
                .disabled(viewModel.isCreatingBackup)

                if let latestBackup = viewModel.backups.first {
                    adminButton("Download Latest") {
                        Task { await viewModel.downloadBackup(latestBackup) }
                    }
                }
            }

            if let backupURL = viewModel.downloadedBackupURL {
                ShareLink(item: backupURL) {
                    Label("Share Downloaded Backup", systemImage: "square.and.arrow.up")
                        .font(.caption.weight(.semibold))
                }
            }

            if let latestBackup = viewModel.backups.first {
                Text("Latest backup: \(latestBackup.filename) · \(ByteCountFormatter.string(fromByteCount: Int64(latestBackup.sizeBytes), countStyle: .file))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if let stackControl = viewModel.stackControl {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Label(stackControl.action, systemImage: stackStatusIcon(stackControl.status))
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(stackStatusColor(stackControl.status))
                        Spacer()
                        Text(stackControl.status.capitalized)
                            .font(.caption2)
                            .foregroundStyle(stackStatusColor(stackControl.status))
                    }

                    if !stackControl.message.isEmpty {
                        Text(stackControl.message)
                            .font(.caption)
                            .foregroundStyle(stackStatusColor(stackControl.status))
                    }

                    if stackControl.action == "status",
                       stackControl.status.lowercased() != "ok",
                       viewModel.systemStatus?.qdrant?.ok == true {
                        Text("Qdrant health is currently green, so this stack result reflects the Pi's managed stack status rather than the active API connection.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }

                    if let publicURL = stackControl.publicURL, !publicURL.isEmpty {
                        Text(publicURL)
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }

                    Text(stackControl.output)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(8)
                }
                .padding(10)
                .background(stackStatusColor(stackControl.status).opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 10))
            }
        }
        .padding()
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }

    private func failuresCard(_ failures: SyncFailureSummary) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("Retryable Sync Issues", systemImage: "exclamationmark.triangle")
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Button("Retry") { Task { await viewModel.resetStuck() } }
                    .font(.caption.weight(.semibold))
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }

            miniStat("\(failures.actionableCount)", "retryable", .orange)

            Text("These items are still eligible for retry.")
                .font(.caption2)
                .foregroundStyle(.secondary)

            ForEach(failures.actionable.prefix(4)) { item in
                failureItemRow(item, tint: .orange)
            }
        }
        .padding()
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }

    private func failureItemRow(_ item: SyncFailureItem, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Circle()
                    .fill(tint)
                    .frame(width: 6, height: 6)
                Text(item.title ?? item.recordingId ?? "Recording")
                    .font(.caption.weight(.semibold))
                    .lineLimit(1)
            }

            if let reason = item.reason, !reason.isEmpty {
                Text(reason)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
            }

            if let source = item.source, !source.isEmpty {
                Text(source.capitalized)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(8)
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private func miniStat(_ value: String, _ label: String, _ color: Color = .primary) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.title3.weight(.bold).monospacedDigit())
                .foregroundStyle(color)
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 6)
    }

    private func stackStatusColor(_ status: String) -> Color {
        switch status.lowercased() {
        case "ok", "healthy", "success":
            return .green
        case "warn", "warning":
            return .orange
        case "failed", "error", "down":
            return .red
        default:
            return .secondary
        }
    }

    private func stackStatusIcon(_ status: String) -> String {
        switch status.lowercased() {
        case "ok", "healthy", "success":
            return "checkmark.circle.fill"
        case "warn", "warning":
            return "exclamationmark.triangle.fill"
        case "failed", "error", "down":
            return "xmark.circle.fill"
        default:
            return "info.circle.fill"
        }
    }

    // MARK: - 5. Upload Candidates

    private var uploadCandidatesCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("Upload Candidates", systemImage: "icloud.and.arrow.up")
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Button(viewModel.isUploadingCandidates ? "Uploading…" : "Upload & Process All") {
                    Task {
                        await viewModel.uploadAllCandidates(
                            templateId: uploadTemplateId.isEmpty ? nil : uploadTemplateId,
                            model: uploadModel
                        )
                    }
                }
                .font(.caption.weight(.semibold))
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(viewModel.isUploadingCandidates || !viewModel.supportsUploadProcessEndpoint)
            }

            Text("\(viewModel.uploadCandidates.count) recording\(viewModel.uploadCandidates.count == 1 ? "" : "s") eligible for Plaud cloud upload")
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack(spacing: 8) {
                TextField("Template ID (optional)", text: $uploadTemplateId)
                    .platformTextEntryBehavior()
                Picker("AI Provider", selection: $uploadModel) {
                    ForEach(workflowProviders, id: \.self) { provider in
                        Text(provider.capitalized).tag(provider)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 180)
            }
            .font(.caption)

            Text("Uploads default to OpenAI. OpenAI and Local LLM fallback can be configured in settings.")
                .font(.caption2)
                .foregroundStyle(.secondary)

            ForEach(viewModel.uploadCandidates.prefix(10)) { candidate in
                HStack(spacing: 8) {
                    Image(systemName: "waveform")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(candidate.filename)
                            .font(.caption.weight(.medium))
                            .lineLimit(1)
                        Text(String(format: "%.1f MB · %@", candidate.sizeMb, candidate.format.uppercased()))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Image(systemName: candidate.inCloud ? "checkmark.icloud" : "icloud.slash")
                        .font(.caption2)
                        .foregroundStyle(candidate.inCloud ? .green : .orange)
                }
            }

            if viewModel.uploadCandidates.count > 10 {
                Text("+ \(viewModel.uploadCandidates.count - 10) more")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            if !viewModel.supportsUploadProcessEndpoint {
                Text("Upload and process needs the newer backend API. Upload candidates still work on this server build.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }

    // MARK: - Batch Workflow Sheet

    private var batchWorkflowSheet: some View {
        NavigationStack {
            Form {
                Section("Parameters") {
                    Stepper("Days Back: \(batchDaysBack)", value: $batchDaysBack, in: 1...90)
                    Stepper("Limit: \(batchLimit)", value: $batchLimit, in: 1...50)
                    TextField("Template ID (optional)", text: $batchTemplateId)
                        .platformTextEntryBehavior()
                    Picker("AI Provider", selection: $batchWorkflowModel) {
                        ForEach(workflowProviders, id: \.self) { provider in
                            Text(provider.capitalized).tag(provider)
                        }
                    }
                    .pickerStyle(.segmented)
                    Text("Batch workflows default to OpenAI. OpenAI and Local LLM fallback can be configured in settings.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section {
                    Button {
                        Task {
                            await viewModel.submitWorkflows(
                                daysBack: batchDaysBack,
                                limit: batchLimit,
                                templateId: batchTemplateId.isEmpty ? nil : batchTemplateId,
                                model: batchWorkflowModel
                            )
                            showBatchWorkflowSheet = false
                        }
                    } label: {
                        HStack {
                            Spacer()
                            Label("Submit Batch", systemImage: "brain")
                                .font(.headline)
                            Spacer()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
            .navigationTitle("Batch Workflows")
            .platformNavigationBarTitleDisplayModeInline()
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { showBatchWorkflowSheet = false }
                }
            }
        }
        .presentationDetents([.medium])
    }

    // MARK: - Shared Helpers

    private func pipelineReadinessWarning(_ sys: SystemStatus) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                Text("Pipeline Not Ready")
                    .font(.subheadline.weight(.semibold))
            }

            if sys.database?.ok != true {
                warningLine("Database is down", detail: sys.database?.error)
            }
            if sys.qdrant?.ok != true {
                warningLine("Qdrant is not connected", detail: sys.qdrant?.error)
            }
            if sys.gemini?.isUp != true && sys.localLLM?.isUp != true && sys.openai?.isOperational != true {
                warningLine("No AI provider available", detail: "Ensure OpenAI is enabled, Gemini is configured, or Local LLM is ready.")
            }
        }
        .padding()
        .background(.orange.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }

    private var compatibilityCard: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "arrow.triangle.branch")
                    .foregroundStyle(.orange)
                Text("Backend Is Older Than The App")
                    .font(.subheadline.weight(.semibold))
            }

            if !viewModel.supportsAdminEndpoints {
                Text("Admin controls and backups are hidden because this server build does not expose `/api/v1/admin/...` yet.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if !viewModel.supportsSyncFailuresEndpoint {
                Text("Sync failure summaries are hidden because `/api/v1/sync/failures` is not available on this backend.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if !viewModel.supportsUploadProcessEndpoint {
                Text("One-tap upload and process is disabled because `/api/v1/sync/upload/process` is not available on this backend.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if !viewModel.supportsBackfillStage {
                Text("Full history Plaud backfill is not available on this server yet. Update the backend Plaud reliability changes first.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .background(.orange.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }

    private func plaudNoticesCard(_ notices: [PlaudSyncNotice]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: "exclamationmark.bubble.fill")
                    .foregroundStyle(.orange)
                Text("Plaud Sync Notices")
                    .font(.subheadline.weight(.semibold))
                Spacer()
            }

            ForEach(notices) { notice in
                plaudNoticeLine(notice)
            }
        }
        .padding()
        .background(.orange.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }

    private func plaudNoticeLine(_ notice: PlaudSyncNotice) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: noticeIcon(for: notice.kind))
                .foregroundStyle(noticeColor(for: notice.kind))
                .font(.caption)
            VStack(alignment: .leading, spacing: 2) {
                Text(notice.title)
                    .font(.caption.weight(.semibold))
                Text(notice.message)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
        }
    }

    private func warningLine(_ message: String, detail: String?) -> some View {
        HStack(spacing: 6) {
            Image(systemName: "xmark.circle.fill")
                .foregroundStyle(.red)
                .font(.caption)
            VStack(alignment: .leading, spacing: 1) {
                Text(message)
                    .font(.caption.weight(.medium))
                if let detail, !detail.isEmpty {
                    Text(detail)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }
        }
    }

    private func noticeIcon(for kind: PlaudSyncNotice.Kind) -> String {
        switch kind {
        case .repeatedPage:
            "arrow.trianglehead.branch"
        case .partialProgress:
            "tray.full"
        case .rateLimited:
            "tortoise"
        case .warning:
            "exclamationmark.triangle"
        }
    }

    private func noticeColor(for kind: PlaudSyncNotice.Kind) -> Color {
        switch kind {
        case .repeatedPage:
            .orange
        case .partialProgress:
            .blue
        case .rateLimited:
            .yellow
        case .warning:
            .red
        }
    }

    private func sourceColor(_ source: String) -> Color {
        switch source {
        case "pipeline": return .cyan
        case "ingest": return .green
        case "gemini": return .yellow
        case "embed": return .purple
        case "qdrant": return .orange
        case "sync": return .blue
        case "data": return .mint
        case "graph": return .indigo
        default: return .white.opacity(0.5)
        }
    }

    private func adminButton(_ title: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.caption.weight(.semibold))
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .disabled(viewModel.isRunningStackAction)
    }

    private func phaseIcon(for status: String) -> String {
        switch status.lowercased() {
        case "running": return "arrow.triangle.2.circlepath.circle.fill"
        case "completed": return "checkmark.circle.fill"
        case "failed": return "xmark.circle.fill"
        default: return "circle"
        }
    }

    private func phaseColor(for status: String) -> Color {
        switch status.lowercased() {
        case "running": return .green
        case "completed": return .blue
        case "failed": return .red
        default: return .secondary
        }
    }

    private func formattedDuration(_ ms: Double) -> String {
        if ms < 1000 { return String(format: "%.0fms", ms) }
        if ms < 60000 { return String(format: "%.1fs", ms / 1000) }
        return String(format: "%.0fm", ms / 60000)
    }

    private func durationColor(_ ms: Double) -> Color {
        switch ms {
        case ..<500: return .green
        case ..<3000: return .yellow
        case ..<10000: return .orange
        default: return .red
        }
    }
}

// MARK: - Pulsing Dot

private struct PulsingDot: View {
    @State private var isAnimating = false

    var body: some View {
        Circle()
            .fill(.green)
            .frame(width: 8, height: 8)
            .scaleEffect(isAnimating ? 1.3 : 1.0)
            .opacity(isAnimating ? 0.6 : 1.0)
            .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: isAnimating)
            .onAppear { isAnimating = true }
    }
}
