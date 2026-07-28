import Charts
import SwiftUI
import UniformTypeIdentifiers

/// Side-by-side acoustic benchmark of two recordings.
struct DualBenchmarkView: View {
    @Environment(APIClient.self) private var api
    @State private var model = DualBenchmarkViewModel()
    /// Safe as an optional: `.sheet(item:)` hands the value to its content
    /// closure, so unlike `.fileImporter` it cannot be cleared before use.
    @State private var pickingSlot: BenchmarkSlot?
    @State private var isImporting = false
    @State private var importingSlot: BenchmarkSlot = .a

    private let columnA = Color.red
    private let columnB = Color.green

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                fileCards

                // Transcripts stand on their own: the corpus already holds them,
                // and comparing text should not require analysing two WAVs
                // first. Only the acoustic sections depend on both reports.
                transcriptViewer

                if let comparison = model.comparison {
                    executiveOverview(comparison)
                    waveformChart(comparison)
                    loudnessChart(comparison)
                    spectralChart(comparison)
                    metricTable
                } else {
                    emptyState
                }
            }
            .padding(.vertical)
        }
        .navigationTitle("Audio Benchmark")
        .navigationBarTitleDisplayMode(.inline)
        .fileImporter(
            isPresented: $isImporting,
            allowedContentTypes: [.wav, .aiff, .mp3, .mpeg4Audio, .audio],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first { model.analyze(url: url, into: importingSlot, api: api) }
            case .failure(let error):
                model.errorMessage = error.localizedDescription
            }
        }
        .sheet(item: $pickingSlot) { slot in
            transcriptPicker(for: slot)
        }
        .task { await model.loadLibrary(api: api) }
        .alert("Analysis Failed", isPresented: Binding(
            get: { model.errorMessage != nil },
            set: { if !$0 { model.errorMessage = nil } }
        )) {
            Button("OK", role: .cancel) { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
        }
    }

    // MARK: - File selection

    private var fileCards: some View {
        HStack(alignment: .top, spacing: 12) {
            fileCard(for: .a, tint: columnA)
            fileCard(for: .b, tint: columnB)
        }
        .padding(.horizontal)
    }

    private func fileCard(for slot: BenchmarkSlot, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(slot.label)
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(tint)

            if let report = model.report(for: slot) {
                Text(report.fileName)
                    .font(.footnote)
                    .fontWeight(.medium)
                    .lineLimit(2)
                Text("\(report.durationFormatted) · \(DualBenchmarkViewModel.number(report.fileSizeMB, 1)) MB")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text("\(Int(report.sampleRateHz)) Hz · \(report.bitDepthEncoding)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)

                Button("Replace") {
                    importingSlot = slot
                    isImporting = true
                }
                    .font(.caption2)
                    .buttonStyle(.bordered)
            } else if model.isAnalyzing(slot) {
                ProgressView(value: model.progress(for: slot))
                    .progressViewStyle(.linear)
                Text("Analyzing… \(Int(model.progress(for: slot) * 100))%")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } else {
                Button {
                    importingSlot = slot
                    isImporting = true
                } label: {
                    Label("Choose Audio", systemImage: "waveform")
                        .font(.caption)
                }
                .buttonStyle(.borderedProminent)
                .tint(tint)
            }
        }
        .frame(maxWidth: .infinity, minHeight: 130, alignment: .topLeading)
        .padding(12)
        .background(tint.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "waveform.badge.magnifyingglass")
                .font(.largeTitle)
                .foregroundStyle(.secondary)
            Text("Select two recordings to compare")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Text("35 acoustic metrics, computed on-device.")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 48)
    }

    // MARK: - Executive overview

    private func executiveOverview(_ c: BenchmarkComparison) -> some View {
        let a = c.reportA
        let b = c.reportB
        let sim = model.transcriptSimilarity
        return VStack(alignment: .leading, spacing: 8) {
            sectionHeader("Executive Overview", icon: "chart.bar.doc.horizontal")
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 10)], spacing: 10) {
                statCard("Word Count",
                         sim.wordsA > 0 ? "\(sim.wordsA)" : "—",
                         sim.wordsB > 0 ? "\(sim.wordsB)" : "—")
                statCard("Clipped Samples",
                         "\(a.totalClippedSamples)",
                         "\(b.totalClippedSamples)")
                statCard("Noise Floor Δ",
                         DualBenchmarkViewModel.number(a.p5NoiseFloorDBFS, 1) + " dB",
                         DualBenchmarkViewModel.number(b.p5NoiseFloorDBFS, 1) + " dB")
                statCard("Vocal Clarity",
                         DualBenchmarkViewModel.number(a.vocalSpeechBandPct, 1) + "%",
                         DualBenchmarkViewModel.number(b.vocalSpeechBandPct, 1) + "%")
            }
            .padding(.horizontal)
        }
    }

    private func statCard(_ title: String, _ valueA: String, _ valueB: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 1) {
                    Text("A").font(.system(size: 9)).foregroundStyle(columnA)
                    Text(valueA).font(.subheadline).fontWeight(.semibold)
                }
                Divider().frame(height: 24)
                VStack(alignment: .leading, spacing: 1) {
                    Text("B").font(.system(size: 9)).foregroundStyle(columnB)
                    Text(valueB).font(.subheadline).fontWeight(.semibold)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Charts

    private struct WaveformPoint: Identifiable {
        let id = UUID()
        let file: String
        let time: Double
        let dbfs: Double
    }

    private struct BandPoint: Identifiable {
        let id = UUID()
        let file: String
        let label: String
        let value: Double
    }

    private func waveformChart(_ c: BenchmarkComparison) -> some View {
        let points = c.reportA.waveformSeries().map { WaveformPoint(file: "A", time: $0.time, dbfs: $0.dbfs) }
            + c.reportB.waveformSeries().map { WaveformPoint(file: "B", time: $0.time, dbfs: $0.dbfs) }

        return chartCard("Loudness Over Time", icon: "waveform.path.ecg") {
            Chart(points) { point in
                LineMark(
                    x: .value("Time (s)", point.time),
                    y: .value("dBFS", point.dbfs)
                )
                .foregroundStyle(by: .value("File", point.file))
                .lineStyle(StrokeStyle(lineWidth: 1))
            }
            .chartForegroundStyleScale(["A": columnA, "B": columnB])
            .chartYScale(domain: -90...0)
            .frame(height: 170)
        }
    }

    private func loudnessChart(_ c: BenchmarkComparison) -> some View {
        let points: [BandPoint] = [
            BandPoint(file: "A", label: "P95 Peak", value: c.reportA.p95SpeechPeakDBFS),
            BandPoint(file: "B", label: "P95 Peak", value: c.reportB.p95SpeechPeakDBFS),
            BandPoint(file: "A", label: "P50 Median", value: c.reportA.p50MedianDBFS),
            BandPoint(file: "B", label: "P50 Median", value: c.reportB.p50MedianDBFS),
            BandPoint(file: "A", label: "P5 Floor", value: c.reportA.p5NoiseFloorDBFS),
            BandPoint(file: "B", label: "P5 Floor", value: c.reportB.p5NoiseFloorDBFS),
        ]

        return chartCard("Loudness & Noise Floor", icon: "speaker.wave.3") {
            Chart(points) { point in
                BarMark(
                    x: .value("Band", point.label),
                    y: .value("dBFS", point.value)
                )
                .foregroundStyle(by: .value("File", point.file))
                .position(by: .value("File", point.file))
            }
            .chartForegroundStyleScale(["A": columnA, "B": columnB])
            .frame(height: 170)
        }
    }

    private func spectralChart(_ c: BenchmarkComparison) -> some View {
        let points: [BandPoint] = [
            BandPoint(file: "A", label: "Sub-Bass", value: c.reportA.subBassRumblePct),
            BandPoint(file: "B", label: "Sub-Bass", value: c.reportB.subBassRumblePct),
            BandPoint(file: "A", label: "Low-Mid", value: c.reportA.lowMidVoicePct),
            BandPoint(file: "B", label: "Low-Mid", value: c.reportB.lowMidVoicePct),
            BandPoint(file: "A", label: "Vocal", value: c.reportA.vocalSpeechBandPct),
            BandPoint(file: "B", label: "Vocal", value: c.reportB.vocalSpeechBandPct),
            BandPoint(file: "A", label: "Air", value: c.reportA.highAirSibilancePct),
            BandPoint(file: "B", label: "Air", value: c.reportB.highAirSibilancePct),
        ]

        return chartCard("Spectral Energy Distribution", icon: "chart.bar.fill") {
            Chart(points) { point in
                BarMark(
                    x: .value("Band", point.label),
                    y: .value("Energy %", point.value)
                )
                .foregroundStyle(by: .value("File", point.file))
                .position(by: .value("File", point.file))
            }
            .chartForegroundStyleScale(["A": columnA, "B": columnB])
            .frame(height: 170)
        }
    }

    private func chartCard<Content: View>(
        _ title: String,
        icon: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionHeader(title, icon: icon)
            content()
                .padding(12)
                .background(.ultraThinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .padding(.horizontal)
        }
    }

    // MARK: - 35-metric table

    /// Column widths, defined once. Header and rows both read these, so the two
    /// cannot drift out of alignment.
    private enum Col {
        static let index: CGFloat = 34
        static let parameter: CGFloat = 150
        static let value: CGFloat = 96
        static let variance: CGFloat = 92
        static let takeaway: CGFloat = 320
    }

    private var metricTable: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionHeader("35-Metric Comparison", icon: "tablecells")
            // Horizontal scrolling only. The table is left at its natural height
            // so the page's own vertical scroll reaches every row: a fixed
            // maxHeight here clipped rows with no way to scroll to them, and
            // nesting a second vertical scroll view fights the page's gesture.
            ScrollView(.horizontal, showsIndicators: true) {
                LazyVStack(alignment: .leading, spacing: 0) {
                    tableHeader
                    ForEach(model.metricRows) { row in
                        metricRowView(row)
                    }
                }
            }
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .padding(.horizontal)
        }
    }

    private var tableHeader: some View {
        HStack(alignment: .top, spacing: 0) {
            cell("#", width: Col.index, bold: true)
            cell("Parameter", width: Col.parameter, bold: true)
            cell("File A", width: Col.value, bold: true).foregroundStyle(columnA)
            cell("File B", width: Col.value, bold: true).foregroundStyle(columnB)
            cell("Variance", width: Col.variance, bold: true)
            cell("Engineering & Takeaways", width: Col.takeaway, bold: true)
        }
        .background(.regularMaterial)
        .overlay(alignment: .bottom) { Divider() }
    }

    private func metricRowView(_ row: MetricRow) -> some View {
        HStack(alignment: .top, spacing: 0) {
            cell("\(row.index)", width: Col.index)
                .foregroundStyle(.tertiary)
            cell(row.parameter, width: Col.parameter, bold: true)
            cell(row.valueA, width: Col.value, monospaced: true)
                .background(columnA.opacity(0.10))
            cell(row.valueB, width: Col.value, monospaced: true)
                .background(columnB.opacity(0.10))
            cell(row.variance, width: Col.variance, monospaced: true)
            cell(row.takeaway, width: Col.takeaway)
                .foregroundStyle(.secondary)
        }
        .background(row.isCritical ? Color.cyan.opacity(0.12) : Color.clear)
        .overlay(alignment: .bottom) {
            Divider()
        }
    }

    private func cell(_ text: String,
                      width: CGFloat,
                      bold: Bool = false,
                      monospaced: Bool = false) -> some View {
        Text(text)
            .font(.caption2)
            .fontWeight(bold ? .semibold : .regular)
            .monospaced(monospaced)
            .frame(width: width, alignment: .topLeading)
            .padding(.horizontal, 6)
            .padding(.vertical, 7)
            .fixedSize(horizontal: false, vertical: true)
    }

    // MARK: - Transcripts

    private var transcriptViewer: some View {
        let sim = model.transcriptSimilarity
        return VStack(alignment: .leading, spacing: 8) {
            sectionHeader("Transcripts", icon: "text.alignleft")

            HStack(spacing: 8) {
                if model.isComparingText {
                    ProgressView().controlSize(.mini)
                    Text("Comparing…").font(.caption2).foregroundStyle(.secondary)
                } else if let ratio = sim.ratio {
                    Text("Similarity \(DualBenchmarkViewModel.number(ratio * 100, 1))%")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .monospaced()
                    if sim.truncated {
                        // No silent culling: say when only a prefix was scored.
                        Text("first 4,000 words")
                            .font(.system(size: 9))
                            .padding(.horizontal, 5)
                            .padding(.vertical, 2)
                            .background(Color.orange.opacity(0.18))
                            .clipShape(Capsule())
                    }
                } else {
                    Text("Load a transcript into each side to score similarity.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal)

            corpusStatus

            // Stacked, not side by side: two ~170pt columns left no room for the
            // load control or the match explanation.
            VStack(alignment: .leading, spacing: 14) {
                transcriptPane(text: $model.transcriptA, slot: .a, tint: columnA)
                Divider()
                transcriptPane(text: $model.transcriptB, slot: .b, tint: columnB)
            }
            .padding(.horizontal)
        }
    }

    /// Whether the corpus actually loaded. A silent empty list is impossible to
    /// diagnose, so say what happened either way and offer a retry.
    private var corpusStatus: some View {
        HStack(spacing: 8) {
            if model.isLoadingLibrary {
                ProgressView().controlSize(.mini)
                Text("Loading corpus…").font(.caption2).foregroundStyle(.secondary)
            } else if let error = model.libraryError {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.caption2)
                    .foregroundStyle(.orange)
                Text(error)
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .lineLimit(3)
                Button("Retry") { Task { await model.reloadLibrary(api: api) } }
                    .font(.caption2)
                    .buttonStyle(.bordered)
            } else {
                Image(systemName: "tray.full")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text("\(model.library.count) recordings available")
                    .font(.caption2)
                    .monospaced()
                    .foregroundStyle(.secondary)
                Button("Reload") { Task { await model.reloadLibrary(api: api) } }
                    .font(.caption2)
                    .buttonStyle(.bordered)
            }
            Spacer()
        }
        .padding(.horizontal)
    }

    private func transcriptPane(text: Binding<String>, slot: BenchmarkSlot, tint: Color) -> some View {
        let loops = model.loops(for: slot)
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(slot.label).font(.caption).fontWeight(.semibold).foregroundStyle(tint)
                Spacer()
                Text("\(text.wrappedValue.split(whereSeparator: \.isWhitespace).count) words")
                    .font(.caption2)
                    .monospaced()
                    .foregroundStyle(.secondary)
            }

            matchStatus(for: slot)

            // The corpus already holds every transcript, so pull from it rather
            // than making similarity depend on hand-pasted text.
            Button {
                pickingSlot = slot
            } label: {
                if model.loadingTranscriptFor == slot {
                    HStack(spacing: 4) {
                        ProgressView().controlSize(.mini)
                        Text("Loading…").font(.caption2)
                    }
                } else if let pick = model.picked(for: slot) {
                    Label(pick.title, systemImage: "text.book.closed")
                        .font(.caption2)
                        .lineLimit(1)
                } else {
                    Label("Load from Library", systemImage: "tray.and.arrow.down")
                        .font(.caption2)
                }
            }
            .buttonStyle(.bordered)
            .disabled(model.loadingTranscriptFor != nil)

            TextEditor(text: text)
                .font(.caption2)
                .frame(height: 150)
                .scrollContentBackground(.hidden)
                .background(.ultraThinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 8))

            if !loops.isEmpty {
                Text("Possible decoder loops")
                    .font(.system(size: 9))
                    .foregroundStyle(.secondary)
                FlowRow(spacing: 4) {
                    ForEach(loops) { loop in
                        Text("\"\(loop.token)\" ×\(loop.count)")
                            .font(.system(size: 9))
                            .padding(.horizontal, 5)
                            .padding(.vertical, 2)
                            .background(Color.orange.opacity(0.18))
                            .clipShape(Capsule())
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Shared

    @ViewBuilder
    private func transcriptPicker(for slot: BenchmarkSlot) -> some View {
        NavigationStack {
            List(model.library) { pick in
                Button {
                    pickingSlot = nil
                    Task {
                        await model.attachTranscript(pick, api: api, into: slot)
                    }
                } label: {
                    VStack(alignment: .leading) {
                        Text(pick.title).font(.headline).foregroundColor(.primary)
                        Text(pick.subtitle).font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Select Transcript")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { pickingSlot = nil }
                }
            }
            .overlay {
                if model.isLoadingLibrary {
                    ProgressView("Loading...")
                } else if let error = model.libraryError {
                    Text(error).foregroundStyle(.red).padding()
                }
            }
        }
    }

    /// What we recognised the imported file as. Reuse is labelled, ambiguity is
    /// shown rather than resolved by guessing, and a miss offers transcription.
    @ViewBuilder
    private func matchStatus(for slot: BenchmarkSlot) -> some View {
        switch model.match(for: slot) {
        case .unique(let candidate, let reason):
            Label("Reused “\(candidate.title)” — \(reason)", systemImage: "checkmark.seal")
                .font(.system(size: 9))
                .foregroundStyle(.green)
                .lineLimit(2)

        case .ambiguous(let candidates, let reason):
            VStack(alignment: .leading, spacing: 3) {
                Label(reason, systemImage: "questionmark.circle")
                    .font(.system(size: 9))
                    .foregroundStyle(.orange)
                ForEach(candidates) { candidate in
                    Button(candidate.title) {
                        if let pick = model.library.first(where: { $0.id == candidate.id }) {
                            Task { await model.attachTranscript(pick, api: api, into: slot) }
                        }
                    }
                    .font(.system(size: 9))
                    .buttonStyle(.bordered)
                }
            }

        case .none:
            if model.report(for: slot) != nil, model.transcript(for: slot).isEmpty {
                if model.transcribingSlot == slot {
                    VStack(alignment: .leading, spacing: 2) {
                        ProgressView(value: model.transcribeProgress)
                        Text(model.transcribeNeedsDownload
                             ? "Downloading language model…"
                             : "Transcribing \(Int(model.transcribeProgress * 100))%")
                            .font(.system(size: 9))
                            .foregroundStyle(.secondary)
                    }
                } else {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("No corpus match for this file.")
                            .font(.system(size: 9))
                            .foregroundStyle(.secondary)
                        Button("Transcribe on device") {
                            Task { await model.transcribeOnDevice(slot: slot) }
                        }
                        .font(.system(size: 9))
                        .buttonStyle(.bordered)
                        .disabled(model.transcribingSlot != nil)
                    }
                }
            }
        }
    }

    private func sectionHeader(_ title: String, icon: String) -> some View {
        Label(title, systemImage: icon)
            .font(.caption)
            .fontWeight(.semibold)
            .foregroundStyle(.secondary)
            .padding(.horizontal)
    }
}

/// Minimal wrapping row, so the loop badges reflow instead of clipping.
private struct FlowRow: Layout {
    var spacing: CGFloat = 4

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var x: CGFloat = 0, y: CGFloat = 0, rowHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > maxWidth, x > 0 {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
        return CGSize(width: maxWidth == .infinity ? x : maxWidth, height: y + rowHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX, y = bounds.minY, rowHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > bounds.maxX, x > bounds.minX {
                x = bounds.minX
                y += rowHeight + spacing
                rowHeight = 0
            }
            subview.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
    }
}
