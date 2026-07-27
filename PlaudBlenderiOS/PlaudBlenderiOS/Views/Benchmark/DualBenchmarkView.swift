import Charts
import SwiftUI
import UniformTypeIdentifiers

/// Side-by-side acoustic benchmark of two recordings.
struct DualBenchmarkView: View {
    @State private var model = DualBenchmarkViewModel()
    @State private var isImporting = false
    @State private var importingSlot: BenchmarkSlot = .a

    private let columnA = Color.red
    private let columnB = Color.green

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                fileCards

                if let comparison = model.comparison {
                    executiveOverview(comparison)
                    waveformChart(comparison)
                    loudnessChart(comparison)
                    spectralChart(comparison)
                    metricTable
                    transcriptViewer
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
                if let url = urls.first { model.analyze(url: url, into: importingSlot) }
            case .failure(let error):
                model.errorMessage = error.localizedDescription
            }
        }
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
        return VStack(alignment: .leading, spacing: 8) {
            sectionHeader("Executive Overview", icon: "chart.bar.doc.horizontal")
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 10)], spacing: 10) {
                statCard("Word Count",
                         a.wordCount.map(String.init) ?? "—",
                         b.wordCount.map(String.init) ?? "—")
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

    private var metricTable: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionHeader("35-Metric Comparison", icon: "tablecells")
            ScrollView(.horizontal, showsIndicators: true) {
                LazyVStack(alignment: .leading, spacing: 0) {
                    tableHeader
                    ForEach(model.metricRows) { row in
                        metricRowView(row)
                    }
                }
            }
            .frame(maxHeight: 520)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .padding(.horizontal)
        }
    }

    private var tableHeader: some View {
        HStack(spacing: 0) {
            cell("#", width: 34, bold: true)
            cell("Parameter", width: 150, bold: true)
            cell("File A", width: 96, bold: true).foregroundStyle(columnA)
            cell("File B", width: 96, bold: true).foregroundStyle(columnB)
            cell("Variance", width: 92, bold: true)
            cell("Engineering & Takeaways", width: 330, bold: true)
        }
        .background(.regularMaterial)
    }

    private func metricRowView(_ row: MetricRow) -> some View {
        HStack(spacing: 0) {
            cell("\(row.index)", width: 34)
                .foregroundStyle(.tertiary)
            cell(row.parameter, width: 150, bold: true)
            cell(row.valueA, width: 96)
                .background(columnA.opacity(0.10))
            cell(row.valueB, width: 96)
                .background(columnB.opacity(0.10))
            cell(row.variance, width: 92)
            cell(row.takeaway, width: 330)
                .foregroundStyle(.secondary)
        }
        .background(row.isCritical ? Color.cyan.opacity(0.12) : Color.clear)
        .overlay(alignment: .bottom) {
            Divider()
        }
    }

    private func cell(_ text: String, width: CGFloat, bold: Bool = false) -> some View {
        Text(text)
            .font(.caption2)
            .fontWeight(bold ? .semibold : .regular)
            .frame(width: width, alignment: .leading)
            .padding(.horizontal, 6)
            .padding(.vertical, 7)
            .fixedSize(horizontal: false, vertical: true)
    }

    // MARK: - Transcripts

    private var transcriptViewer: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionHeader("Transcripts", icon: "text.alignleft")
            HStack(alignment: .top, spacing: 12) {
                transcriptPane(text: $model.transcriptA, slot: .a, tint: columnA)
                transcriptPane(text: $model.transcriptB, slot: .b, tint: columnB)
            }
            .padding(.horizontal)
        }
    }

    private func transcriptPane(text: Binding<String>, slot: BenchmarkSlot, tint: Color) -> some View {
        let loops = model.loops(for: slot)
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(slot.label).font(.caption).fontWeight(.semibold).foregroundStyle(tint)
                Spacer()
                Text("\(text.wrappedValue.split(whereSeparator: \.isWhitespace).count) words")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

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
