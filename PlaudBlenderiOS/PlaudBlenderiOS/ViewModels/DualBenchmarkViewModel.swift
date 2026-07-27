import Foundation
import Observation

/// One side of the A/B benchmark.
enum BenchmarkSlot: String, CaseIterable, Sendable {
    case a, b

    var label: String { self == .a ? "File A" : "File B" }
}

/// A token that repeats far more than natural speech would explain -- the
/// signature of an STT decoder stuck in a loop.
struct RepeatedToken: Identifiable, Sendable {
    var id: String { token }
    let token: String
    let count: Int
}

@MainActor
@Observable
final class DualBenchmarkViewModel {

    private(set) var reportA: AcousticReport?
    private(set) var reportB: AcousticReport?

    var transcriptA: String = "" { didSet { scheduleRebuild() } }
    var transcriptB: String = "" { didSet { scheduleRebuild() } }

    /// Derived state, rebuilt when inputs change rather than on every render.
    ///
    /// These were computed properties, so a body evaluation re-ran the
    /// word-level Levenshtein -- up to 16M cell updates -- on every keystroke
    /// in the transcript editors.
    private(set) var comparison: BenchmarkComparison?
    private(set) var metricRows: [MetricRow] = []
    private(set) var loopsA: [RepeatedToken] = []
    private(set) var loopsB: [RepeatedToken] = []

    private var rebuildTask: Task<Void, Never>?

    var progressA: Double = 0
    var progressB: Double = 0
    var isAnalyzingA = false
    var isAnalyzingB = false

    var errorMessage: String?

    private var taskA: Task<Void, Never>?
    private var taskB: Task<Void, Never>?

    // MARK: - Analysis

    func analyze(url: URL, into slot: BenchmarkSlot) {
        let transcript = slot == .a ? transcriptA : transcriptB
        setAnalyzing(true, for: slot)
        setProgress(0, for: slot)

        // Built here rather than inside the task below: nesting a capture list
        // inside another closure's capture list re-captures `self` through the
        // outer binding, which is an error under Swift 6 concurrency.
        let onProgress: @Sendable (Double) -> Void = { [weak self] value in
            // Bind to an immutable local before the nested Task: a `weak self`
            // capture is a var, and capturing a var in concurrent code is an
            // error under Swift 6.
            guard let model = self else { return }
            Task { @MainActor in model.setProgress(value, for: slot) }
        }

        let task = Task { [weak self] in
            do {
                let report = try await AcousticAnalyzer.analyze(
                    url: url,
                    transcript: transcript.isEmpty ? nil : transcript,
                    progress: onProgress
                )
                guard !Task.isCancelled else { return }
                await MainActor.run { [weak self] in
                    self?.store(report, in: slot)
                    self?.setAnalyzing(false, for: slot)
                }
            } catch {
                guard !Task.isCancelled else { return }
                await MainActor.run { [weak self] in
                    self?.errorMessage = error.localizedDescription
                    self?.setAnalyzing(false, for: slot)
                }
            }
        }

        switch slot {
        case .a: taskA?.cancel(); taskA = task
        case .b: taskB?.cancel(); taskB = task
        }
    }

    func clear(_ slot: BenchmarkSlot) {
        switch slot {
        case .a: taskA?.cancel(); reportA = nil; progressA = 0; isAnalyzingA = false
        case .b: taskB?.cancel(); reportB = nil; progressB = 0; isAnalyzingB = false
        }
        rebuild()
    }

    private func store(_ report: AcousticReport, in slot: BenchmarkSlot) {
        switch slot {
        case .a: reportA = report
        case .b: reportB = report
        }
        rebuild()
    }

    /// Coalesce keystrokes so a long transcript is not re-aligned per character.
    private func scheduleRebuild() {
        rebuildTask?.cancel()
        rebuildTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled else { return }
            self?.rebuild()
        }
    }

    private func rebuild() {
        loopsA = Self.repeatedTokens(in: transcriptA)
        loopsB = Self.repeatedTokens(in: transcriptB)

        guard let a = reportA, let b = reportB else {
            comparison = nil
            metricRows = []
            return
        }

        let built = BenchmarkComparison(
            reportA: a,
            reportB: b,
            transcriptA: transcriptA.isEmpty ? nil : transcriptA,
            transcriptB: transcriptB.isEmpty ? nil : transcriptB
        )
        comparison = built
        metricRows = Self.buildRows(built)
    }

    func loops(for slot: BenchmarkSlot) -> [RepeatedToken] {
        slot == .a ? loopsA : loopsB
    }

    private func setProgress(_ value: Double, for slot: BenchmarkSlot) {
        switch slot {
        case .a: progressA = value
        case .b: progressB = value
        }
    }

    private func setAnalyzing(_ value: Bool, for slot: BenchmarkSlot) {
        switch slot {
        case .a: isAnalyzingA = value
        case .b: isAnalyzingB = value
        }
    }

    func report(for slot: BenchmarkSlot) -> AcousticReport? {
        slot == .a ? reportA : reportB
    }

    func isAnalyzing(_ slot: BenchmarkSlot) -> Bool {
        slot == .a ? isAnalyzingA : isAnalyzingB
    }

    func progress(for slot: BenchmarkSlot) -> Double {
        slot == .a ? progressA : progressB
    }

    // MARK: - Comparison

    /// Tokens repeated often enough to look like decoder loops rather than speech.
    static func repeatedTokens(in transcript: String, minimumCount: Int = 25, limit: Int = 6) -> [RepeatedToken] {
        guard !transcript.isEmpty else { return [] }
        var counts: [String: Int] = [:]
        for raw in transcript.split(whereSeparator: { $0.isWhitespace }) {
            let token = raw.lowercased().trimmingCharacters(in: .punctuationCharacters)
            guard token.count > 1 else { continue }
            counts[token, default: 0] += 1
        }
        return counts
            .filter { $0.value >= minimumCount }
            .sorted { $0.value > $1.value }
            .prefix(limit)
            .map { RepeatedToken(token: $0.key, count: $0.value) }
    }

    // MARK: - 35-metric table

    static func buildRows(_ comparison: BenchmarkComparison) -> [MetricRow] {
        let a = comparison.reportA
        let b = comparison.reportB
        let clock = comparison.clockSync

        var rows: [MetricRow] = []
        var index = 0

        func add(_ parameter: String,
                 _ valueA: String,
                 _ valueB: String,
                 _ variance: String,
                 _ takeaway: String,
                 critical: Bool = false) {
            index += 1
            rows.append(MetricRow(
                index: index,
                parameter: parameter,
                valueA: valueA,
                valueB: valueB,
                variance: variance,
                takeaway: takeaway,
                isCritical: critical
            ))
        }

        /// Numeric row with an automatically formatted signed delta (B - A).
        func addNumeric(_ parameter: String,
                        _ x: Double,
                        _ y: Double,
                        unit: String,
                        decimals: Int = 1,
                        takeaway: String,
                        critical: Bool = false) {
            add(parameter,
                Self.number(x, decimals) + unit,
                Self.number(y, decimals) + unit,
                Self.signed(y - x, decimals) + unit,
                takeaway,
                critical: critical)
        }

        // A. File metadata & encoding (1-9)
        add("File Name", a.fileName, b.fileName, "—",
            "💡 Which capture is which. Everything below is anchored to these two files.")
        addNumeric("File Size", Double(a.fileSizeBytes), Double(b.fileSizeBytes), unit: " B", decimals: 0,
                   takeaway: "💡 Raw bytes on disk. Bigger is not better — it usually just means longer or less compressed.")
        addNumeric("File Size (MB)", a.fileSizeMB, b.fileSizeMB, unit: " MB", decimals: 2,
                   takeaway: "💡 Practical footprint. Useful when deciding what to keep on device.")
        add("Duration", a.durationFormatted, b.durationFormatted,
            Self.signed(b.durationSeconds - a.durationSeconds, 1) + " s",
            "💡 Total runtime. A gap here between two devices recording the same event is your first clue about clock drift.")
        addNumeric("Duration (s)", a.durationSeconds, b.durationSeconds, unit: " s", decimals: 3,
                   takeaway: "💡 Millisecond-precise length, which is what the drift maths below actually uses.")
        addNumeric("Total Frames", Double(a.totalFrames), Double(b.totalFrames), unit: "", decimals: 0,
                   takeaway: "💡 Individual PCM samples captured. Frames ÷ sample rate = duration.")
        addNumeric("Sample Rate", a.sampleRateHz, b.sampleRateHz, unit: " Hz", decimals: 0,
                   takeaway: "💡 Ceiling on capturable frequency — you get at most half the sample rate. 16 kHz tops out at 8 kHz of audio.")
        add("Bit Depth / Encoding", a.bitDepthEncoding, b.bitDepthEncoding, "—",
            "💡 Resolution of each sample. 16-bit gives ~96 dB of headroom between loudest and quietest.")
        add("Channels / Bitrate", a.channelCountBitrate, b.channelCountBitrate, "—",
            "💡 Mono is normal for voice recorders. Bitrate is the raw uncompressed data rate.")

        // B. Speech recognition & AI fidelity (10-11)
        add("Word Count",
            a.wordCount.map(String.init) ?? "—",
            b.wordCount.map(String.init) ?? "—",
            a.wordCount != nil && b.wordCount != nil ? Self.signed(Double(b.wordCount! - a.wordCount!), 0) : "—",
            "💡 Words the transcriber produced. Wildly more words than the other file often means hallucinated filler, not richer capture.")
        add("Text Similarity",
            comparison.textSimilarityRatio != nil ? Self.number(comparison.textSimilarityRatio! * 100, 1) + "%" : "—",
            comparison.similarityWasTruncated ? "(first 4k words)" : "—",
            "—",
            "💡 How closely the two transcripts agree, word by word. Low agreement on the same event means one device misheard a lot.")

        // C. Preamp gain, peak levels & saturation (12-16)
        addNumeric("Peak Sample (raw)", Double(a.peakSampleRaw), Double(b.peakSampleRaw), unit: "", decimals: 0,
                   takeaway: "💡 Loudest single sample on the ±32,767 scale. Sitting exactly at 32,767 means the preamp ran out of room.")
        addNumeric("Peak Level", a.peakLevelDBFS, b.peakLevelDBFS, unit: " dBFS",
                   takeaway: "💡 Headroom before distortion. 0 dBFS is the hard ceiling; −3 to −6 is a healthy target.", critical: true)
        addNumeric("Clipped Samples", Double(a.totalClippedSamples), Double(b.totalClippedSamples), unit: "", decimals: 0,
                   takeaway: "💡 Samples that hit the ceiling and got flattened. This is permanent distortion — no amount of processing recovers it.", critical: true)
        addNumeric("Positive Clips", Double(a.positivePeakClips), Double(b.positivePeakClips), unit: "", decimals: 0,
                   takeaway: "💡 Clipping on the upward half of the wave.", critical: true)
        addNumeric("Negative Clips", Double(a.negativePeakClips), Double(b.negativePeakClips), unit: "", decimals: 0,
                   takeaway: "💡 Clipping on the downward half. A big imbalance versus positive clips suggests a DC offset problem.", critical: true)

        // D. Loudness dynamics & noise floor (17-24)
        addNumeric("Overall RMS", a.overallRMSDBFS, b.overallRMSDBFS, unit: " dBFS",
                   takeaway: "💡 Average perceived loudness across the whole file.")
        addNumeric("P95 Speech Peak", a.p95SpeechPeakDBFS, b.p95SpeechPeakDBFS, unit: " dBFS",
                   takeaway: "💡 Voice punchiness — how loud you actually get when speaking, ignoring rare outliers.")
        addNumeric("P50 Median", a.p50MedianDBFS, b.p50MedianDBFS, unit: " dBFS",
                   takeaway: "💡 The typical moment in the recording. Close to the noise floor means mostly silence.")
        addNumeric("P5 Noise Floor", a.p5NoiseFloorDBFS, b.p5NoiseFloorDBFS, unit: " dBFS",
                   takeaway: "💡 The hiss underneath everything. Lower is better; every 6 dB is a halving of audible noise.", critical: true)
        addNumeric("P1 Silence Floor", a.p1SilenceFloorDBFS, b.p1SilenceFloorDBFS, unit: " dBFS",
                   takeaway: "💡 The quietest the hardware ever gets — effectively the noise floor of the preamp itself.")
        addNumeric("Usable Dynamic Range", a.usableDynamicRangeDB, b.usableDynamicRangeDB, unit: " dB",
                   takeaway: "💡 The gap your voice occupies above the noise. Bigger is cleaner; under ~20 dB and speech starts fighting hiss.", critical: true)
        addNumeric("Crest Factor", a.crestFactorRatio, b.crestFactorRatio, unit: "×", decimals: 2,
                   takeaway: "💡 Peak-to-average ratio. Natural speech runs high; a low number means heavy compression or clipping.")
        addNumeric("Crest Factor (dB)", a.crestFactorDB, b.crestFactorDB, unit: " dB",
                   takeaway: "💡 Same thing in decibels. Speech typically lands 12–20 dB.")

        // E. Environmental acoustic ratios (25-28)
        addNumeric("Silence Ratio", a.silenceRatioPct, b.silenceRatioPct, unit: "%",
                   takeaway: "💡 Share of the recording below −45 dBFS — dead air. High on a long recording is normal.")
        addNumeric("Quiet Ambient", a.quietAmbientPct, b.quietAmbientPct, unit: "%",
                   takeaway: "💡 Room tone: HVAC, traffic, distant noise. Not speech, not silence.")
        addNumeric("Speech Active", a.speechActivePct, b.speechActivePct, unit: "%",
                   takeaway: "💡 Share of time in the normal conversational band. This is your actual signal.")
        addNumeric("Loud Speech", a.loudSpeechPct, b.loudSpeechPct, unit: "%",
                   takeaway: "💡 Time spent above −25 dBFS. A lot of this alongside clipping means the gain is set too hot.")

        // F. Spectral energy distribution (29-33)
        addNumeric("Spectral Centroid", a.spectralCentroidHz, b.spectralCentroidHz, unit: " Hz", decimals: 0,
                   takeaway: "💡 The centre of gravity of the sound — brightness. Higher reads crisper, lower reads muffled.")
        addNumeric("Sub-Bass Rumble", a.subBassRumblePct, b.subBassRumblePct, unit: "%", decimals: 2,
                   takeaway: "💡 Energy at 20–150 Hz: handling noise, pockets, desk thumps. Almost pure waste in a voice recording.", critical: true)
        addNumeric("Low-Mid Voice", a.lowMidVoicePct, b.lowMidVoicePct, unit: "%", decimals: 2,
                   takeaway: "💡 150–500 Hz — the body and warmth of a voice. Too much sounds boomy.")
        addNumeric("Vocal Speech Band", a.vocalSpeechBandPct, b.vocalSpeechBandPct, unit: "%", decimals: 2,
                   takeaway: "💡 500–3,000 Hz, where intelligibility lives. More energy here is the single best predictor of clean transcription.", critical: true)
        addNumeric("High Air / Sibilance", a.highAirSibilancePct, b.highAirSibilancePct, unit: "%", decimals: 2,
                   takeaway: "💡 3–8 kHz: consonants and clarity. Too little sounds dull; too much is harsh and hissy.")

        // G. Clock sync (34-35)
        add("Start Time Offset",
            clock.startTimeOffset.map { Self.signed($0, 2) + " s" } ?? "—",
            clock.startTimeOffset != nil ? "(A relative to B)" : "—",
            "—",
            "💡 How far apart the two recordings began, from file timestamps. Needed before drift can be read meaningfully.")
        add("Cumulative Clock Drift",
            Self.signed(clock.cumulativeClockDriftPPM, 1) + " ppm",
            Self.signed(clock.cumulativeDriftSeconds, 2) + " s total",
            "—",
            "💡 How far the two device clocks diverged over the same event. Consumer crystals drift 20–100 ppm — about 1 second per 3 hours.")

        return rows
    }

    // MARK: - Formatting

    static func number(_ value: Double, _ decimals: Int) -> String {
        guard value.isFinite else { return "—" }
        return String(format: "%.\(decimals)f", value)
    }

    static func signed(_ value: Double, _ decimals: Int) -> String {
        guard value.isFinite else { return "—" }
        let formatted = String(format: "%.\(decimals)f", abs(value))
        if abs(value) < pow(10, -Double(decimals)) / 2 { return "0" }
        return (value > 0 ? "+" : "−") + formatted
    }
}
