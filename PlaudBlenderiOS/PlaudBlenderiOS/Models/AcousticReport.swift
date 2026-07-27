import Foundation

/// Deep acoustic telemetry for a single audio file.
///
/// Metrics 1-33 of the 35-metric benchmark are per-file and live here. Metrics
/// 34-35 (clock sync) are inherently *relative* -- drift has no meaning without
/// a reference clock -- so they live on `ClockSyncMetrics`, which is derived
/// from a pair of reports. See `BenchmarkComparison`.
nonisolated struct AcousticReport: Sendable, Identifiable {
    let id: UUID

    // MARK: A. File metadata & encoding (1-9)

    let fileName: String                 // 1
    let fileSizeBytes: Int64             // 2
    let durationSeconds: Double          // 5, millisecond precision
    let totalFrames: Int64               // 6
    let sampleRateHz: Double             // 7
    let bitDepthEncoding: String         // 8
    let channelCount: UInt32

    /// Wall-clock time the recording started, from filesystem metadata. Feeds
    /// metric 34 once two files are compared.
    let recordingStartDate: Date?

    var fileSizeMB: Double { Double(fileSizeBytes) / 1_048_576 }  // 3

    var durationFormatted: String {      // 4
        let total = Int(durationSeconds.rounded())
        return String(format: "%02dh %02dm %02ds", total / 3600, (total % 3600) / 60, total % 60)
    }

    var channelCountBitrate: String {    // 9
        let label = channelCount == 1 ? "Mono" : (channelCount == 2 ? "Stereo" : "\(channelCount)ch")
        let kbps = Int((sampleRateHz * Double(bitsPerSample) * Double(channelCount)) / 1000)
        return "\(channelCount) \(label) (\(kbps) kbps)"
    }

    /// Bits per sample of the *source* encoding, used for the bitrate estimate.
    let bitsPerSample: Int

    // MARK: B. Speech recognition & AI fidelity (10-11)

    /// Word count of the supplied transcript. Nil when no transcript was given
    /// -- this analyzer does not run speech-to-text, it reports on text handed
    /// to it. Metric 11 is comparative and lives on `BenchmarkComparison`.
    let wordCount: Int?                  // 10

    // MARK: C. Preamp gain, peak levels & saturation (12-16)

    let peakSampleRaw: Int16             // 12
    let peakLevelDBFS: Double            // 13
    let totalClippedSamples: Int         // 14
    let positivePeakClips: Int           // 15
    let negativePeakClips: Int           // 16

    // MARK: D. Loudness dynamics & noise floor (17-24)

    let overallRMSDBFS: Double           // 17
    let p95SpeechPeakDBFS: Double        // 18
    let p50MedianDBFS: Double            // 19
    let p5NoiseFloorDBFS: Double         // 20
    let p1SilenceFloorDBFS: Double       // 21

    var usableDynamicRangeDB: Double { p95SpeechPeakDBFS - p5NoiseFloorDBFS }  // 22
    let crestFactorRatio: Double         // 23
    var crestFactorDB: Double { 20 * log10(max(crestFactorRatio, .leastNormalMagnitude)) }  // 24

    // MARK: E. Environmental acoustic ratios (25-28)

    let silenceRatioPct: Double          // 25, frames < -45 dBFS
    let quietAmbientPct: Double          // 26, -45 to -35
    let speechActivePct: Double          // 27, -35 to -25
    let loudSpeechPct: Double            // 28, > -25

    // MARK: F. Spectral energy distribution (29-33)

    let spectralCentroidHz: Double       // 29
    let subBassRumblePct: Double         // 30, 20-150 Hz
    let lowMidVoicePct: Double           // 31, 150-500 Hz
    let vocalSpeechBandPct: Double       // 32, 500-3000 Hz
    let highAirSibilancePct: Double      // 33, 3000-8000 Hz

    /// Per-frame loudness in dBFS, one entry per analysis frame, for waveform
    /// plotting. Downsampled for display by `waveformSeries(maxPoints:)`.
    let frameLoudnessDBFS: [Double]

    /// Evenly-spaced loudness samples for charting, capped at `maxPoints`.
    func waveformSeries(maxPoints: Int = 240) -> [(time: Double, dbfs: Double)] {
        guard !frameLoudnessDBFS.isEmpty else { return [] }
        let stride = max(1, frameLoudnessDBFS.count / maxPoints)
        let secondsPerFrame = durationSeconds / Double(frameLoudnessDBFS.count)
        return Swift.stride(from: 0, to: frameLoudnessDBFS.count, by: stride).map {
            (time: Double($0) * secondsPerFrame, dbfs: frameLoudnessDBFS[$0])
        }
    }
}

// MARK: - Clock sync (metrics 34-35)

/// Metrics 34-35. Both require a reference, so they are only defined for a
/// *pair* of recordings -- typically two devices capturing the same event.
nonisolated struct ClockSyncMetrics: Sendable {
    /// Metric 34. Seconds between the two files' recording start times. Nil
    /// when the filesystem did not report a creation date for both.
    let startTimeOffset: Double?

    /// Metric 35. Relative drift between the two device clocks, in parts per
    /// million, derived from how far their durations diverge over the same
    /// event. Positive means A ran long relative to B.
    let cumulativeClockDriftPPM: Double

    /// Metric 35, expressed as total seconds of accumulated divergence.
    let cumulativeDriftSeconds: Double

    static func between(_ a: AcousticReport, _ b: AcousticReport) -> ClockSyncMetrics {
        let offset: Double? = {
            guard let sa = a.recordingStartDate, let sb = b.recordingStartDate else { return nil }
            return sa.timeIntervalSince(sb)
        }()

        let drift = a.durationSeconds - b.durationSeconds
        let mean = (a.durationSeconds + b.durationSeconds) / 2
        let ppm = mean > 0 ? (drift / mean) * 1_000_000 : 0

        return ClockSyncMetrics(
            startTimeOffset: offset,
            cumulativeClockDriftPPM: ppm,
            cumulativeDriftSeconds: drift
        )
    }
}

// MARK: - Comparison

/// A single row of the side-by-side 35-metric table.
nonisolated struct MetricRow: Identifiable, Sendable {
    let id = UUID()
    let index: Int
    let parameter: String
    let valueA: String
    let valueB: String
    let variance: String
    let takeaway: String
    /// Critical metrics (clipping, rumble, noise floor) get a highlighted row.
    let isCritical: Bool
}

/// Pairs two reports and derives everything that only exists in relation:
/// transcript similarity, clock drift, and the full comparison table.
nonisolated struct BenchmarkComparison: Sendable {
    let reportA: AcousticReport
    let reportB: AcousticReport
    let clockSync: ClockSyncMetrics

    /// Metric 11. Levenshtein alignment ratio between the two transcripts,
    /// 0...1. Nil when either transcript is missing.
    let textSimilarityRatio: Double?

    /// True when the transcripts were longer than the comparison cap and the
    /// ratio above reflects a truncated prefix.
    let similarityWasTruncated: Bool

    init(reportA: AcousticReport,
         reportB: AcousticReport,
         transcriptA: String?,
         transcriptB: String?,
         similarityTokenCap: Int = 4_000) {
        self.reportA = reportA
        self.reportB = reportB
        self.clockSync = .between(reportA, reportB)

        if let ta = transcriptA, let tb = transcriptB, !ta.isEmpty, !tb.isEmpty {
            let wordsA = ta.split(whereSeparator: \.isWhitespace).map(String.init)
            let wordsB = tb.split(whereSeparator: \.isWhitespace).map(String.init)
            self.similarityWasTruncated = wordsA.count > similarityTokenCap || wordsB.count > similarityTokenCap
            self.textSimilarityRatio = Self.levenshteinRatio(
                Array(wordsA.prefix(similarityTokenCap)),
                Array(wordsB.prefix(similarityTokenCap))
            )
        } else {
            self.textSimilarityRatio = nil
            self.similarityWasTruncated = false
        }
    }

    /// Word-level Levenshtein similarity, `1 - distance / maxLength`.
    ///
    /// Two-row DP: O(n*m) time but only O(min(n,m)) memory, so a 4,000x4,000
    /// comparison costs ~16M cell updates and two 4k rows rather than a 16M-cell
    /// matrix.
    static func levenshteinRatio(_ a: [String], _ b: [String]) -> Double {
        if a.isEmpty && b.isEmpty { return 1 }
        if a.isEmpty || b.isEmpty { return 0 }

        // Iterate over the longer sequence so the rows stay as short as possible.
        let (short, long) = a.count <= b.count ? (a, b) : (b, a)

        var previous = Array(0...short.count)
        var current = [Int](repeating: 0, count: short.count + 1)

        for (i, longWord) in long.enumerated() {
            current[0] = i + 1
            for j in 0..<short.count {
                let cost = longWord == short[j] ? 0 : 1
                current[j + 1] = Swift.min(
                    current[j] + 1,          // insertion
                    previous[j + 1] + 1,     // deletion
                    previous[j] + cost       // substitution
                )
            }
            swap(&previous, &current)
        }

        let distance = Double(previous[short.count])
        return 1 - distance / Double(long.count)
    }
}
