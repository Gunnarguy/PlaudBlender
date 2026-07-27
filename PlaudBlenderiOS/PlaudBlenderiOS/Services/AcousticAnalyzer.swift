import Accelerate
import AVFoundation
import Foundation

nonisolated enum AcousticAnalyzerError: LocalizedError {
    case unreadableFile(String)
    case unsupportedFormat
    case emptyFile
    case fftSetupFailed

    var errorDescription: String? {
        switch self {
        case .unreadableFile(let reason): return "Could not read audio file: \(reason)"
        case .unsupportedFormat: return "Unsupported audio format."
        case .emptyFile: return "The audio file contains no samples."
        case .fftSetupFailed: return "Could not initialise the FFT processor."
        }
    }
}

/// Streaming acoustic analyzer built on vDSP.
///
/// Decodes an audio file in fixed-size blocks and folds each block into running
/// accumulators, so peak memory is bounded by the block size rather than the
/// file length -- a four-hour 16 kHz recording is ~230M samples but never has
/// more than a few hundred kilobytes resident.
nonisolated enum AcousticAnalyzer {

    /// Decode block size in frames. 32,768 float frames is 128 KB per channel.
    private static let blockFrames: AVAudioFrameCount = 32_768

    /// Loudness analysis window. 100 ms at any sample rate.
    private static let analysisWindowSeconds = 0.1

    private static let fftSize = 1_024
    private static let fftHop = 512

    /// Samples at or beyond ±32,767 count as clipped.
    ///
    /// Int16 is asymmetric: +32,767 converts to 32767/32768 = 0.99996948, which
    /// is *below* full scale, while -32,768 converts to exactly -1.0. A naive
    /// `1.0 - halfLSB` threshold therefore catches every negative clip and no
    /// positive one -- a symmetric clipped sine reported +0/-50000. Anchoring
    /// half an LSB under +32,767 instead makes both rails count.
    private static let clipThreshold: Float = 32_766.5 / 32_768.0

    /// Floor for dBFS conversion, so digital silence reports a finite number.
    private static let dbFloor: Double = -120

    /// Analyze `url`, reporting progress in 0...1.
    ///
    /// - Parameter transcript: optional text used for metric 10. This analyzer
    ///   does not perform speech-to-text; it reports on text it is handed.
    static func analyze(
        url: URL,
        transcript: String? = nil,
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> AcousticReport {
        try await Task.detached(priority: .userInitiated) {
            try analyzeSync(url: url, transcript: transcript, progress: progress)
        }.value
    }

    // MARK: - Core

    private static func analyzeSync(
        url: URL,
        transcript: String?,
        progress: (@Sendable (Double) -> Void)?
    ) throws -> AcousticReport {
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }

        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw AcousticAnalyzerError.unreadableFile(error.localizedDescription)
        }

        let sampleRate = file.fileFormat.sampleRate
        let channels = file.fileFormat.channelCount
        let totalFrames = file.length
        guard totalFrames > 0, sampleRate > 0 else { throw AcousticAnalyzerError.emptyFile }

        guard let readFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: channels,
            interleaved: false
        ), let buffer = AVAudioPCMBuffer(pcmFormat: readFormat, frameCapacity: blockFrames) else {
            throw AcousticAnalyzerError.unsupportedFormat
        }

        guard let fft = FFTProcessor(size: fftSize) else {
            throw AcousticAnalyzerError.fftSetupFailed
        }

        let windowLength = max(1, Int(sampleRate * analysisWindowSeconds))

        var accumulator = Accumulator(
            sampleRate: sampleRate,
            windowLength: windowLength,
            fftSize: fftSize,
            fftHop: fftHop
        )

        var framesRead: Int64 = 0
        var mono = [Float]()

        while framesRead < totalFrames {
            try file.read(into: buffer, frameCount: blockFrames)
            let count = Int(buffer.frameLength)
            if count == 0 { break }

            downmix(buffer, frameCount: count, into: &mono)
            accumulator.consume(mono, fft: fft)

            framesRead += Int64(count)
            progress?(min(1, Double(framesRead) / Double(totalFrames)))

            if Task.isCancelled { break }
        }

        accumulator.finish(fft: fft)
        progress?(1)

        return accumulator.makeReport(
            url: url,
            sampleRate: sampleRate,
            channels: channels,
            totalFrames: totalFrames,
            bitsPerSample: bitsPerSample(of: file),
            bitDepthEncoding: encodingLabel(of: file),
            transcript: transcript
        )
    }

    /// Average all channels into a single mono buffer, reused across blocks.
    private static func downmix(_ buffer: AVAudioPCMBuffer, frameCount: Int, into mono: inout [Float]) {
        guard let data = buffer.floatChannelData else {
            mono = []
            return
        }
        let channels = Int(buffer.format.channelCount)

        if mono.count != frameCount {
            mono = [Float](repeating: 0, count: frameCount)
        }

        if channels == 1 {
            mono.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.update(from: data[0], count: frameCount)
            }
            return
        }

        mono.withUnsafeMutableBufferPointer { dst in
            guard let base = dst.baseAddress else { return }
            vDSP_vclr(base, 1, vDSP_Length(frameCount))
            for channel in 0..<channels {
                vDSP_vadd(base, 1, data[channel], 1, base, 1, vDSP_Length(frameCount))
            }
            var divisor = Float(channels)
            vDSP_vsdiv(base, 1, &divisor, base, 1, vDSP_Length(frameCount))
        }
    }

    private static func bitsPerSample(of file: AVAudioFile) -> Int {
        let bits = Int(file.fileFormat.streamDescription.pointee.mBitsPerChannel)
        return bits > 0 ? bits : 32
    }

    private static func encodingLabel(of file: AVAudioFile) -> String {
        let asbd = file.fileFormat.streamDescription.pointee
        let bits = bitsPerSample(of: file)
        let isFloat = asbd.mFormatFlags & kAudioFormatFlagIsFloat != 0
        switch asbd.mFormatID {
        case kAudioFormatLinearPCM: return "\(bits)-bit \(isFloat ? "Float" : "PCM")"
        case kAudioFormatMPEG4AAC: return "AAC"
        case kAudioFormatMPEGLayer3: return "MP3"
        case kAudioFormatOpus: return "Opus"
        case kAudioFormatAppleLossless: return "ALAC \(bits)-bit"
        default: return "\(bits)-bit"
        }
    }

    static func dbfs(_ linear: Double) -> Double {
        linear <= 0 ? dbFloor : max(dbFloor, 20 * log10(linear))
    }
}

// MARK: - Running accumulator

/// Folds decoded blocks into the statistics needed for the report. Kept as a
/// struct of scalars plus two small carry buffers so memory stays flat.
nonisolated private struct Accumulator {
    let sampleRate: Double
    let windowLength: Int
    let fftSize: Int
    let fftHop: Int

    // Global
    var sumOfSquares: Double = 0
    var sampleCount: Int = 0
    var peakMagnitude: Float = 0
    var positiveClips: Int = 0
    var negativeClips: Int = 0

    // Per-window loudness
    var loudnessDBFS: [Double] = []
    var windowCarry: [Float] = []

    // Spectral
    var fftCarry: [Float] = []
    var bandEnergy = [Double](repeating: 0, count: 4)   // sub-bass, low-mid, vocal, air
    var spectralTotal: Double = 0
    var centroidNumerator: Double = 0
    var centroidDenominator: Double = 0

    mutating func consume(_ mono: [Float], fft: FFTProcessor) {
        guard !mono.isEmpty else { return }

        mono.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return }
            let n = vDSP_Length(mono.count)

            var blockSumSquares: Float = 0
            vDSP_svesq(base, 1, &blockSumSquares, n)
            sumOfSquares += Double(blockSumSquares)
            sampleCount += mono.count

            var blockPeak: Float = 0
            vDSP_maxmgv(base, 1, &blockPeak, n)
            peakMagnitude = max(peakMagnitude, blockPeak)

            // Clip counting needs the sign, so it is a direct scan. Bounds-check
            // free via the unsafe pointer; this is a single comparison per sample.
            let threshold = AcousticAnalyzer.clipThresholdValue
            for i in 0..<mono.count {
                let v = base[i]
                if v >= threshold { positiveClips += 1 }
                else if v <= -threshold { negativeClips += 1 }
            }
        }

        windowCarry.append(contentsOf: mono)
        drainWindows()

        fftCarry.append(contentsOf: mono)
        drainFFT(fft: fft)
    }

    private mutating func drainWindows() {
        var offset = 0
        while windowCarry.count - offset >= windowLength {
            let slice = windowCarry[offset..<(offset + windowLength)]
            var rms: Float = 0
            slice.withUnsafeBufferPointer { ptr in
                if let base = ptr.baseAddress {
                    vDSP_rmsqv(base, 1, &rms, vDSP_Length(windowLength))
                }
            }
            loudnessDBFS.append(AcousticAnalyzer.dbfs(Double(rms)))
            offset += windowLength
        }
        if offset > 0 { windowCarry.removeFirst(offset) }
    }

    private mutating func drainFFT(fft: FFTProcessor) {
        var offset = 0
        while fftCarry.count - offset >= fftSize {
            let magnitudes = fft.magnitudes(fftCarry[offset..<(offset + fftSize)])
            fold(magnitudes)
            offset += fftHop
        }
        if offset > 0 { fftCarry.removeFirst(offset) }
    }

    /// Accumulate one spectrum into the band totals and the centroid sums.
    ///
    /// vDSP's real FFT output carries a constant scale factor, which cancels in
    /// both the band percentages and the magnitude-weighted centroid, so no
    /// normalisation is needed here.
    private mutating func fold(_ magnitudes: [Float]) {
        let binWidth = sampleRate / Double(fftSize)
        for (bin, magnitude) in magnitudes.enumerated() where bin > 0 {
            let frequency = Double(bin) * binWidth
            let m = Double(magnitude)

            centroidNumerator += frequency * m
            centroidDenominator += m
            spectralTotal += m

            switch frequency {
            case 20..<150:    bandEnergy[0] += m
            case 150..<500:   bandEnergy[1] += m
            case 500..<3_000: bandEnergy[2] += m
            case 3_000..<8_000: bandEnergy[3] += m
            default: break
            }
        }
    }

    /// Flush any trailing partial window so short files still report loudness.
    mutating func finish(fft: FFTProcessor) {
        if loudnessDBFS.isEmpty, !windowCarry.isEmpty {
            var rms: Float = 0
            windowCarry.withUnsafeBufferPointer { ptr in
                if let base = ptr.baseAddress {
                    vDSP_rmsqv(base, 1, &rms, vDSP_Length(windowCarry.count))
                }
            }
            loudnessDBFS.append(AcousticAnalyzer.dbfs(Double(rms)))
        }

        if spectralTotal == 0, !fftCarry.isEmpty {
            var padded = fftCarry
            padded.append(contentsOf: [Float](repeating: 0, count: max(0, fftSize - padded.count)))
            fold(fft.magnitudes(padded[0..<fftSize]))
        }
    }

    func makeReport(
        url: URL,
        sampleRate: Double,
        channels: UInt32,
        totalFrames: Int64,
        bitsPerSample: Int,
        bitDepthEncoding: String,
        transcript: String?
    ) -> AcousticReport {
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        let fileSize = (attributes?[.size] as? NSNumber)?.int64Value ?? 0
        let created = attributes?[.creationDate] as? Date

        let rmsLinear = sampleCount > 0 ? sqrt(sumOfSquares / Double(sampleCount)) : 0
        let peakLinear = Double(peakMagnitude)

        let sorted = loudnessDBFS.sorted()
        func percentile(_ p: Double) -> Double {
            guard !sorted.isEmpty else { return AcousticAnalyzer.dbfs(0) }
            let idx = Int((Double(sorted.count - 1) * p).rounded())
            return sorted[min(max(idx, 0), sorted.count - 1)]
        }

        let total = Double(max(loudnessDBFS.count, 1))
        let silence = Double(loudnessDBFS.filter { $0 < -45 }.count) / total * 100
        let quiet = Double(loudnessDBFS.filter { $0 >= -45 && $0 < -35 }.count) / total * 100
        let active = Double(loudnessDBFS.filter { $0 >= -35 && $0 < -25 }.count) / total * 100
        let loud = Double(loudnessDBFS.filter { $0 >= -25 }.count) / total * 100

        let spectral = max(spectralTotal, .leastNormalMagnitude)
        let centroid = centroidDenominator > 0 ? centroidNumerator / centroidDenominator : 0

        let words = transcript.map { $0.split(whereSeparator: \.isWhitespace).count }

        return AcousticReport(
            id: UUID(),
            fileName: url.lastPathComponent,
            fileSizeBytes: fileSize,
            durationSeconds: Double(totalFrames) / sampleRate,
            totalFrames: totalFrames,
            sampleRateHz: sampleRate,
            bitDepthEncoding: bitDepthEncoding,
            channelCount: channels,
            recordingStartDate: created,
            bitsPerSample: bitsPerSample,
            wordCount: words,
            peakSampleRaw: Int16(clamping: Int(( peakLinear * 32_767).rounded())),
            peakLevelDBFS: AcousticAnalyzer.dbfs(peakLinear),
            totalClippedSamples: positiveClips + negativeClips,
            positivePeakClips: positiveClips,
            negativePeakClips: negativeClips,
            overallRMSDBFS: AcousticAnalyzer.dbfs(rmsLinear),
            p95SpeechPeakDBFS: percentile(0.95),
            p50MedianDBFS: percentile(0.50),
            p5NoiseFloorDBFS: percentile(0.05),
            p1SilenceFloorDBFS: percentile(0.01),
            crestFactorRatio: rmsLinear > 0 ? peakLinear / rmsLinear : 0,
            silenceRatioPct: silence,
            quietAmbientPct: quiet,
            speechActivePct: active,
            loudSpeechPct: loud,
            spectralCentroidHz: centroid,
            subBassRumblePct: bandEnergy[0] / spectral * 100,
            lowMidVoicePct: bandEnergy[1] / spectral * 100,
            vocalSpeechBandPct: bandEnergy[2] / spectral * 100,
            highAirSibilancePct: bandEnergy[3] / spectral * 100,
            frameLoudnessDBFS: loudnessDBFS
        )
    }
}

nonisolated private extension AcousticAnalyzer {
    static var clipThresholdValue: Float { clipThreshold }
}

// MARK: - FFT

/// Reusable real-FFT front end. Holds the vDSP setup and Hann window so the
/// per-frame cost is just the transform itself.
nonisolated private final class FFTProcessor {
    private let size: Int
    private let halfSize: Int
    private let log2n: vDSP_Length
    private let setup: FFTSetup
    private var window: [Float]

    private var real: [Float]
    private var imaginary: [Float]
    private var windowed: [Float]

    init?(size: Int) {
        guard size > 0, size & (size - 1) == 0 else { return nil }
        self.size = size
        self.halfSize = size / 2
        self.log2n = vDSP_Length(log2(Double(size)).rounded())

        guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { return nil }
        self.setup = setup

        self.window = [Float](repeating: 0, count: size)
        vDSP_hann_window(&window, vDSP_Length(size), Int32(vDSP_HANN_NORM))

        self.real = [Float](repeating: 0, count: halfSize)
        self.imaginary = [Float](repeating: 0, count: halfSize)
        self.windowed = [Float](repeating: 0, count: size)
    }

    deinit { vDSP_destroy_fftsetup(setup) }

    /// Magnitude spectrum of one `size`-sample frame. Index `i` is centred on
    /// `i * sampleRate / size` Hz.
    func magnitudes(_ frame: ArraySlice<Float>) -> [Float] {
        precondition(frame.count == size, "FFT frame must be exactly \(size) samples")

        frame.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            vDSP_vmul(base, 1, window, 1, &windowed, 1, vDSP_Length(size))
        }

        var magnitudes = [Float](repeating: 0, count: halfSize)

        real.withUnsafeMutableBufferPointer { realPtr in
            imaginary.withUnsafeMutableBufferPointer { imagPtr in
                var split = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)

                windowed.withUnsafeBufferPointer { src in
                    src.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: halfSize) { reinterpreted in
                        vDSP_ctoz(reinterpreted, 2, &split, 1, vDSP_Length(halfSize))
                    }
                }

                vDSP_fft_zrip(setup, &split, 1, log2n, FFTDirection(FFT_FORWARD))

                // zrip packs the Nyquist term into imagp[0]; zeroing it keeps it
                // from contaminating the DC bin's magnitude.
                imagPtr[0] = 0

                vDSP_zvmags(&split, 1, &magnitudes, 1, vDSP_Length(halfSize))
            }
        }

        // vDSP_zvmags returns squared magnitude; take the root for a linear
        // magnitude spectrum, which is what the centroid definition expects.
        var count = Int32(halfSize)
        vvsqrtf(&magnitudes, magnitudes, &count)
        return magnitudes
    }
}
