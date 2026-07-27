import AVFoundation
import Foundation
import Speech

/// A recording already held in the corpus that an imported file might be.
nonisolated struct TranscriptCandidate: Identifiable, Hashable, Sendable {
    let id: String
    let title: String
    let durationSeconds: Double
    let startAt: Date?
}

/// The result of trying to recognise an imported file as something already
/// transcribed.
nonisolated enum TranscriptMatchOutcome: Sendable {
    /// Nothing in the corpus plausibly corresponds to this file.
    case none
    /// Exactly one candidate fits. Safe to attach, with `reason` shown.
    case unique(TranscriptCandidate, reason: String)
    /// Several candidates fit equally well. The caller must ask rather than
    /// guess -- picking one silently attaches the wrong transcript.
    case ambiguous([TranscriptCandidate], reason: String)
}

/// Recognises an imported audio file as a recording already in the corpus, so
/// its transcript can be reused instead of re-derived.
///
/// Deliberately refuses to guess. The corpus holds the same sessions ingested
/// from two Plaud apps that do not sync with each other, so duplicate
/// durations *and* duplicate start times both occur -- measured on real data,
/// two pairs collided exactly. Where the evidence is ambiguous this reports it
/// as ambiguous and lets the user decide.
nonisolated enum TranscriptMatcher {

    /// Seconds of duration difference still treated as the same recording.
    /// Corpus durations are whole seconds while the analyzer measures to the
    /// millisecond, so some slack is required even for a true match.
    static let defaultTolerance: Double = 2

    static func match(
        fileName: String,
        durationSeconds: Double,
        in candidates: [TranscriptCandidate],
        tolerance: Double = defaultTolerance
    ) -> TranscriptMatchOutcome {
        guard !candidates.isEmpty else { return .none }

        // Tier 1 -- the file name carries a recording id. Decisive when present.
        let haystack = fileName.lowercased()
        let byName = candidates.filter { candidate in
            // Ids may be namespaced ("notion:abc123"); match the bare part too.
            let bare = candidate.id.split(separator: ":").last.map(String.init) ?? candidate.id
            return bare.count >= 8 && haystack.contains(bare.lowercased())
        }
        if byName.count == 1 {
            return .unique(byName[0], reason: "file name contains this recording's id")
        }

        // Tier 2 -- duration.
        let near = candidates
            .map { (candidate: $0, delta: abs($0.durationSeconds - durationSeconds)) }
            .filter { $0.delta <= tolerance }
            .sorted { $0.delta < $1.delta }

        guard let closest = near.first else { return .none }

        if near.count == 1 {
            return .unique(
                closest.candidate,
                reason: "duration matches within \(String(format: "%.1f", closest.delta)) s"
            )
        }

        return .ambiguous(
            near.map(\.candidate),
            reason: "\(near.count) recordings share this duration — pick the right one"
        )
    }
}

// MARK: - On-device transcription

nonisolated enum AudioTranscriberError: LocalizedError {
    case noSupportedLocale
    case unreadableFile(String)

    var errorDescription: String? {
        switch self {
        case .noSupportedLocale:
            return "On-device transcription is not available for this device's language."
        case .unreadableFile(let reason):
            return "Could not read audio for transcription: \(reason)"
        }
    }
}

/// Transcribes an audio file entirely on device via `SpeechAnalyzer`.
///
/// Used only when `TranscriptMatcher` finds nothing to reuse -- transcription
/// costs real time on a multi-hour file, so an existing transcript always wins.
nonisolated enum AudioTranscriber {

    /// A locale the transcriber supports, preferring the device's own.
    static func resolveLocale(preferring preferred: Locale = .current) async -> Locale? {
        let supported = await SpeechTranscriber.supportedLocales
        guard !supported.isEmpty else { return nil }

        if let exact = supported.first(where: { $0.identifier == preferred.identifier }) {
            return exact
        }
        // Fall back to the same language in another region, then to English.
        if let language = preferred.language.languageCode?.identifier,
           let sameLanguage = supported.first(where: { $0.language.languageCode?.identifier == language }) {
            return sameLanguage
        }
        return supported.first(where: { $0.language.languageCode?.identifier == "en" }) ?? supported.first
    }

    /// True when the model for `locale` is already on device, so the caller can
    /// warn before a download.
    static func isInstalled(_ locale: Locale) async -> Bool {
        await SpeechTranscriber.installedLocales.contains { $0.identifier == locale.identifier }
    }

    /// Transcribe `url`, downloading the language model first if needed.
    ///
    /// `progress` reports 0...1 based on how far through the file the analyzer
    /// has reached.
    static func transcribe(
        url: URL,
        locale: Locale,
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> String {
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }

        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw AudioTranscriberError.unreadableFile(error.localizedDescription)
        }

        let totalSeconds = file.length > 0 && file.fileFormat.sampleRate > 0
            ? Double(file.length) / file.fileFormat.sampleRate
            : 0

        let transcriber = SpeechTranscriber(locale: locale, preset: .transcription)

        // Pull the model down if this locale has never been used before.
        if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
            try await request.downloadAndInstall()
        }

        let analyzer = SpeechAnalyzer(modules: [transcriber])

        // Consume results concurrently with analysis; the sequence finishes when
        // the analyzer is finalized.
        let collector = Task { () -> String in
            var assembled = AttributedString()
            for try await result in transcriber.results where result.isFinal {
                assembled.append(result.text)
                if totalSeconds > 0 {
                    let seconds = result.range.end.seconds
                    if seconds.isFinite {
                        progress?(min(1, max(0, seconds / totalSeconds)))
                    }
                }
            }
            return String(assembled.characters)
        }

        do {
            _ = try await analyzer.analyzeSequence(from: file)
            try await analyzer.finalizeAndFinishThroughEndOfInput()
        } catch {
            collector.cancel()
            throw error
        }

        let text = try await collector.value
        progress?(1)
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
