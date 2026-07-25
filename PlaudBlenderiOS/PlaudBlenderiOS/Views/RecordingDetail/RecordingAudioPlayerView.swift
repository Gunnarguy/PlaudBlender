import AVFoundation
import Observation
import SwiftUI

/// Playback state for a recording's audio.
///
/// The broker honors byte ranges, so AVPlayer seeks without downloading the
/// whole file. Auth travels through AVURLAsset's header options because the
/// endpoint requires a bearer token and AVURLAsset cannot take a URLRequest.
@Observable
final class RecordingAudioPlayerModel {
    var duration: Double = 0
    var currentTime: Double = 0
    var isPlaying = false
    var isScrubbing = false
    var loadFailed = false
    var isReady = false

    private var player: AVPlayer?
    private var timeObserver: Any?

    func load(recordingId: String, apiClient: APIClient) async {
        guard player == nil, !loadFailed else { return }
        guard let target = apiClient.streamingTarget(
            "/api/plaud/integrations/files/\(recordingId)/audio"
        ) else {
            loadFailed = true
            return
        }

        let asset = AVURLAsset(
            url: target.url,
            options: ["AVURLAssetHTTPHeaderFieldsKey": target.headers]
        )

        do {
            let seconds = CMTimeGetSeconds(try await asset.load(.duration))
            guard seconds.isFinite, seconds > 0 else {
                loadFailed = true
                return
            }
            duration = seconds
        } catch {
            loadFailed = true
            return
        }

        let newPlayer = AVPlayer(playerItem: AVPlayerItem(asset: asset))
        timeObserver = newPlayer.addPeriodicTimeObserver(
            forInterval: CMTime(seconds: 0.25, preferredTimescale: 600),
            queue: .main
        ) { [weak self] time in
            // Delivered on .main, so direct mutation is already main-thread safe.
            guard let self, !self.isScrubbing else { return }
            let seconds = CMTimeGetSeconds(time)
            if seconds.isFinite { self.currentTime = seconds }
        }
        player = newPlayer
        isReady = true
    }

    func togglePlayback() {
        guard let player else { return }
        if isPlaying {
            player.pause()
        } else {
            player.play()
        }
        isPlaying.toggle()
    }

    func seek(to seconds: Double) {
        player?.seek(
            to: CMTime(seconds: seconds, preferredTimescale: 600),
            toleranceBefore: .zero,
            toleranceAfter: .zero
        )
    }

    func teardown() {
        if let timeObserver { player?.removeTimeObserver(timeObserver) }
        timeObserver = nil
        player?.pause()
        player = nil
        isPlaying = false
        isReady = false
    }
}

/// Inline scrubbing player shown on the recording detail screen.
struct RecordingAudioPlayerView: View {
    let recordingId: String
    let apiClient: APIClient

    @State private var model = RecordingAudioPlayerModel()

    var body: some View {
        VStack(spacing: 4) {
            HStack(spacing: 12) {
                Button {
                    model.togglePlayback()
                } label: {
                    Image(systemName: model.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                        .font(.title)
                }
                .disabled(!model.isReady)

                VStack(spacing: 2) {
                    Slider(
                        value: Binding(
                            get: { model.currentTime },
                            set: { model.currentTime = $0 }
                        ),
                        in: 0...max(model.duration, 0.01),
                        onEditingChanged: { editing in
                            model.isScrubbing = editing
                            if !editing { model.seek(to: model.currentTime) }
                        }
                    )
                    .disabled(!model.isReady)

                    HStack {
                        Text(timeLabel(model.currentTime))
                        Spacer()
                        Text(timeLabel(model.duration))
                    }
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
                }
            }

            if model.loadFailed {
                Text("Audio unavailable")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal)
        .task { await model.load(recordingId: recordingId, apiClient: apiClient) }
        .onDisappear { model.teardown() }
    }

    private func timeLabel(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "0:00" }
        let total = Int(seconds)
        return String(format: "%d:%02d", total / 60, total % 60)
    }
}
