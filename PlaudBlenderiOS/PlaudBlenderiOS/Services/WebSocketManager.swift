import Foundation
import Observation
import OSLog

private let logger = Logger(subsystem: "com.gunndamental.PlaudBlenderiOS", category: "WebSocket")

/// Manages a persistent WebSocket connection for X-ray telemetry events.
@Observable
final class WebSocketManager {
    var isConnected = false
    var events: [XRayEvent] = []
    var latestSeq: Int = 0

    private var wsTask: URLSessionWebSocketTask?
    private var isListening = false
    private let maxEvents = 500
    @ObservationIgnored var onEvent: (@MainActor (XRayEvent) -> Void)?

    func connect(baseURL: String, token: String?) {
        let wsURL = baseURL
            .replacingOccurrences(of: "http://", with: "ws://")
            .replacingOccurrences(of: "https://", with: "wss://")
        guard let url = URL(string: "\(wsURL)/api/v1/xray/ws") else { return }

        var request = URLRequest(url: url)
        if let token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        request.setValue("true", forHTTPHeaderField: "ngrok-skip-browser-warning")

        wsTask = URLSession.shared.webSocketTask(with: request)
        wsTask?.resume()
        isConnected = true
        receiveLoop()
    }

    func disconnect() {
        wsTask?.cancel(with: .goingAway, reason: nil)
        wsTask = nil
        isConnected = false
        isListening = false
    }

    private func receiveLoop() {
        guard isConnected else { return }
        isListening = true

        wsTask?.receive { [weak self] result in
            guard let self else { return }

            switch result {
            case .success(let message):
                if case .string(let text) = message,
                   let data = text.data(using: .utf8) {
                    Task { @MainActor in
                        do {
                            if let heartbeat = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                               heartbeat["type"] as? String == "heartbeat" {
                                return
                            }
                            let event = try JSONDecoder().decode(XRayEvent.self, from: data)
                            self.events.insert(event, at: 0)
                            if self.events.count > self.maxEvents {
                                self.events.removeLast()
                            }
                            self.latestSeq = event.seq
                            self.onEvent?(event)
                        } catch {
                            logger.error("Failed to decode XRayEvent: \(error.localizedDescription, privacy: .public) — raw: \(text.prefix(200), privacy: .public)")
                        }
                    }
                }
                self.receiveLoop()

            case .failure(let error):
                logger.error("WebSocket receive error: \(error.localizedDescription, privacy: .public)")
                Task { @MainActor in
                    self.isConnected = false
                    self.isListening = false
                }
            }
        }
    }
}
