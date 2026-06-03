import Foundation
import Observation
import OSLog

private let logger = Logger(subsystem: "com.gunndamental.PlaudBlenderiOS", category: "Auth")

/// Manages authentication state and token storage via Keychain.
@Observable
final class AuthManager: Sendable {
    private static let apiKeyKey = "chronos_api_key"
    private static let serverURLKey = "chronos_server_url"
    private static let serverURLInfoPlistKey = "ChronosServerURL"
    private static let allowLoopbackLaunchArgument = "-PlaudBlenderAllowLoopbackServer"

    private static let overrides: [String: String] = {
        guard let url = Bundle.main.url(forResource: "LocalOverrides", withExtension: "plist"),
              let data = try? Data(contentsOf: url),
              let plist = try? PropertyListSerialization.propertyList(from: data, options: [], format: nil) as? [String: String] else {
            return [:]
        }
        return plist
    }()

    /// Pi's Tailscale MagicDNS host for remote access when the iPhone is on the tailnet.
    private static var piTailscaleMagicDNSURL: String {
        overrides["piTailscaleMagicDNSURL"] ?? "http://your-device.your-tailnet.ts.net:8000"
    }
    /// Pi's stable Tailscale IP fallback when MagicDNS is unavailable on the client.
    private static var piTailscaleURL: String {
        overrides["piTailscaleURL"] ?? "http://100.x.y.z:8000"
    }
    /// Pi's known LAN IP for fast local access on home Wi-Fi.
    private static var piLanURL: String {
        overrides["piLanURL"] ?? "http://10.x.y.z:8000"
    }
    /// Legacy Mac-local recovery URL from earlier debugging sessions.
    private static var legacyMacRecoveryHost: String {
        overrides["legacyMacRecoveryHost"] ?? "10.x.y.z"
    }
    /// Legacy temporary ngrok API URL from earlier debugging sessions.
    private static var legacyNgrokRecoveryHost: String {
        overrides["legacyNgrokRecoveryHost"] ?? "your-ngrok-domain.ngrok-free.app"
    }
    /// Reserved ngrok tunnel — public fallback backend URL for the Pi API.
    private static var ngrokURL: String {
        overrides["ngrokURL"] ?? "https://your-ngrok-domain.ngrok-free.dev"
    }

    /// Default server URL — uses Info.plist value (ngrok), then falls back.
    static var defaultServerURL: String {
        if let configuredServerURL = Self.configuredServerURL {
            return configuredServerURL
        }
        return ngrokURL
    }

    var isAuthenticated: Bool {
        getToken() != nil
    }

    var serverURL: String {
        if let storedURL = Self.preferredServerURL(from: KeychainService.load(key: Self.serverURLKey)) {
            return storedURL
        }
        return Self.defaultServerURL
    }

    func getToken() -> String? {
        KeychainService.load(key: Self.apiKeyKey)
    }

    func setToken(_ token: String) throws {
        try KeychainService.save(key: Self.apiKeyKey, value: token)
    }

    func setServerURL(_ url: String) throws {
        let normalized = Self.preferredServerURL(from: url) ?? Self.defaultServerURL
        try KeychainService.save(key: Self.serverURLKey, value: normalized)
    }

    func candidateServerURLs() -> [String] {
        var candidates: [String] = []

        let storedURL = Self.preferredServerURL(from: KeychainService.load(key: Self.serverURLKey))
        let configuredServerURL = Self.configuredServerURL

        // 1. User-stored URL (from Settings) when it is a true override.
        if let storedURL, !Self.isBuiltInServerURL(storedURL) {
            candidates.append(storedURL)
        }

        // 2. Info.plist configured URL when it is not one of the built-in fallbacks.
        if let configuredServerURL, !Self.isBuiltInServerURL(configuredServerURL) {
            candidates.append(configuredServerURL)
        }

        // 3. Prefer direct Pi access over Tailscale before public tunneling.
        candidates.append(Self.piTailscaleMagicDNSURL)
        candidates.append(Self.piTailscaleURL)

        // 4. Public Pi API tunnel (works when Tailscale is unavailable and ngrok quota permits)
        candidates.append(Self.ngrokURL)

        // 5. Pi's LAN IP (faster when on home Wi-Fi)
        candidates.append(Self.piLanURL)

        var seen = Set<String>()
        return candidates.filter { seen.insert($0).inserted }
    }

    func clearServerURL() {
        KeychainService.delete(key: Self.serverURLKey)
    }

    func logout() {
        KeychainService.delete(key: Self.apiKeyKey)
    }

    // MARK: - Server Resolution

    private static var configuredServerURL: String? {
        normalizedServerURL(Bundle.main.object(forInfoDictionaryKey: serverURLInfoPlistKey) as? String)
    }

    private static func preferredServerURL(from value: String?) -> String? {
        guard let normalized = normalizedServerURL(value) else {
            return nil
        }

        if isLoopbackURL(normalized) {
            return allowsLoopbackServerURL ? normalized : configuredServerURL
        }

        return normalized
    }

    private static var allowsLoopbackServerURL: Bool {
        #if DEBUG
        ProcessInfo.processInfo.arguments.contains(allowLoopbackLaunchArgument)
        #else
        false
        #endif
    }

    private static func normalizedServerURL(_ value: String?) -> String? {
        guard var rawValue = value?.trimmingCharacters(in: .whitespacesAndNewlines), !rawValue.isEmpty else {
            return nil
        }

        if !rawValue.contains("://") {
            rawValue = "http://\(rawValue)"
        }

        guard let components = URLComponents(string: rawValue),
              let scheme = components.scheme?.lowercased(),
              ["http", "https"].contains(scheme),
              let host = components.host,
              !host.isEmpty else {
            return nil
        }

        if host == legacyMacRecoveryHost || host == legacyNgrokRecoveryHost {
            return ngrokURL
        }

        var normalized = "\(scheme)://\(host)"
        if let port = components.port {
            normalized += ":\(port)"
        }
        return normalized
    }

    private static func isLoopbackURL(_ value: String) -> Bool {
        guard let host = URLComponents(string: value)?.host?.lowercased() else {
            return false
        }

        return host == "localhost" || host == "127.0.0.1" || host == "::1"
    }

    private static func isBuiltInServerURL(_ value: String) -> Bool {
        [
            piTailscaleMagicDNSURL,
            piTailscaleURL,
            ngrokURL,
            piLanURL,
        ].contains(value)
    }
}
