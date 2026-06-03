# Security Policy: PlaudBlenderiOS

This document outlines the security architecture, credential handling procedures, network boundaries, and logging policies of the native PlaudBlenderiOS client application.

---

## 1. Supported Versions and Status

Security patches and updates are actively released for the following versions:

| Version | Supported | Notes |
|---|---|---|
| `0.8.x` | Yes | Current beta release version. |
| `< 0.8.0`| No | Legacy development prototypes. |

---

## 2. Secret Storage Model

All authentication secrets, custom tokens, and server URLs are isolated from plaintext files:
- **Keychain Storage**: API tokens (`chronos_api_key`) and server URL config overrides (`chronos_server_url`) are persisted using Apple's Keychain Services.
- **Keychain Security Attributes**: Access tags are marked as `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly` to ensure credential decryption is restricted to the specific device and blocked until the passcode is entered following a restart.
- **UserDefaults Restriction**: Plain text UserDefaults arrays are restricted to non-sensitive runtime parameters (like the user's selected app tab or light/dark theme preference).

---

## 3. Network Transport Security

- **Enforced Transport Security (ATS)**: Outbound requests require HTTPS connections.
- **ngrok Interstitial Bypass**: APIClient injects `ngrok-skip-browser-warning = true` headers on all connection probes. This prevents the ngrok gateway from injecting browser warning HTML pages, which would cause JSON decoding loops on the Swift compiler.
- **Tailscale Encrypted Transit**: Requests traversing local tailnet addresses (e.g. `100.x.y.z`) are encrypted inside Tailscale's WireGuard tunnels automatically.

---

## 4. Log Sanitation Policy

PlaudBlenderiOS implements filters inside [APIClient.swift](PlaudBlenderiOS/Services/APIClient.swift) to ensure runtime telemetry displays do not leak user keys:
- **Redaction Arrays**: Header parameters including `Authorization`, `Cookie`, `Set-Cookie`, and `X-API-Key` are automatically replaced with a `<redacted>` tag before being appended to the `ClientNetworkEvent` history.
- **OSLog Privacy Options**: OSLog strings containing dynamic server target IPs are declared as `privacy: .public` for networking debug ease, while credentials and token mappings are declared as `privacy: .private`.

```swift
// Example log filter implementation in APIClient.swift:
private func sanitizedHeaders(from headers: [AnyHashable: Any]) -> [String: String] {
    var sanitized: [String: String] = [:]
    for (key, value) in headers {
        let headerKey = String(describing: key)
        if ["authorization", "cookie", "set-cookie", "x-api-key"].contains(headerKey.lowercased()) {
            sanitized[headerKey] = "<redacted>"
        } else {
            sanitized[headerKey] = String(describing: value)
        }
    }
    return sanitized
}
```

---

## 5. Vulnerability Reporting Process

If you discover a security vulnerability within PlaudBlenderiOS:
1. **Reporting channel**: Contact Gunnar Hostetler directly or open a secure issue using specific GPG keys if present.
2. **Review period**: Issues will be reviewed and mitigated within a 30-day window.
3. **Disclosure**: Details are published alongside standard release notes on github.

---

## 6. Security Checklist for Future Changes

When contributing or refactoring source paths, you must adhere to these directives:
- [ ] **No Secrets in Code**: Never hardcode API keys, developer passwords, or raw test URLs in Swift source views or build configurations.
- [ ] **Check Log Additions**: Ensure any new print or OSLog command does not output raw token variables.
- [ ] **Secure WebView Config**: Disable file system access inside WKWebView configuration instances unless explicitly required.
- [ ] **Validate Keychain Key Strings**: Ensure all new Keychain lookups match predefined security keys and are not stored in plaintext user domains.
