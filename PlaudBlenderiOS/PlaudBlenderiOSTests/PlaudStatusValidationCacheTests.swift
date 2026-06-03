import Foundation
import XCTest
@testable import PlaudBlenderiOS

@MainActor
final class PlaudStatusValidationCacheTests: XCTestCase {
    func testFreshValidationIsCachedAndDoesNotFetchTwice() async throws {
        var cache = PlaudStatusValidationCache()
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let status = TokenStatus(
            isAuthenticated: true,
            hasAccessToken: true,
            expiresAt: isoString(now.addingTimeInterval(2 * 3600)),
            workspaceName: "Chronos",
            extra: nil
        )

        var fetchCount = 0

        _ = try await cache.load(forceRefresh: false, now: now) {
            fetchCount += 1
            return status
        }

        let reused = try await cache.load(forceRefresh: false, now: now.addingTimeInterval(90)) {
            fetchCount += 1
            return status
        }

        XCTAssertEqual(fetchCount, 1)
        XCTAssertEqual(reused.status?.isAuthenticated, true)
        XCTAssertEqual(reused.diagnostics.state, .reusedCachedValidation)
    }

    func testTransientValidationFailureKeepsUsableCachedSession() {
        var cache = PlaudStatusValidationCache()
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let status = TokenStatus(
            isAuthenticated: true,
            hasAccessToken: true,
            expiresAt: isoString(now.addingTimeInterval(90 * 60)),
            workspaceName: nil,
            extra: nil
        )

        _ = cache.recordServerStatus(status, now: now)
        let recovered = cache.recoverFromTransientFailure(
            APIError.httpError(status: 429, body: "rate limited"),
            now: now.addingTimeInterval(120)
        )

        XCTAssertEqual(recovered?.status?.isAuthenticated, true)
        XCTAssertEqual(recovered?.diagnostics.state, .usedCachedAfterTransientFailure)
    }

    func testHardFailureClearsCachedSessionAndRequiresReconnect() {
        var cache = PlaudStatusValidationCache()
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let status = TokenStatus(
            isAuthenticated: true,
            hasAccessToken: true,
            expiresAt: isoString(now.addingTimeInterval(3600)),
            workspaceName: nil,
            extra: nil
        )

        _ = cache.recordServerStatus(status, now: now)
        let invalidated = cache.requireReauthentication(
            message: "Reconnect Plaud.",
            now: now.addingTimeInterval(30)
        )

        XCTAssertNil(cache.cachedStatus)
        XCTAssertNil(invalidated.status)
        XCTAssertEqual(invalidated.diagnostics.state, .requiresReauthentication)
    }

    private func isoString(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: date)
    }
}
