import Foundation
import XCTest
@testable import PlaudBlenderiOS

@MainActor
final class PlaudSyncDiagnosticsTests: XCTestCase {
    func testBackfillFailureDetailsRecognizeUnsupportedStageValidation() throws {
        let error = APIError.httpError(
            status: 422,
            body: #"{"detail":[{"type":"enum","loc":["body","stage"],"msg":"Input should be 'full', 'ingest', 'process', 'index', 'graph', or 'reindex'","input":"backfill"}]}"#
        )

        let details = try XCTUnwrap(SyncViewModel.backfillFailureDetails(from: error))

        XCTAssertEqual(details.statusCode, 422)
        XCTAssertTrue(details.indicatesUnsupportedStage)
        XCTAssertTrue(details.summary.contains("stage:"))
        XCTAssertTrue(details.summary.contains("backfill"))
        XCTAssertTrue(details.summary.contains("Input should"))
    }

    func testBackfillFailureDetailsPreserveOperationalBackendErrors() throws {
        let error = APIError.httpError(
            status: 400,
            body: #"{"detail":"Plaud token expired while requesting history."}"#
        )

        let details = try XCTUnwrap(SyncViewModel.backfillFailureDetails(from: error))

        XCTAssertEqual(details.statusCode, 400)
        XCTAssertEqual(details.summary, "Plaud token expired while requesting history.")
        XCTAssertFalse(details.indicatesUnsupportedStage)
    }

    func testBackfillPartialProgressNoticeIsSurfaced() {
        let status = PipelineStatus(
            status: "failed",
            runId: "run-1",
            currentPhase: "backfill",
            elapsedSeconds: 42,
            ageSeconds: nil,
            trigger: "manual",
            syncMode: "backfill",
            partialSuccess: true,
            message: "Rate limited on page 4, but preserved the pages fetched earlier.",
            detail: nil,
            warning: nil,
            warnings: ["Partial Plaud backfill preserved earlier pages."],
            phases: [
                PipelinePhase(
                    name: "backfill",
                    status: "failed",
                    totalItems: 4,
                    completedItems: 3,
                    currentStep: "page 4",
                    currentItem: "429",
                    elapsedSeconds: 42,
                    summary: "Preserved prior pages before stopping.",
                    error: "Rate limit while backfilling Plaud history.",
                    warnings: nil
                )
            ]
        )

        let notices = PlaudSyncNoticeBuilder.build(
            status: status,
            dbStats: RecordingDbStats(total: 80, completed: 80, pending: 0, processing: 0, failed: 1, noTranscript: 0),
            failures: nil
        )

        XCTAssertTrue(notices.contains(where: { $0.kind == .partialProgress && $0.message.contains("80") }))
    }

    func testRepeatedPageWarningIsSurfaced() {
        let status = PipelineStatus(
            status: "failed",
            runId: "run-2",
            currentPhase: "backfill",
            elapsedSeconds: 8,
            ageSeconds: nil,
            trigger: "manual",
            syncMode: "backfill",
            partialSuccess: true,
            message: nil,
            detail: nil,
            warning: "Repeated page signature detected on page 2; stopping backfill.",
            warnings: nil,
            phases: [
                PipelinePhase(
                    name: "backfill",
                    status: "failed",
                    totalItems: 2,
                    completedItems: 1,
                    currentStep: "page 2",
                    currentItem: "same ids",
                    elapsedSeconds: 8,
                    summary: "Detected the same recording page twice.",
                    error: "Repeated page signature during Plaud history sync.",
                    warnings: ["Stopping to avoid an infinite Plaud page loop."]
                )
            ]
        )

        let notices = PlaudSyncNoticeBuilder.build(
            status: status,
            dbStats: RecordingDbStats(total: 20, completed: 20, pending: 0, processing: 0, failed: 0, noTranscript: 0),
            failures: nil
        )

        XCTAssertTrue(notices.contains(where: { $0.kind == .repeatedPage }))
    }

    func testRecordingTimestampPrefersStartTimeOverCreatedAt() {
        let summary = RecordingSummary(
            recordingId: "rec-1",
            startTime: "2026-05-10T09:00:00Z",
            createdAt: "2026-05-10T11:15:00Z",
            endTime: "2026-05-10T09:45:00Z",
            durationSeconds: 2700,
            durationFormatted: "45m",
            topCategory: "work",
            eventCount: 12,
            timeRangeFormatted: "9:00 AM – 9:45 AM",
            timeIsEstimated: false,
            timeEstimateReason: nil,
            title: "Morning round-up",
            plaudAiSummary: nil,
            cloudStatus: "cloud",
            source: nil,
            processingStatus: nil,
            hasPlaudAI: nil,
            plaudWorkflowStatus: nil,
            notionState: nil,
            notionPageId: nil,
            notionPageURL: nil,
            notionPageTitle: nil,
            notionMatchCount: nil
        )

        XCTAssertEqual(summary.canonicalStartTime, summary.startTime)
        XCTAssertNotNil(summary.cloudCreatedLagDescription)
    }

    func testDaySummaryComparatorSortsNewestDaysFirst() throws {
        let json = #"""
        {
            "days": [
                {
                    "date": "2025-12-14",
                    "date_display": "December 14, 2025",
                    "total_duration_seconds": 1200,
                    "recording_count": 1,
                    "event_count": 4,
                    "recordings": []
                },
                {
                    "date": "2026-05-10",
                    "date_display": "May 10, 2026",
                    "total_duration_seconds": 5400,
                    "recording_count": 3,
                    "event_count": 21,
                    "recordings": []
                },
                {
                    "date": "2026-01-03",
                    "date_display": "January 3, 2026",
                    "total_duration_seconds": 1800,
                    "recording_count": 1,
                    "event_count": 7,
                    "recordings": []
                }
            ]
        }
        """#.data(using: .utf8)!

        let response = try JSONDecoder().decode(DaysResponse.self, from: json)
        let ordered = response.days.sorted(by: DaySummary.sortMostRecentFirst)

        XCTAssertEqual(ordered.map(\.date), ["2026-05-10", "2026-01-03", "2025-12-14"])
    }

        func testDaySummaryFilteringDropsSyntheticNotionOnlyDaysFromTimeline() throws {
                let json = #"""
                {
                    "days": [
                        {
                            "date": "2099-12-30",
                            "date_display": "Wednesday, Dec 30",
                            "total_duration_seconds": 1200,
                            "recording_count": 2,
                            "event_count": 4,
                            "recordings": [
                                {
                                    "recording_id": "notion:abc123",
                                    "start_time": "2099-12-30 07:30:00",
                                    "duration_seconds": 600,
                                    "top_category": "work",
                                    "event_count": 2,
                                    "title": "12-30 Future placeholder"
                                }
                            ]
                        },
                        {
                            "date": "2026-05-10",
                            "date_display": "May 10, 2026",
                            "total_duration_seconds": 1800,
                            "recording_count": 1,
                            "event_count": 8,
                            "recordings": [
                                {
                                    "recording_id": "rec-1",
                                    "start_time": "2026-05-10T09:00:00Z",
                                    "duration_seconds": 1800,
                                    "top_category": "work",
                                    "event_count": 8,
                                    "title": "Real recording"
                                }
                            ]
                        }
                    ]
                }
                """#.data(using: .utf8)!

                let response = try JSONDecoder().decode(DaysResponse.self, from: json)
                let visibleDays = response.days.filter(\.shouldDisplayInTimeline)

                XCTAssertEqual(visibleDays.map(\.date), ["2026-05-10"])
                XCTAssertEqual(response.days.first?.recordings?.isEmpty, true)
        }

    func testDaySummaryDecodingSortsRecordingsNewestFirst() throws {
        let json = #"""
        {
            "date": "2026-05-10",
            "date_display": "May 10, 2026",
            "total_duration_seconds": 5400,
            "recording_count": 3,
            "event_count": 21,
            "recordings": [
                {
                    "recording_id": "rec-middle",
                    "start_time": "2026-05-10T10:00:00Z",
                    "created_at": "2026-05-10T10:05:00Z",
                    "end_time": "2026-05-10T10:30:00Z",
                    "duration_seconds": 1800,
                    "top_category": "work",
                    "event_count": 8
                },
                {
                    "recording_id": "rec-newest",
                    "start_time": "2026-05-10T11:30:00Z",
                    "created_at": "2026-05-10T11:35:00Z",
                    "end_time": "2026-05-10T12:00:00Z",
                    "duration_seconds": 1800,
                    "top_category": "work",
                    "event_count": 7
                },
                {
                    "recording_id": "rec-created-fallback",
                    "created_at": "2026-05-10T12:15:00Z",
                    "end_time": "2026-05-10T12:45:00Z",
                    "duration_seconds": 1800,
                    "top_category": "work",
                    "event_count": 6
                }
            ]
        }
        """#.data(using: .utf8)!

        let day = try JSONDecoder().decode(DaySummary.self, from: json)

        XCTAssertEqual(day.recordings?.map(\.recordingId), ["rec-created-fallback", "rec-newest", "rec-middle"])
    }
}
