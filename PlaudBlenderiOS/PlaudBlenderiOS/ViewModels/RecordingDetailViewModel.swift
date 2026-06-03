import Foundation
import Observation

@Observable
final class RecordingDetailViewModel {
    var detail: RecordingDetail?
    var isLoading = false
    var error: String?

    private let api: APIClient

    init(api: APIClient) {
        self.api = api
    }

    func loadDetail(recordingId: String) async {
        isLoading = true
        error = nil
        do {
            detail = try await Self.loadFully(api: api, recordingId: recordingId)
        } catch {
            self.error = error.localizedDescription
        }
        isLoading = false
    }

    func overrideCategory(eventId: String, newCategory: String) async -> Bool {
        Haptics.selectionChanged()
        guard let rid = detail?.summary.recordingId else {
            self.error = "No recording loaded"
            return false
        }
        do {
            let body = CategoryOverrideRequest(category: newCategory)
            let _: SuccessResponse = try await api.put(
                "/api/recordings/\(rid)/events/\(eventId)/category",
                body: body
            )
            await loadDetail(recordingId: rid)
            return true
        } catch {
            self.error = error.localizedDescription
            return false
        }
    }

    func runWorkflow(recordingId: String, templateId: String?, model: String) async -> Bool {
        Haptics.impact()
        do {
            let body = RecordingWorkflowRequest(templateId: templateId, model: model)
            let _: SuccessResponse = try await api.post(
                "/api/sync/workflows/\(recordingId)",
                body: body
            )
            return true
        } catch {
            self.error = error.localizedDescription
            return false
        }
    }

    static func loadFully(api: APIClient, recordingId: String) async throws -> RecordingDetail {
        let base: RecordingDetail = try await api.get("/api/recordings/\(recordingId)")

        let transcriptResponse: TranscriptResponse? = try? await api.get(
            "/api/recordings/\(recordingId)/transcript"
        )
        let aiSummaryResponse: AISummaryResponse? = try? await api.get(
            "/api/recordings/\(recordingId)/ai-summary"
        )
        let plaudTranscriptResponse: PlaudTranscriptResponse? = try? await api.get(
            "/api/recordings/\(recordingId)/plaud-transcript"
        )
        let extractedDataResponse: ExtractedDataResponse? = try? await api.get(
            "/api/recordings/\(recordingId)/extracted-data"
        )
        let processing: RecordingProcessing? = try? await api.get(
            "/api/recordings/\(recordingId)/processing"
        )

        let transcript = nonEmpty(base.transcript) ?? nonEmpty(transcriptResponse?.transcript)
        let plaudAiSummary = nonEmpty(base.summary.plaudAiSummary) ?? nonEmpty(aiSummaryResponse?.aiSummary)
        let plaudTranscript = nonEmpty(base.plaudTranscript) ?? nonEmpty(plaudTranscriptResponse?.plaudTranscript)
        let extractedData = (base.extractedData?.isEmpty == false)
            ? base.extractedData
            : extractedDataResponse?.extractedData

        let mergedSummary = RecordingSummary(
            recordingId: base.summary.recordingId,
            startTime: base.summary.startTime,
            createdAt: base.summary.createdAt,
            endTime: base.summary.endTime,
            durationSeconds: base.summary.durationSeconds,
            durationFormatted: base.summary.durationFormatted,
            topCategory: base.summary.topCategory,
            eventCount: base.summary.eventCount,
            timeRangeFormatted: base.summary.timeRangeFormatted,
            timeIsEstimated: base.summary.timeIsEstimated,
            timeEstimateReason: base.summary.timeEstimateReason,
            title: base.summary.title,
            plaudAiSummary: plaudAiSummary,
            cloudStatus: base.summary.cloudStatus,
            source: base.summary.source,
            processingStatus: base.summary.processingStatus,
            hasPlaudAI: base.summary.hasPlaudAI,
            plaudWorkflowStatus: base.summary.plaudWorkflowStatus,
            notionState: base.summary.notionState,
            notionPageId: base.summary.notionPageId,
            notionPageURL: base.summary.notionPageURL,
            notionPageTitle: base.summary.notionPageTitle,
            notionMatchCount: base.summary.notionMatchCount
        )

        return RecordingDetail(
            summary: mergedSummary,
            events: base.events,
            categoryPercentages: base.categoryPercentages,
            transcript: transcript,
            aiSummary: base.aiSummary,
            extractedData: extractedData,
            workflowStatus: base.workflowStatus,
            plaudTranscript: plaudTranscript,
            processing: processing ?? base.processing
        )
    }

    private static func nonEmpty(_ value: String?) -> String? {
        guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines), !trimmed.isEmpty else {
            return nil
        }
        return trimmed
    }
}

private struct TranscriptResponse: Decodable {
    let transcript: String
}

private struct AISummaryResponse: Decodable {
    let aiSummary: String

    enum CodingKeys: String, CodingKey {
        case aiSummary = "ai_summary"
    }
}

private struct PlaudTranscriptResponse: Decodable {
    let plaudTranscript: String

    enum CodingKeys: String, CodingKey {
        case plaudTranscript = "plaud_transcript"
    }
}

private struct ExtractedDataResponse: Decodable {
    let extractedData: [String: JSONValue]

    enum CodingKeys: String, CodingKey {
        case extractedData = "extracted_data"
    }
}
