import Foundation

/// Matches `XRayEventOut` from the FastAPI backend.
struct XRayEvent: Codable, Identifiable, Sendable {
    let seq: Int
    let ts: Double
    let source: String
    let op: String
    let message: String
    let durationMs: Double?
    let detail: String?
    let level: String
    let runId: String?
    let spanId: String?
    let parentSpanId: String?
    let recordingId: String?
    let eventId: String?
    let stage: String?
    let provider: String?
    let model: String?
    let status: String?
    let inputTokens: Int?
    let outputTokens: Int?
    let costUsd: Double?
    let requestId: String?

    var id: Int { seq }

    var date: Date {
        Date(timeIntervalSince1970: ts)
    }

    enum CodingKeys: String, CodingKey {
        case seq, ts, source, op, message
        case durationMs = "duration_ms"
        case detail, level, stage, provider, model, status
        case runId = "run_id"
        case spanId = "span_id"
        case parentSpanId = "parent_span_id"
        case recordingId = "recording_id"
        case eventId = "event_id"
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case costUsd = "cost_usd"
        case requestId = "request_id"
    }
}

/// Wrapper for `/api/xray/events` response.
struct XRayEventsResponse: Codable, Sendable {
    let events: [XRayEvent]
    let latestSeq: Int

    enum CodingKeys: String, CodingKey {
        case events
        case latestSeq = "latest_seq"
    }
}

/// Throughput response from `/api/xray/throughput`.
struct XRayThroughput: Codable, Sendable {
    let buckets: [Int]
}

/// Persisted execution run from `/api/xray/runs`.
struct TraceRun: Codable, Identifiable, Sendable, Hashable {
    let runId: String
    let trigger: String?
    let source: String?
    let status: String
    let title: String?
    let startedAt: String?
    let endedAt: String?
    let durationMs: Double?
    let summary: [String: JSONValue]?
    let metadata: [String: JSONValue]?
    let errorMessage: String?

    var id: String { runId }

    var startedDate: Date? {
        startedAt?.iso8601Date
    }

    enum CodingKeys: String, CodingKey {
        case trigger, source, status, title, summary, metadata
        case runId = "run_id"
        case startedAt = "started_at"
        case endedAt = "ended_at"
        case durationMs = "duration_ms"
        case errorMessage = "error_message"
    }
}

/// Persisted granular execution span from `/api/xray/spans`.
struct TraceSpan: Codable, Identifiable, Sendable, Hashable {
    let spanId: String
    let runId: String?
    let parentSpanId: String?
    let recordingId: String?
    let eventId: String?
    let stage: String?
    let operation: String
    let source: String?
    let provider: String?
    let model: String?
    let status: String
    let level: String
    let message: String?
    let detail: String?
    let startedAt: String?
    let endedAt: String?
    let durationMs: Double?
    let inputHash: String?
    let outputHash: String?
    let inputSnippet: String?
    let outputSnippet: String?
    let inputTokens: Int?
    let outputTokens: Int?
    let costUsd: Double?
    let requestId: String?
    let retryCount: Int
    let metadata: [String: JSONValue]?
    let errorMessage: String?

    var id: String { spanId }

    var startedDate: Date? {
        startedAt?.iso8601Date
    }

    enum CodingKeys: String, CodingKey {
        case stage, operation, source, provider, model, status, level, message, detail, metadata
        case spanId = "span_id"
        case runId = "run_id"
        case parentSpanId = "parent_span_id"
        case recordingId = "recording_id"
        case eventId = "event_id"
        case startedAt = "started_at"
        case endedAt = "ended_at"
        case durationMs = "duration_ms"
        case inputHash = "input_hash"
        case outputHash = "output_hash"
        case inputSnippet = "input_snippet"
        case outputSnippet = "output_snippet"
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case costUsd = "cost_usd"
        case requestId = "request_id"
        case retryCount = "retry_count"
        case errorMessage = "error_message"
    }
}

struct TraceDag: Codable, Sendable {
    let run: TraceRun?
    let spans: [TraceSpan]
}

struct RecordingProcessing: Codable, Sendable {
    let recordingId: String
    let status: String
    let latestError: String?
    let runs: [TraceRun]
    let spans: [TraceSpan]
    let totals: [String: JSONValue]

    enum CodingKeys: String, CodingKey {
        case status, runs, spans, totals
        case recordingId = "recording_id"
        case latestError = "latest_error"
    }
}
