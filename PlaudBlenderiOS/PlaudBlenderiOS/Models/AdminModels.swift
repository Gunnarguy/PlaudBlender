import Foundation

struct StackControlResponse: Codable, Sendable {
    let action: String
    let status: String
    let message: String
    let output: String
    let publicURL: String?

    enum CodingKeys: String, CodingKey {
        case action, status, message, output
        case publicURL = "public_url"
    }
}

struct SyncFailureItem: Codable, Identifiable, Sendable {
    let recordingId: String?
    let source: String?
    let title: String?
    let error: String
    let reason: String?

    var id: String {
        recordingId ?? [title, reason, error].compactMap { $0 }.joined(separator: "|")
    }

    enum CodingKeys: String, CodingKey {
        case recordingId = "recording_id"
        case source, title, error, reason
    }
}

struct SyncFailureSummary: Codable, Sendable {
    let actionableCount: Int
    let archivedCount: Int
    let actionable: [SyncFailureItem]
    let archived: [SyncFailureItem]

    enum CodingKeys: String, CodingKey {
        case actionableCount = "actionable_count"
        case archivedCount = "archived_count"
        case actionable, archived
    }
}

struct UploadProcessRequest: Codable, Sendable {
    var filePaths: [String]?
    var templateId: String?
    var model: String = "openai"

    enum CodingKeys: String, CodingKey {
        case filePaths = "file_paths"
        case templateId = "template_id"
        case model
    }
}

struct UploadProcessItem: Codable, Identifiable, Sendable {
    let path: String
    let fileId: String?
    let workflowId: String?
    let error: String?

    var id: String { path }

    enum CodingKeys: String, CodingKey {
        case path
        case fileId = "file_id"
        case workflowId = "workflow_id"
        case error
    }
}

struct UploadProcessResult: Codable, Sendable {
    let uploadedCount: Int
    let errorCount: Int
    let uploaded: [UploadProcessItem]
    let errors: [UploadProcessItem]

    enum CodingKeys: String, CodingKey {
        case uploadedCount = "uploaded_count"
        case errorCount = "error_count"
        case uploaded, errors
    }
}

struct AdminBackupInfo: Codable, Identifiable, Sendable {
    let filename: String
    let createdAt: String
    let sizeBytes: Int
    let downloadPath: String
    let message: String

    var id: String { filename }

    enum CodingKeys: String, CodingKey {
        case filename, message
        case createdAt = "created_at"
        case sizeBytes = "size_bytes"
        case downloadPath = "download_path"
    }
}
