import Foundation

/// A recording eligible for Plaud cloud upload.
/// Matches backend response: path, name, filename, size_mb, format, in_cloud
struct UploadCandidate: Codable, Identifiable, Sendable {
    let path: String
    let name: String
    let filename: String
    let sizeMb: Double
    let format: String
    let inCloud: Bool

    var id: String { path }

    enum CodingKeys: String, CodingKey {
        case path, name, filename, format
        case sizeMb = "size_mb"
        case inCloud = "in_cloud"
    }
}
