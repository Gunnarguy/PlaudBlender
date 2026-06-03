import Foundation

extension Date {
    /// ISO 8601 string representation.
    nonisolated var iso8601String: String {
        ISO8601DateFormatter().string(from: self)
    }

    /// Short date string like "Mar 21".
    nonisolated var shortDateString: String {
        let f = DateFormatter()
        f.dateFormat = "MMM d"
        return f.string(from: self)
    }

    /// Full date string like "March 21, 2026".
    nonisolated var fullDateString: String {
        let f = DateFormatter()
        f.dateStyle = .long
        return f.string(from: self)
    }

    /// Time string like "2:30 PM".
    nonisolated var timeString: String {
        let f = DateFormatter()
        f.timeStyle = .short
        return f.string(from: self)
    }

    /// Relative description like "2 hours ago".
    nonisolated var relativeString: String {
        let f = RelativeDateTimeFormatter()
        f.unitsStyle = .full
        return f.localizedString(for: self, relativeTo: Date())
    }
}

extension TimeInterval {
    /// Format seconds as "Xh Ym" or "Ym Zs".
    nonisolated var durationFormatted: String {
        let hours = Int(self) / 3600
        let minutes = (Int(self) % 3600) / 60
        let seconds = Int(self) % 60

        if hours > 0 {
            return "\(hours)h \(minutes)m"
        } else if minutes > 0 {
            return "\(minutes)m \(seconds)s"
        } else {
            return "\(seconds)s"
        }
    }
}

extension String {
    /// Parse an ISO 8601 date string to Date.
    nonisolated var iso8601Date: Date? {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f.date(from: self) ?? ISO8601DateFormatter().date(from: self)
    }
}
