import SwiftUI

/// Renders the Markdown subset that Plaud and Chronos summaries actually use:
/// ATX headings, bullet and numbered lists, horizontal rules, and inline emphasis.
///
/// `Text(LocalizedStringKey:)` on its own is not enough. It interprets *inline*
/// syntax (`**bold**`, `*italic*`, `` `code` ``) but drops block structure, so
/// `# Header Metadata` renders literally and every newline collapses into one
/// run-on line. We split into lines, strip each block marker, and style what is
/// left — falling back to `AttributedString` for the inline pass.
///
/// The view intentionally has no line limit: it grows to fit its content. For
/// compact one- or two-line previews use `String.markdownPlainPreview` instead.
struct MarkdownText: View {
    private let blocks: [Block]
    private let baseFont: Font

    init(_ markdown: String, baseFont: Font = .subheadline) {
        self.blocks = Block.parse(markdown)
        self.baseFont = baseFont
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                switch block {
                case let .heading(level, text):
                    Self.inline(text)
                        .font(headingFont(for: level))
                        .padding(.top, level <= 2 ? 4 : 2)

                case let .bullet(text):
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Text("•").font(baseFont)
                        Self.inline(text).font(baseFont)
                    }
                    .padding(.leading, 2)

                case let .numbered(marker, text):
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Text(marker).font(baseFont.weight(.medium))
                        Self.inline(text).font(baseFont)
                    }
                    .padding(.leading, 2)

                case let .paragraph(text):
                    Self.inline(text).font(baseFont)

                case .rule:
                    Divider().padding(.vertical, 1)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .multilineTextAlignment(.leading)
        // Let the text lay out at full height instead of being clipped by a
        // parent that offers an ambiguous vertical proposal.
        .fixedSize(horizontal: false, vertical: true)
    }

    private func headingFont(for level: Int) -> Font {
        switch level {
        case 1: return .headline
        case 2: return .subheadline.weight(.semibold)
        default: return baseFont.weight(.semibold)
        }
    }

    /// Inline pass — bold/italic/code/links only, preserving the whitespace we
    /// already decided is significant when we split the block.
    private static func inline(_ source: String) -> Text {
        var options = AttributedString.MarkdownParsingOptions()
        options.interpretedSyntax = .inlineOnlyPreservingWhitespace
        options.failurePolicy = .returnPartiallyParsedIfPossible
        if let attributed = try? AttributedString(markdown: source, options: options) {
            return Text(attributed)
        }
        return Text(source)
    }
}

// MARK: - Block parsing

private enum Block {
    case heading(level: Int, text: String)
    case bullet(String)
    case numbered(marker: String, text: String)
    case paragraph(String)
    case rule

    static func parse(_ raw: String) -> [Block] {
        var blocks: [Block] = []

        for rawLine in raw.replacingOccurrences(of: "\r\n", with: "\n").components(separatedBy: "\n") {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty else { continue }

            // `---` / `***` rules. Checked before bullets so `-` runs don't
            // parse as an empty list item.
            if line.count >= 3, line.allSatisfy({ $0 == "-" || $0 == "*" || $0 == "_" }) {
                blocks.append(.rule)
                continue
            }

            if let (level, text) = Self.heading(line) {
                blocks.append(.heading(level: level, text: text))
                continue
            }

            if let text = Self.bulletBody(line) {
                blocks.append(.bullet(text))
                continue
            }

            if let (marker, text) = Self.numberedBody(line) {
                blocks.append(.numbered(marker: marker, text: text))
                continue
            }

            blocks.append(.paragraph(line))
        }

        return blocks
    }

    private static func heading(_ line: String) -> (Int, String)? {
        let hashes = line.prefix(while: { $0 == "#" }).count
        guard (1...6).contains(hashes) else { return nil }
        let rest = String(line.dropFirst(hashes))
        // A heading marker must be followed by whitespace — this keeps `#hashtag`
        // as prose.
        guard rest.first?.isWhitespace == true else { return nil }
        let text = rest.trimmingCharacters(in: .whitespaces)
        return text.isEmpty ? nil : (hashes, text)
    }

    /// Requires "marker + space" so a bold run like `**Key Points List:**` is
    /// not mistaken for a bullet.
    private static func bulletBody(_ line: String) -> String? {
        guard let first = line.first, first == "-" || first == "*" || first == "+" else { return nil }
        let rest = line.dropFirst()
        guard rest.first?.isWhitespace == true else { return nil }
        let text = rest.trimmingCharacters(in: .whitespaces)
        return text.isEmpty ? nil : text
    }

    private static func numberedBody(_ line: String) -> (String, String)? {
        let digits = line.prefix(while: { $0.isNumber })
        guard !digits.isEmpty, digits.count <= 3 else { return nil }
        var rest = line.dropFirst(digits.count)
        guard let delimiter = rest.first, delimiter == "." || delimiter == ")" else { return nil }
        rest = rest.dropFirst()
        guard rest.first?.isWhitespace == true else { return nil }
        let text = rest.trimmingCharacters(in: .whitespaces)
        return text.isEmpty ? nil : ("\(digits)\(delimiter)", text)
    }
}

// MARK: - Compact previews

extension String {
    /// A short plain-prose preview of a Markdown summary.
    ///
    /// Plaud summaries open with a `# Header Metadata` block of `**Date:**` and
    /// `**Total Duration:**` rows. Those are scaffolding, not content, so merely
    /// stripping the syntax yields "Header Metadata Date: 2026-07-24 Total
    /// Duration: ~45 minutes …" — technically de-marked up, still unreadable.
    ///
    /// So we drop the scaffolding and prefer the summary's own themes row, which
    /// is the one line that says what actually happened. Mirrors
    /// `_summary_headline()` in `app_v2/services/data_service.py`, so a preview
    /// reads the same whether the text was condensed server-side or not.
    var markdownPlainPreview: String {
        var prose: [String] = []

        for rawLine in replacingOccurrences(of: "\r\n", with: "\n").components(separatedBy: "\n") {
            var line = rawLine.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty else { continue }

            // Rules and headings are structure, never content.
            if line.count >= 3, line.allSatisfy({ $0 == "-" || $0 == "*" || $0 == "_" }) { continue }
            if line.hasPrefix("#") { continue }

            // Strip a leading list marker, then inline emphasis.
            if let first = line.first, first == "-" || first == "*" || first == "+",
               line.dropFirst().first?.isWhitespace == true {
                line = String(line.dropFirst()).trimmingCharacters(in: .whitespaces)
            }
            line = line
                .replacingOccurrences(of: "**", with: "")
                .replacingOccurrences(of: "`", with: "")
                .trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty else { continue }

            let label = line.prefix(while: { $0 != ":" })
                .trimmingCharacters(in: .whitespaces)
                .lowercased()
            let value = line.contains(":")
                ? String(line.drop(while: { $0 != ":" }).dropFirst()).trimmingCharacters(in: .whitespaces)
                : ""

            if !value.isEmpty, Self.themeLabels.contains(label) { return value }
            if Self.scaffoldingLabels.contains(label) { continue }

            prose.append(line)
        }

        return prose.joined(separator: " ")
    }

    private static let themeLabels: Set<String> = [
        "primary contexts/themes", "primary contexts", "themes",
    ]

    private static let scaffoldingLabels: Set<String> = [
        "date", "total duration", "duration", "header metadata", "recording summary",
    ]
}
