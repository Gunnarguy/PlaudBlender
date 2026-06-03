import Foundation

struct AskChoice: Identifiable, Hashable, Sendable {
    let id: String
    let title: String
    let detail: String?

    init(_ id: String, title: String, detail: String? = nil) {
        self.id = id
        self.title = title
        self.detail = detail
    }
}

struct AskChip: Identifiable, Hashable, Sendable {
    let label: String
    let value: String

    var id: String { "\(label):\(value)" }
}

struct AskConversationTurn: Identifiable, Sendable {
    let id: String
    let question: String
    let answer: AIAnswer
    let supportingMoments: Int?

    init(question: String, answer: AIAnswer, supportingMoments: Int? = nil) {
        self.id = answer.responseId ?? UUID().uuidString
        self.question = question
        self.answer = answer
        self.supportingMoments = supportingMoments
    }
}

enum AskChronosPreset: String, CaseIterable, Identifiable, Sendable {
    case quick
    case balanced
    case deepDive

    var id: String { rawValue }

    var title: String {
        switch self {
        case .quick:
            return "Quick"
        case .balanced:
            return "Balanced"
        case .deepDive:
            return "Deep Dive"
        }
    }

    var subtitle: String {
        switch self {
        case .quick:
            return "Fast answer"
        case .balanced:
            return "Default feel"
        case .deepDive:
            return "More deliberate"
        }
    }

    func apply(to settings: inout AskChronosSettings) {
        switch self {
        case .quick:
            settings.model = "gpt-5.4-mini"
            settings.reasoning = "none"
            settings.verbosity = "low"
            settings.reasoningSummary = "off"
            settings.temperature = nil
            settings.topP = nil
            settings.maxOutputTokens = 768
            settings.serviceTier = "auto"
        case .balanced:
            settings = AskChronosSettings()
        case .deepDive:
            settings.model = "gpt-5.5-pro"
            settings.reasoning = "high"
            settings.verbosity = "high"
            settings.reasoningSummary = "auto"
            settings.temperature = nil
            settings.topP = nil
            settings.maxOutputTokens = 4096
            settings.serviceTier = "auto"
        }
    }
}

struct AskChronosSettings: Codable, Equatable, Sendable {
    var model: String = "gpt-5.5"
    var reasoning: String = "default"
    var verbosity: String = "default"
    var reasoningSummary: String = "off"
    var temperature: Double?
    var topP: Double?
    var maxOutputTokens: Int?
    var serviceTier: String = "auto"

    static let modelOptions: [AskChoice] = [
        AskChoice("gpt-5.5", title: "GPT-5.5", detail: "Best default"),
        AskChoice("gpt-5.5-pro", title: "GPT-5.5 Pro", detail: "Heavier reasoning"),
        AskChoice("gpt-5.4", title: "GPT-5.4", detail: "Stable fallback"),
        AskChoice("gpt-5.4-mini", title: "GPT-5.4 Mini", detail: "Faster and cheaper"),
        AskChoice("gpt-5.4-nano", title: "GPT-5.4 Nano", detail: "Lowest latency")
    ]

    static let reasoningOptions: [AskChoice] = [
        AskChoice("default", title: "Model default"),
        AskChoice("none", title: "None"),
        AskChoice("low", title: "Low"),
        AskChoice("medium", title: "Medium"),
        AskChoice("high", title: "High"),
        AskChoice("xhigh", title: "xHigh")
    ]

    static let verbosityOptions: [AskChoice] = [
        AskChoice("default", title: "Model default"),
        AskChoice("low", title: "Low"),
        AskChoice("medium", title: "Medium"),
        AskChoice("high", title: "High")
    ]

    static let reasoningSummaryOptions: [AskChoice] = [
        AskChoice("off", title: "Off"),
        AskChoice("auto", title: "Auto")
    ]

    static let serviceTierOptions: [AskChoice] = [
        AskChoice("auto", title: "Auto"),
        AskChoice("default", title: "Default"),
        AskChoice("flex", title: "Flex"),
        AskChoice("priority", title: "Priority")
    ]

    var supportsTemperatureControl: Bool {
        Self.supportsTemperatureControl(for: model)
    }

    func normalizedForModelCapabilities() -> AskChronosSettings {
        var normalized = self
        if !normalized.supportsTemperatureControl {
            normalized.temperature = nil
        }
        return normalized
    }

    static func supportsTemperatureControl(for model: String?) -> Bool {
        guard let model = model?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(), !model.isEmpty else {
            return true
        }
        return !model.hasPrefix("gpt-5")
    }

    var requestReasoning: String? {
        reasoning == "default" ? nil : reasoning
    }

    var requestVerbosity: String? {
        verbosity == "default" ? nil : verbosity
    }

    var requestReasoningSummary: String? {
        reasoningSummary == "off" ? nil : reasoningSummary
    }

    var requestTemperature: Double? {
        guard supportsTemperatureControl else { return nil }
        guard let temperature else { return nil }
        return min(max(temperature, 0.0), 2.0)
    }

    var requestTopP: Double? {
        guard let topP else { return nil }
        return min(max(topP, 0.0), 1.0)
    }

    var requestMaxOutputTokens: Int? {
        guard let maxOutputTokens else { return nil }
        return max(maxOutputTokens, 16)
    }

    var summaryChips: [AskChip] {
        var chips = [AskChip(label: "Model", value: displayTitle(for: model, in: Self.modelOptions))]
        if reasoning != "default" {
            chips.append(AskChip(label: "Reasoning", value: displayTitle(for: reasoning, in: Self.reasoningOptions)))
        }
        if verbosity != "default" {
            chips.append(AskChip(label: "Verbosity", value: displayTitle(for: verbosity, in: Self.verbosityOptions)))
        }
        if reasoningSummary != "off" {
            chips.append(AskChip(label: "Summary", value: displayTitle(for: reasoningSummary, in: Self.reasoningSummaryOptions)))
        }
        if let temperature = requestTemperature {
            chips.append(AskChip(label: "Temp", value: Self.format(number: temperature)))
        }
        if let topP = requestTopP {
            chips.append(AskChip(label: "Top-p", value: Self.format(number: topP)))
        }
        if let maxOutputTokens = requestMaxOutputTokens {
            chips.append(AskChip(label: "Max", value: "\(maxOutputTokens)"))
        }
        if serviceTier != "auto" {
            chips.append(AskChip(label: "Tier", value: displayTitle(for: serviceTier, in: Self.serviceTierOptions)))
        }
        return chips
    }

    func displayTitle(for id: String, in options: [AskChoice]) -> String {
        options.first(where: { $0.id == id })?.title ?? id
    }

    static func format(number: Double) -> String {
        String(format: "%.2f", number)
            .replacingOccurrences(of: #"\.?0+$"#, with: "", options: .regularExpression)
    }
}

// MARK: - Search Request / Response

/// Matches `SearchRequest` from the FastAPI backend.
struct SearchRequest: Codable, Sendable {
    let query: String
    var limit: Int = 50
    var categories: [String]?
    var startDate: String?
    var endDate: String?

    enum CodingKeys: String, CodingKey {
        case query, limit, categories
        case startDate = "start_date"
        case endDate = "end_date"
    }
}

/// Matches `SearchResultOut` from the FastAPI backend.
struct SearchResult: Codable, Identifiable, Sendable {
    let event: Event
    let score: Double
    let contextBefore: String?
    let contextAfter: String?

    var id: String { event.id }

    enum CodingKeys: String, CodingKey {
        case event, score
        case contextBefore = "context_before"
        case contextAfter = "context_after"
    }
}

/// Matches `SearchResponse` from the FastAPI backend.
struct SearchResponse: Codable, Sendable {
    let results: [SearchResult]
    let aiAnswer: AIAnswer?
    let total: Int

    enum CodingKeys: String, CodingKey {
        case results
        case aiAnswer = "ai_answer"
        case total
    }
}

struct AIAnswerConfig: Codable, Sendable, Hashable {
    let model: String?
    let reasoning: String?
    let reasoningSummary: String?
    let temperature: Double?
    let topP: Double?
    let maxOutputTokens: Int?
    let verbosity: String?
    let serviceTier: String?

    enum CodingKeys: String, CodingKey {
        case model, reasoning, temperature, verbosity
        case reasoningSummary = "reasoning_summary"
        case topP = "top_p"
        case maxOutputTokens = "max_output_tokens"
        case serviceTier = "service_tier"
    }

    init(
        model: String? = nil,
        reasoning: String? = nil,
        reasoningSummary: String? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        maxOutputTokens: Int? = nil,
        verbosity: String? = nil,
        serviceTier: String? = nil
    ) {
        self.model = model
        self.reasoning = reasoning
        self.reasoningSummary = reasoningSummary
        self.temperature = temperature
        self.topP = topP
        self.maxOutputTokens = maxOutputTokens
        self.verbosity = verbosity
        self.serviceTier = serviceTier
    }

    init(settings: AskChronosSettings) {
        self.init(
            model: settings.model,
            reasoning: settings.requestReasoning,
            reasoningSummary: settings.requestReasoningSummary,
            temperature: settings.requestTemperature,
            topP: settings.requestTopP,
            maxOutputTokens: settings.requestMaxOutputTokens,
            verbosity: settings.requestVerbosity,
            serviceTier: settings.serviceTier
        )
    }

    var chips: [AskChip] {
        var chips = [AskChip(label: "Model", value: model ?? "Unknown")]
        if let reasoning {
            chips.append(AskChip(label: "Reasoning", value: reasoning.capitalized))
        }
        if let verbosity {
            chips.append(AskChip(label: "Verbosity", value: verbosity.capitalized))
        }
        if let reasoningSummary {
            chips.append(AskChip(label: "Summary", value: reasoningSummary.capitalized))
        }
        if AskChronosSettings.supportsTemperatureControl(for: model), let temperature {
            chips.append(AskChip(label: "Temp", value: AskChronosSettings.format(number: temperature)))
        }
        if let topP {
            chips.append(AskChip(label: "Top-p", value: AskChronosSettings.format(number: topP)))
        }
        if let maxOutputTokens {
            chips.append(AskChip(label: "Max", value: "\(maxOutputTokens)"))
        }
        if let serviceTier, serviceTier != "auto" {
            chips.append(AskChip(label: "Tier", value: serviceTier.capitalized))
        }
        return chips
    }
}

/// Matches `AIAnswerOut` from the FastAPI backend.
struct AIAnswer: Codable, Sendable {
    let answer: String
    let model: String
    let responseId: String?
    let reasoningSummary: String?
    let config: AIAnswerConfig?
    let usage: [String: JSONValue]?

    enum CodingKeys: String, CodingKey {
        case answer, model, config, usage
        case responseId = "response_id"
        case reasoningSummary = "reasoning_summary"
    }

    var displayModel: String {
        config?.model ?? model
    }

    var totalTokens: Int? {
        usage?["total_tokens"]?.intValue
    }

    var reasoningTokens: Int? {
        usage?["reasoning_tokens"]?.intValue
    }

    var detailChips: [AskChip] {
        var chips = config?.chips ?? [AskChip(label: "Model", value: displayModel)]
        if let totalTokens {
            chips.append(AskChip(label: "Tokens", value: "\(totalTokens)"))
        }
        if let reasoningTokens, reasoningTokens > 0 {
            chips.append(AskChip(label: "Reasoning tokens", value: "\(reasoningTokens)"))
        }
        return chips
    }

    func applyingFallback(settings: AskChronosSettings) -> AIAnswer {
        AIAnswer(
            answer: answer,
            model: model.isEmpty ? settings.model : model,
            responseId: responseId,
            reasoningSummary: reasoningSummary,
            config: config ?? AIAnswerConfig(settings: settings),
            usage: usage
        )
    }
}

/// Matches `AskRequest` from the FastAPI backend.
struct AskRequest: Codable, Sendable {
    let question: String
    let previousResponseId: String?
    let model: String?
    var reasoning: String?

    var reasoningSummary: String?
    var temperature: Double?
    var topP: Double?
    var maxOutputTokens: Int?
    var verbosity: String?
    var serviceTier: String?

    enum CodingKeys: String, CodingKey {
        case question, model, reasoning, temperature, verbosity
        case previousResponseId = "previous_response_id"
        case reasoningSummary = "reasoning_summary"
        case topP = "top_p"
        case maxOutputTokens = "max_output_tokens"
        case serviceTier = "service_tier"
    }

    init(question: String, settings: AskChronosSettings, previousResponseId: String? = nil) {
        self.question = question
        self.previousResponseId = previousResponseId
        self.model = settings.model
        self.reasoning = settings.requestReasoning
        self.reasoningSummary = settings.requestReasoningSummary
        self.temperature = settings.requestTemperature
        self.topP = settings.requestTopP
        self.maxOutputTokens = settings.requestMaxOutputTokens
        self.verbosity = settings.requestVerbosity
        self.serviceTier = settings.serviceTier
    }
}
