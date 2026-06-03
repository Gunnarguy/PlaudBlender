import Foundation

struct ServerSettingsFlags: Codable, Sendable {
    var hasGeminiAPIKey: Bool = false
    var hasOpenAIAPIKey: Bool = false
    var hasQdrantAPIKey: Bool = false
    var hasNotionToken: Bool = false
    var hasNotionOAuth: Bool = false

    enum CodingKeys: String, CodingKey {
        case hasGeminiAPIKey = "has_gemini_api_key"
        case hasOpenAIAPIKey = "has_openai_api_key"
        case hasQdrantAPIKey = "has_qdrant_api_key"
        case hasNotionToken = "has_notion_token"
        case hasNotionOAuth = "has_notion_oauth"
    }
}

struct ServerSettings: Codable, Sendable {
    var processingProvider: String
    var cleaningModel: String
    var analystModel: String
    var embeddingModel: String
    var openAIModel: String
    var thinkingLevel: String
    var openAITemperature: Double
    var embeddingDim: Int
    var plaudLanguage: String
    var plaudDiarization: Bool
    var logLevel: String
    var customCategories: String
    var notionWeekdayStart: String
    var notionWeekendStart: String
    var qdrantURL: String
    var qdrantCollectionName: String
    var chronosOpenAIEnabled: Bool
    var chronosLocalLLMEnabled: Bool
    var chronosLocalLLMProvider: String
    var chronosLocalLLMBaseURL: String
    var chronosLocalLLMModel: String
    var chronosLocalLLMMaxContext: Int
    var chronosLocalLLMAllowedTasks: String
    var flags: ServerSettingsFlags

    enum CodingKeys: String, CodingKey {
        case processingProvider = "processing_provider"
        case cleaningModel = "cleaning_model"
        case analystModel = "analyst_model"
        case embeddingModel = "embedding_model"
        case openAIModel = "openai_model"
        case thinkingLevel = "thinking_level"
        case openAITemperature = "openai_temperature"
        case embeddingDim = "embedding_dim"
        case plaudLanguage = "plaud_language"
        case plaudDiarization = "plaud_diarization"
        case logLevel = "log_level"
        case customCategories = "custom_categories"
        case notionWeekdayStart = "notion_weekday_start"
        case notionWeekendStart = "notion_weekend_start"
        case qdrantURL = "qdrant_url"
        case qdrantCollectionName = "qdrant_collection_name"
        case chronosOpenAIEnabled = "chronos_openai_enabled"
        case chronosLocalLLMEnabled = "chronos_local_llm_enabled"
        case chronosLocalLLMProvider = "chronos_local_llm_provider"
        case chronosLocalLLMBaseURL = "chronos_local_llm_base_url"
        case chronosLocalLLMModel = "chronos_local_llm_model"
        case chronosLocalLLMMaxContext = "chronos_local_llm_max_context"
        case chronosLocalLLMAllowedTasks = "chronos_local_llm_allowed_tasks"
        case flags
    }
}

struct ServerSettingsUpdateRequest: Codable, Sendable {
    var processingProvider: String
    var cleaningModel: String
    var analystModel: String
    var embeddingModel: String
    var openAIModel: String
    var thinkingLevel: String
    var openAITemperature: Double
    var embeddingDim: Int
    var plaudLanguage: String
    var plaudDiarization: Bool
    var logLevel: String
    var customCategories: String
    var notionWeekdayStart: String
    var notionWeekendStart: String
    var qdrantURL: String
    var qdrantCollectionName: String
    var chronosOpenAIEnabled: Bool
    var chronosLocalLLMEnabled: Bool
    var chronosLocalLLMProvider: String
    var chronosLocalLLMBaseURL: String
    var chronosLocalLLMModel: String
    var chronosLocalLLMMaxContext: Int
    var chronosLocalLLMAllowedTasks: String

    enum CodingKeys: String, CodingKey {
        case processingProvider = "processing_provider"
        case cleaningModel = "cleaning_model"
        case analystModel = "analyst_model"
        case embeddingModel = "embedding_model"
        case openAIModel = "openai_model"
        case thinkingLevel = "thinking_level"
        case openAITemperature = "openai_temperature"
        case embeddingDim = "embedding_dim"
        case plaudLanguage = "plaud_language"
        case plaudDiarization = "plaud_diarization"
        case logLevel = "log_level"
        case customCategories = "custom_categories"
        case notionWeekdayStart = "notion_weekday_start"
        case notionWeekendStart = "notion_weekend_start"
        case qdrantURL = "qdrant_url"
        case qdrantCollectionName = "qdrant_collection_name"
        case chronosOpenAIEnabled = "chronos_openai_enabled"
        case chronosLocalLLMEnabled = "chronos_local_llm_enabled"
        case chronosLocalLLMProvider = "chronos_local_llm_provider"
        case chronosLocalLLMBaseURL = "chronos_local_llm_base_url"
        case chronosLocalLLMModel = "chronos_local_llm_model"
        case chronosLocalLLMMaxContext = "chronos_local_llm_max_context"
        case chronosLocalLLMAllowedTasks = "chronos_local_llm_allowed_tasks"
    }
}
