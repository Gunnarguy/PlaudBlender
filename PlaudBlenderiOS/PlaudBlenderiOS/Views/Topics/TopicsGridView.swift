import SwiftUI

struct TopicsGridView: View {
    let viewModel: TopicsViewModel
    @State private var selectedTopic: Topic?

    var body: some View {
        NavigationStack {
            Group {
                if viewModel.isLoading && viewModel.topics.isEmpty {
                    LoadingView(message: "Loading topics...")
                } else if viewModel.topics.isEmpty {
                    EmptyStateView(
                        icon: "tag",
                        title: "No Topics Yet",
                        message: "Topics are extracted from your recording events."
                    )
                } else {
                    topicsGrid
                }
            }
            .navigationTitle("Topics")
            .refreshable { await viewModel.refresh() }
            .task { await viewModel.loadTopics() }
            .sheet(item: $selectedTopic) { topic in
                TopicTimelineSheet(topic: topic, viewModel: viewModel)
            }
        }
    }

    private var topicsGrid: some View {
        ScrollView {
            if let error = viewModel.error {
                ErrorBanner(message: error)
            }

            LazyVGrid(columns: [
                GridItem(.adaptive(minimum: 100, maximum: 180))
            ], spacing: 10) {
                ForEach(viewModel.topics) { topic in
                    Button { selectedTopic = topic } label: {
                        topicCard(topic)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding()
        }
    }

    private func topicCard(_ topic: Topic) -> some View {
        VStack(spacing: 6) {
            Text(topic.name)
                .font(.subheadline)
                .fontWeight(.medium)
                .lineLimit(2)
                .multilineTextAlignment(.center)
            Text("\(topic.count)")
                .font(.title3)
                .fontWeight(.bold)
                .foregroundStyle(Color.accentPrimary)
            Text("occurrences")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

// MARK: - Topic Timeline Sheet

struct TopicTimelineSheet: View {
    let topic: Topic
    let viewModel: TopicsViewModel

    var body: some View {
        NavigationStack {
            Group {
                if let timeline = viewModel.selectedTimeline, timeline.topic == topic.name {
                    List(timeline.occurrences) { occ in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(occ.timestamp.iso8601Date?.shortDateString ?? occ.timestamp)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                Spacer()
                                CategoryPill(category: occ.category)
                            }
                            Text(occ.textSnippet)
                                .font(.subheadline)
                                .lineLimit(3)
                        }
                        .padding(.vertical, 4)
                    }
                } else {
                    LoadingView(message: "Loading timeline...")
                }
            }
            .navigationTitle(topic.name)
            .platformNavigationBarTitleDisplayModeInline()
            .task { await viewModel.loadTimeline(topic: topic.name) }
        }
        .presentationDetents([.medium, .large])
    }
}
