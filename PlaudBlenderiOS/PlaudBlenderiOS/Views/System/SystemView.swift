import SwiftUI

struct SystemView: View {
    @Environment(APIClient.self) private var api
    @Bindable var viewModel: SystemViewModel

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 12) {
                    overviewCard

                    ServiceStatusBar(
                        systemStatus: viewModel.systemStatus,
                        isLoading: viewModel.isLoading
                    ) {
                        await viewModel.refresh()
                    }
                    .background(.regularMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .padding(.horizontal)

                    runtimeManagerCard(viewModel.runtimeManagerInfo)

                    if let access = viewModel.runtimeSnapshot?.access,
                       access.preferredLabel != nil || !access.entryList.isEmpty {
                        accessCard(access)
                    }

                    if !viewModel.serviceEntries.isEmpty {
                        servicesCard
                    }

                    portsCard

                    signalsCard

                    if !viewModel.notes.isEmpty {
                        notesCard
                    }
                }
                .padding(.vertical, 8)
            }
            .navigationTitle("System")
            .refreshable { await viewModel.refresh() }
            .task { await viewModel.bootstrapIfNeeded() }
        }
    }

    private var overviewCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let error = viewModel.error {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.yellow)
                    Text(error)
                        .font(.caption)
                        .lineLimit(3)
                    Spacer()
                }
                .padding(10)
                .background(.red.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) {
                    statusTile(
                        title: "Connection",
                        state: api.isServerReachable ? "Reachable" : "Offline",
                        detail: api.resolvedServerURL,
                        ok: api.isServerReachable
                    )

                    statusTile(
                        title: "Runtime",
                        state: viewModel.runtimeStateText,
                        detail: viewModel.runtimeSummary,
                        ok: api.isServerReachable && viewModel.runtimeIsHealthy
                    )
                }

                VStack(spacing: 10) {
                    statusTile(
                        title: "Connection",
                        state: api.isServerReachable ? "Reachable" : "Offline",
                        detail: api.resolvedServerURL,
                        ok: api.isServerReachable
                    )

                    statusTile(
                        title: "Runtime",
                        state: viewModel.runtimeStateText,
                        detail: viewModel.runtimeSummary,
                        ok: api.isServerReachable && viewModel.runtimeIsHealthy
                    )
                }
            }

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                metricTile("Source", viewModel.runtimeSourceLabel)
                metricTile("Manager", viewModel.managerName)
                metricTile("Plaud", viewModel.plaudAuthSummary)
                metricTile("Updated", viewModel.lastUpdated?.relativeString ?? "Just now")
            }

            if let notice = viewModel.runtimeNotice {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "info.circle.fill")
                        .foregroundStyle(.blue)
                    Text(notice)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
                .padding(10)
                .background(.blue.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            NavigationLink {
                XRayView()
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "waveform.path.ecg")
                        .foregroundStyle(Color.accentColor)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Open Live Telemetry")
                            .font(.caption.weight(.semibold))
                        Text("Inspect the full X-ray stream behind recent operational signals.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Image(systemName: "chevron.right")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.tertiary)
                }
                .padding(10)
                .background(.thinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 10))
            }
            .buttonStyle(.plain)
        }
        .padding()
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .padding(.horizontal)
    }

    private func runtimeManagerCard(_ manager: SystemRuntimeManagerInfo) -> some View {
        contentCard(title: "Runtime Manager", subtitle: viewModel.managerDetail) {
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                metricTile("Name", manager.name)
                metricTile("Mode", manager.mode)
                metricTile("State", manager.state)
                metricTile("Watchdog", manager.watchdog)
                metricTile("Verified", manager.verified)
                metricTile("Version", manager.version)
            }
        }
    }

    private func accessCard(_ access: SystemRuntimeSnapshot.RuntimeAccess) -> some View {
        contentCard(
            title: "Access Paths",
            subtitle: "Preferred ways to reach the live Chronos stack from the backend runtime"
        ) {
            VStack(alignment: .leading, spacing: 10) {
                if let preferredLabel = access.preferredLabel, !preferredLabel.isEmpty {
                    HStack(alignment: .top, spacing: 10) {
                        Image(systemName: "network")
                            .foregroundStyle(Color.accentColor)
                            .frame(width: 18)
                            .padding(.top, 2)

                        VStack(alignment: .leading, spacing: 3) {
                            Text("Preferred Route")
                                .font(.caption.weight(.medium))
                                .foregroundStyle(.secondary)
                            Text(preferredLabel)
                                .font(.subheadline.weight(.semibold))

                            if let preferredUIURL = access.preferredUIURL, !preferredUIURL.isEmpty {
                                Text(preferredUIURL)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .textSelection(.enabled)
                            }

                            if let preferredAPIURL = access.preferredAPIURL,
                               !preferredAPIURL.isEmpty,
                               preferredAPIURL != access.preferredUIURL {
                                Text(preferredAPIURL)
                                    .font(.caption2)
                                    .foregroundStyle(.tertiary)
                                    .textSelection(.enabled)
                            }
                        }
                    }
                }

                if !access.entryList.isEmpty {
                    VStack(spacing: 10) {
                        ForEach(access.entryList) { entry in
                            VStack(alignment: .leading, spacing: 4) {
                                HStack(alignment: .firstTextBaseline) {
                                    Text(entry.label)
                                        .font(.subheadline.weight(.semibold))
                                    Spacer()
                                    if let kind = entry.kind, !kind.isEmpty {
                                        Text(kind.displayLabel)
                                            .font(.caption2.weight(.semibold))
                                            .foregroundStyle(.secondary)
                                    }
                                }

                                Text(entry.url)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .textSelection(.enabled)
                            }
                        }
                    }
                }
            }
        }
    }

    private var servicesCard: some View {
        contentCard(
            title: "Service States",
            subtitle: viewModel.runtimeSnapshot?.serviceList.isEmpty == false
                ? "Runtime-level service health from the backend"
                : "Fallback integration health from /api/status"
        ) {
            VStack(spacing: 10) {
                ForEach(viewModel.serviceEntries) { service in
                    HStack(alignment: .top, spacing: 10) {
                        Circle()
                            .fill(service.isHealthy ? .green : .red)
                            .frame(width: 10, height: 10)
                            .padding(.top, 4)

                        VStack(alignment: .leading, spacing: 3) {
                            HStack(alignment: .firstTextBaseline) {
                                Text(service.title)
                                    .font(.subheadline.weight(.semibold))
                                Spacer()
                                Text(service.state)
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(service.isHealthy ? .green : .red)
                            }

                            if let detail = service.detail, !detail.isEmpty {
                                Text(detail)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }

                            if let footnote = service.footnote, !footnote.isEmpty {
                                Text(footnote)
                                    .font(.caption2)
                                    .foregroundStyle(.tertiary)
                            }
                        }
                    }
                }
            }
        }
    }

    private var portsCard: some View {
        contentCard(title: "Ports", subtitle: viewModel.portSourceLabel) {
            if viewModel.portEntries.isEmpty {
                Text("No port information is available yet.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                VStack(spacing: 10) {
                    ForEach(viewModel.portEntries) { port in
                        HStack(alignment: .top, spacing: 10) {
                            Image(systemName: portSymbol(for: port))
                                .foregroundStyle(portColor(for: port))
                                .frame(width: 18)
                                .padding(.top, 2)

                            VStack(alignment: .leading, spacing: 3) {
                                HStack(alignment: .firstTextBaseline) {
                                    Text(port.name)
                                        .font(.subheadline.weight(.semibold))
                                    Spacer()
                                    Text(port.summary)
                                        .font(.caption.weight(.medium).monospaced())
                                        .foregroundStyle(.secondary)
                                }

                                if let detail = port.detail, !detail.isEmpty {
                                    Text(detail)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private var signalsCard: some View {
        contentCard(title: "Recent Signals", subtitle: viewModel.signalSourceLabel) {
            if viewModel.signalEntries.isEmpty {
                Text("No recent operational events yet.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                VStack(spacing: 10) {
                    ForEach(Array(viewModel.signalEntries.prefix(8))) { signal in
                        HStack(alignment: .top, spacing: 10) {
                            Image(systemName: signalSymbol(for: signal.level))
                                .foregroundStyle(signalColor(for: signal.level))
                                .frame(width: 18)
                                .padding(.top, 2)

                            VStack(alignment: .leading, spacing: 3) {
                                HStack(alignment: .firstTextBaseline) {
                                    Text(signal.title)
                                        .font(.subheadline.weight(.semibold))
                                    Spacer()
                                    Text(signal.source)
                                        .font(.caption2.weight(.semibold))
                                        .foregroundStyle(.secondary)
                                }

                                Text(signal.message)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)

                                if let footnote = signal.footnote, !footnote.isEmpty {
                                    Text(footnote)
                                        .font(.caption2)
                                        .foregroundStyle(.tertiary)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private var notesCard: some View {
        contentCard(title: "Runtime Notes", subtitle: nil) {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(viewModel.notes, id: \.self) { note in
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "circle.fill")
                            .font(.system(size: 6))
                            .foregroundStyle(.secondary)
                            .padding(.top, 6)
                        Text(note)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    private func contentCard<Content: View>(title: String, subtitle: String?, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.headline)

            if let subtitle, !subtitle.isEmpty {
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            content()
        }
        .padding()
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .padding(.horizontal)
    }

    private func statusTile(title: String, state: String, detail: String, ok: Bool) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)

            HStack(spacing: 6) {
                Circle()
                    .fill(ok ? .green : .red)
                    .frame(width: 9, height: 9)
                Text(state)
                    .font(.headline)
            }

            Text(detail)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(ok ? .green.opacity(0.08) : .red.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func metricTile(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.caption2.weight(.medium))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption.weight(.semibold))
                .lineLimit(2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private func signalSymbol(for level: String) -> String {
        switch level {
        case "error", "critical", "fail":
            return "xmark.octagon.fill"
        case "warn", "warning":
            return "exclamationmark.triangle.fill"
        case "ok", "success":
            return "checkmark.circle.fill"
        default:
            return "info.circle.fill"
        }
    }

    private func signalColor(for level: String) -> Color {
        switch level {
        case "error", "critical", "fail":
            return .red
        case "warn", "warning":
            return .orange
        case "ok", "success":
            return .green
        default:
            return .blue
        }
    }

    private func portSymbol(for port: SystemPortEntry) -> String {
        switch port.isReachable {
        case true:
            return "dot.radiowaves.left.and.right"
        case false:
            return "wifi.slash"
        case nil:
            return "ellipsis.circle"
        }
    }

    private func portColor(for port: SystemPortEntry) -> Color {
        switch port.isReachable {
        case true:
            return .green
        case false:
            return .red
        case nil:
            return .secondary
        }
    }
}

private extension String {
    var displayLabel: String {
        replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")
            .capitalized
    }
}
