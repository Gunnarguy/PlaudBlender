import SwiftUI

struct SystemView: View {
    @Environment(APIClient.self) private var api
    @Bindable var viewModel: SystemViewModel

    var body: some View {
        NavigationStack {
            ZStack {
                Color(hex: "070a12")
                    .ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 16) {
                        overviewCard

                        ServiceStatusBar(
                            systemStatus: viewModel.systemStatus,
                            isLoading: viewModel.isLoading
                        ) {
                            await viewModel.refresh()
                        }
                        .background(Color(hex: "0f172a").opacity(0.8))
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                        .overlay(
                            RoundedRectangle(cornerRadius: 16)
                                .stroke(Color.white.opacity(0.08), lineWidth: 1)
                        )
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
                    .padding(.vertical, 12)
                }
            }
            .navigationTitle("System")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.visible, for: .navigationBar)
            .toolbarBackground(Color(hex: "070a12"), for: .navigationBar)
            .refreshable { await viewModel.refresh() }
            .task { await viewModel.bootstrapIfNeeded() }
        }
    }

    private var overviewCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                HStack(spacing: 8) {
                    Circle()
                        .fill(api.isServerReachable ? Color.emeraldGreen : Color.roseRed)
                        .frame(width: 10, height: 10)
                        .shadow(color: (api.isServerReachable ? Color.emeraldGreen : Color.roseRed).opacity(0.8), radius: 6)

                    Text("RUNTIME STATUS")
                        .font(.caption.weight(.heavy))
                        .foregroundStyle(
                            LinearGradient(
                                colors: [.cyan, .purple],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .tracking(1.2)
                }

                Spacer()

                Text(viewModel.lastUpdated?.relativeString ?? "Live")
                    .font(.caption2.weight(.bold))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.white.opacity(0.06))
                    .clipShape(Capsule())
            }

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
                .background(Color.red.opacity(0.15))
                .clipShape(RoundedRectangle(cornerRadius: 10))
            }

            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) {
                    statusTile(
                        title: "Backend Link",
                        state: api.isServerReachable ? "ONLINE" : "OFFLINE",
                        detail: api.resolvedServerURL,
                        ok: api.isServerReachable
                    )

                    statusTile(
                        title: "Daemon Process",
                        state: viewModel.runtimeStateText.uppercased(),
                        detail: viewModel.runtimeSummary,
                        ok: api.isServerReachable && viewModel.runtimeIsHealthy
                    )
                }
            }

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                metricTile("Source", viewModel.runtimeSourceLabel)
                metricTile("Manager", viewModel.managerName)
                metricTile("Plaud OAuth", viewModel.plaudAuthSummary)
                metricTile("Uptime", "100% Stable")
            }

            if let notice = viewModel.runtimeNotice {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "info.circle.fill")
                        .foregroundStyle(.cyan)
                    Text(notice)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
                .padding(10)
                .background(Color.cyan.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 10))
            }

            NavigationLink {
                XRayView()
            } label: {
                HStack(spacing: 10) {
                    Image(systemName: "waveform.path.ecg")
                        .font(.title3)
                        .foregroundStyle(.cyan)
                        .shadow(color: .cyan.opacity(0.6), radius: 6)

                    VStack(alignment: .leading, spacing: 2) {
                        Text("Live System Telemetry")
                            .font(.subheadline.weight(.bold))
                            .foregroundStyle(.white)
                        Text("Inspect real-time trace events, model latency, and token stream")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    Image(systemName: "chevron.right")
                        .font(.caption.weight(.bold))
                        .foregroundStyle(.cyan)
                }
                .padding(12)
                .background(
                    LinearGradient(
                        colors: [Color.cyan.opacity(0.12), Color.purple.opacity(0.12)],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                )
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.cyan.opacity(0.3), lineWidth: 1)
                )
            }
            .buttonStyle(.plain)
        }
        .padding(16)
        .background(Color(hex: "0f172a").opacity(0.9))
        .clipShape(RoundedRectangle(cornerRadius: 20))
        .overlay(
            RoundedRectangle(cornerRadius: 20)
                .stroke(Color.white.opacity(0.08), lineWidth: 1)
        )
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
            title: "Access Routes",
            subtitle: "Network entry points to the live Chronos engine"
        ) {
            VStack(alignment: .leading, spacing: 10) {
                if let preferredLabel = access.preferredLabel, !preferredLabel.isEmpty {
                    HStack(alignment: .top, spacing: 10) {
                        Image(systemName: "network")
                            .foregroundStyle(.cyan)
                            .frame(width: 18)
                            .padding(.top, 2)

                        VStack(alignment: .leading, spacing: 3) {
                            Text("Preferred Route")
                                .font(.caption.weight(.medium))
                                .foregroundStyle(.secondary)
                            Text(preferredLabel)
                                .font(.subheadline.weight(.bold))
                                .foregroundStyle(.white)

                            if let preferredUIURL = access.preferredUIURL, !preferredUIURL.isEmpty {
                                Text(preferredUIURL)
                                    .font(.caption)
                                    .monospaced()
                                    .foregroundStyle(.cyan)
                                    .textSelection(.enabled)
                            }
                        }
                    }
                }

                if !access.entryList.isEmpty {
                    VStack(spacing: 8) {
                        ForEach(access.entryList) { entry in
                            VStack(alignment: .leading, spacing: 4) {
                                HStack(alignment: .firstTextBaseline) {
                                    Text(entry.label)
                                        .font(.subheadline.weight(.semibold))
                                        .foregroundStyle(.white)
                                    Spacer()
                                    if let kind = entry.kind, !kind.isEmpty {
                                        Text(kind.displayLabel)
                                            .font(.caption2.weight(.bold))
                                            .foregroundStyle(.cyan)
                                    }
                                }

                                Text(entry.url)
                                    .font(.caption)
                                    .monospaced()
                                    .foregroundStyle(.secondary)
                                    .textSelection(.enabled)
                            }
                            .padding(8)
                            .background(Color.white.opacity(0.04))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                    }
                }
            }
        }
    }

    private var servicesCard: some View {
        contentCard(
            title: "Active System Services",
            subtitle: "Daemon process health from systemd"
        ) {
            VStack(spacing: 10) {
                ForEach(viewModel.serviceEntries) { service in
                    HStack(alignment: .top, spacing: 12) {
                        Circle()
                            .fill(service.isHealthy ? Color.emeraldGreen : Color.roseRed)
                            .frame(width: 10, height: 10)
                            .shadow(color: (service.isHealthy ? Color.emeraldGreen : Color.roseRed).opacity(0.8), radius: 6)
                            .padding(.top, 4)

                        VStack(alignment: .leading, spacing: 3) {
                            HStack(alignment: .firstTextBaseline) {
                                Text(service.title)
                                    .font(.subheadline.weight(.bold))
                                    .foregroundStyle(.white)
                                Spacer()
                                Text(service.state.uppercased())
                                    .font(.caption2.weight(.heavy))
                                    .foregroundStyle(service.isHealthy ? Color.emeraldGreen : Color.roseRed)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background((service.isHealthy ? Color.emeraldGreen : Color.roseRed).opacity(0.15))
                                    .clipShape(Capsule())
                            }

                            if let detail = service.detail, !detail.isEmpty {
                                Text(detail)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    .padding(10)
                    .background(Color.white.opacity(0.04))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                }
            }
        }
    }

    private var portsCard: some View {
        contentCard(title: "Exposed Ports", subtitle: viewModel.portSourceLabel) {
            if viewModel.portEntries.isEmpty {
                Text("No port information available.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                    ForEach(viewModel.portEntries) { port in
                        HStack(spacing: 8) {
                            Image(systemName: portSymbol(for: port))
                                .foregroundStyle(portColor(for: port))
                                .font(.caption.weight(.bold))

                            VStack(alignment: .leading, spacing: 2) {
                                Text(port.name)
                                    .font(.caption.weight(.bold))
                                    .foregroundStyle(.white)
                                Text(port.summary)
                                    .font(.caption2.weight(.semibold).monospaced())
                                    .foregroundStyle(.cyan)
                            }
                            Spacer()
                        }
                        .padding(10)
                        .background(Color.white.opacity(0.04))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    }
                }
            }
        }
    }

    private var signalsCard: some View {
        contentCard(title: "Recent Signals", subtitle: viewModel.signalSourceLabel) {
            if viewModel.signalEntries.isEmpty {
                Text("No recent operational events.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                VStack(spacing: 8) {
                    ForEach(Array(viewModel.signalEntries.prefix(6))) { signal in
                        HStack(alignment: .top, spacing: 10) {
                            Image(systemName: signalSymbol(for: signal.level))
                                .foregroundStyle(signalColor(for: signal.level))
                                .frame(width: 16)
                                .padding(.top, 2)

                            VStack(alignment: .leading, spacing: 2) {
                                HStack {
                                    Text(signal.title)
                                        .font(.caption.weight(.bold))
                                        .foregroundStyle(.white)
                                    Spacer()
                                    Text(signal.source)
                                        .font(.caption2)
                                        .foregroundStyle(.tertiary)
                                }

                                Text(signal.message)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .padding(8)
                        .background(Color.white.opacity(0.03))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                }
            }
        }
    }

    private var notesCard: some View {
        contentCard(title: "Runtime Notes", subtitle: nil) {
            VStack(alignment: .leading, spacing: 6) {
                ForEach(viewModel.notes, id: \.self) { note in
                    HStack(alignment: .top, spacing: 8) {
                        Circle()
                            .fill(Color.cyan)
                            .frame(width: 5, height: 5)
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
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.headline.weight(.bold))
                    .foregroundStyle(.white)

                if let subtitle, !subtitle.isEmpty {
                    Text(subtitle)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            content()
        }
        .padding(16)
        .background(Color(hex: "0f172a").opacity(0.85))
        .clipShape(RoundedRectangle(cornerRadius: 20))
        .overlay(
            RoundedRectangle(cornerRadius: 20)
                .stroke(Color.white.opacity(0.08), lineWidth: 1)
        )
        .padding(.horizontal)
    }

    private func statusTile(title: String, state: String, detail: String, ok: Bool) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption2.weight(.bold))
                .foregroundStyle(.secondary)

            HStack(spacing: 6) {
                Circle()
                    .fill(ok ? Color.emeraldGreen : Color.roseRed)
                    .frame(width: 8, height: 8)
                    .shadow(color: (ok ? Color.emeraldGreen : Color.roseRed).opacity(0.8), radius: 6)

                Text(state)
                    .font(.subheadline.weight(.heavy))
                    .foregroundStyle(ok ? Color.emeraldGreen : Color.roseRed)
            }

            Text(detail)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background((ok ? Color.emeraldGreen : Color.roseRed).opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .stroke((ok ? Color.emeraldGreen : Color.roseRed).opacity(0.3), lineWidth: 1)
        )
    }

    private func metricTile(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.caption2.weight(.medium))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption.weight(.bold))
                .foregroundStyle(.white)
                .lineLimit(2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(Color.white.opacity(0.04))
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
            return .roseRed
        case "warn", "warning":
            return .orange
        case "ok", "success":
            return .emeraldGreen
        default:
            return .cyan
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
            return .emeraldGreen
        case false:
            return .roseRed
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
