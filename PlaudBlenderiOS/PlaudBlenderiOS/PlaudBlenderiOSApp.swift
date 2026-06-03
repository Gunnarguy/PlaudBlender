//
//  PlaudBlenderiOSApp.swift
//  PlaudBlenderiOS
//
//  Created by Gunnar Hostetler on 3/21/26.
//

import SwiftUI
import OSLog

private let logger = Logger(subsystem: "com.gunndamental.PlaudBlenderiOS", category: "App")

@main
struct PlaudBlenderiOSApp: App {
    @State private var authManager = AuthManager()
    @State private var apiClient: APIClient?
    @State private var notionViewModel: NotionViewModel?
    @State private var syncViewModel: SyncViewModel?
    @State private var xrayViewModel: XRayViewModel?

    var body: some Scene {
        WindowGroup {
            if let apiClient, let notionViewModel, let syncViewModel, let xrayViewModel {
                ContentView()
                    .environment(apiClient)
                    .environment(notionViewModel)
                    .environment(syncViewModel)
                    .environment(xrayViewModel)
                    .environment(authManager)
                    .task {
                        // Connectivity check on every app foreground
                        logger.info("🚀 App launched — checking server at \(authManager.serverURL, privacy: .public)")
                        let ok = await apiClient.healthCheck()
                        logger.info("🏥 Startup health: \(ok ? "CONNECTED" : "UNREACHABLE", privacy: .public)")
                        await xrayViewModel.bootstrapIfNeeded()
                    }
            } else {
                LoadingView(message: "Starting Chronos...")
                    .task {
                        guard apiClient == nil else { return }
                        logger.info("⏳ Creating APIClient for server: \(authManager.serverURL, privacy: .public)")
                        let client = APIClient(authManager: authManager)
                        _ = await client.bootstrapConnection()
                        let notion = NotionViewModel(api: client)
                        let sync = SyncViewModel(api: client)
                        let xray = XRayViewModel(api: client)
                        await sync.bootstrap()
                        xray.isPipelineActive = sync.isRunning
                        xray.isPipelineActive = sync.isRunning
                        await xray.bootstrapIfNeeded()
                        notionViewModel = notion
                        syncViewModel = sync
                        xrayViewModel = xray
                        apiClient = client
                    }
            }
        }
    }
}
