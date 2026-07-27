//
//  PlaudBlenderiOSApp.swift
//  PlaudBlenderiOS
//

import SwiftUI
import OSLog

private let logger = Logger(subsystem: "com.gunndamental.PlaudBlenderiOS", category: "App")

@main
struct PlaudBlenderiOSApp: App {
    @State private var authManager = AuthManager()
    @State private var apiClient: APIClient
    @State private var notionViewModel: NotionViewModel
    @State private var syncViewModel: SyncViewModel
    @State private var xrayViewModel: XRayViewModel

    init() {
        let auth = AuthManager()
        let client = APIClient(authManager: auth)
        self._authManager = State(initialValue: auth)
        self._apiClient = State(initialValue: client)
        self._notionViewModel = State(initialValue: NotionViewModel(api: client))
        self._syncViewModel = State(initialValue: SyncViewModel(api: client))
        self._xrayViewModel = State(initialValue: XRayViewModel(api: client))
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .preferredColorScheme(.dark)
                .environment(apiClient)
                .environment(notionViewModel)
                .environment(syncViewModel)
                .environment(xrayViewModel)
                .environment(authManager)
                .task {
                    // Connectivity check and VM bootstrapping in background asynchronously
                    logger.info("🚀 App launched — starting background bootstrapping")
                    Task {
                        _ = await apiClient.bootstrapConnection()
                        await syncViewModel.bootstrap()
                        xrayViewModel.isPipelineActive = syncViewModel.isRunning
                        await xrayViewModel.bootstrapIfNeeded()
                    }
                }
        }
    }
}
