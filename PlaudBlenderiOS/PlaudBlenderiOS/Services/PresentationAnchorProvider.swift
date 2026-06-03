import AuthenticationServices

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

@MainActor
func currentPresentationAnchor() -> ASPresentationAnchor? {
    #if canImport(UIKit)
    let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
    for scene in scenes {
        if let visibleWindow = scene.windows.first(where: { !$0.isHidden }) {
            return visibleWindow
        }
    }
    return scenes.first?.windows.first
    #elseif canImport(AppKit)
    return NSApp.keyWindow ?? NSApp.mainWindow ?? NSApp.windows.first
    #else
    return nil
    #endif
}
