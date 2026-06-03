import SwiftUI

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

let platformTrailingToolbarPlacement: ToolbarItemPlacement = {
    #if os(macOS)
    .automatic
    #else
    .topBarTrailing
    #endif
}()

let platformLeadingToolbarPlacement: ToolbarItemPlacement = {
    #if os(macOS)
    .automatic
    #else
    .topBarLeading
    #endif
}()

extension View {
    @ViewBuilder
    func platformNavigationBarTitleDisplayModeInline() -> some View {
        #if os(macOS)
        self
        #else
        self.navigationBarTitleDisplayMode(.inline)
        #endif
    }

    @ViewBuilder
    func platformTextEntryBehavior() -> some View {
        #if canImport(UIKit)
        self
            .autocorrectionDisabled()
            .textInputAutocapitalization(.never)
        #else
        self
        #endif
    }
}

func copyToClipboard(_ value: String) {
    #if canImport(UIKit)
    UIPasteboard.general.string = value
    #elseif canImport(AppKit)
    let pasteboard = NSPasteboard.general
    pasteboard.clearContents()
    pasteboard.setString(value, forType: .string)
    #endif
}
