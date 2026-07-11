import SwiftUI

struct ModelSelectorView: View {
    let title: String
    @Binding var selection: String
    let options: [String]

    @State private var showTextField = false
    @State private var customValue = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    if !options.contains(selection) {
                        Text("Current: \(selection)")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                    }
                }
                
                Spacer()
                
                if options.contains(selection) && !showTextField {
                    Picker("", selection: $selection) {
                        ForEach(options, id: \.self) { option in
                            Text(option).tag(option)
                        }
                        Text("Custom...").tag("CUSTOM")
                    }
                    .pickerStyle(.menu)
                    .onChange(of: selection) { _, newValue in
                        if newValue == "CUSTOM" {
                            customValue = ""
                            showTextField = true
                        }
                    }
                } else {
                    Button(action: {
                        selection = options.first ?? ""
                        customValue = ""
                        showTextField = false
                    }) {
                        Text("Reset to standard")
                            .font(.caption)
                    }
                }
            }

            if showTextField || !options.contains(selection) {
                TextField("Custom Model Identifier", text: $customValue)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
                    .onChange(of: customValue) { _, newValue in
                        if !newValue.isEmpty {
                            selection = newValue
                        }
                    }
            }
        }
        .padding(.vertical, 4)
        .onAppear {
            if !options.contains(selection) {
                customValue = selection
                showTextField = true
            }
        }
    }
}
