SDK
iOS SDK
Start building customized, interactive iOS apps with Plaud devices.
​
Installation

Add the Plaud SDK to your iOS project using the Swift Package Manager.
1
Add Package Dependency in Xcode

With your project open in Xcode, navigate to File > Add Package Dependencies…
In the search bar that appears, enter the repository URL: https://github.com/Plaud-AI/plaud-sdk.git
Click Add Package and wait for Xcode to resolve the package.
Add Swift Package Dependency in Xcode
2
Import the SDK

Now, in any Swift file where you need to use the SDK, import the necessary modules.
import PenBleSDK
import PlaudDeviceBasicSDK
import PenWiFiSDK
​
Requirements

Supported Platforms: iOS 13.0+
Development Environment: Xcode 15.0+ and Swift 5.7+
Permissions: Your app’s Info.plist file must include keys for Bluetooth usage, such as NSBluetoothAlwaysUsageDescription.
​
Start Development

​
Initialization

The best place to initialize the SDK is within your AppDelegate or in a dedicated singleton manager class. This ensures it’s configured once when your app starts.
import UIKit

// In your AppDelegate or a central manager class
class YourAppManager {
    // Hold a strong reference to the device agent
    var deviceAgent: PlaudDeviceAgent?

    func initializePlaudSDK() {
        // Get the shared instance of the agent
        let agent = PlaudDeviceAgent.shared()
        
        // Assign the delegate to receive callbacks
        agent.delegate = self 
        
        // Initialize the SDK with your app credentials
        agent.initSDK(
            hostName: "YourAppName",
            appKey: "YOUR_APP_KEY",
            bindToken: "YOUR_BIND_TOKEN", // A unique identifier for the user or session
            extra: [:]
        )
        
        self.deviceAgent = agent
    }
}
​
Delegate Callbacks Overview

To receive events from the SDK, you must implement the PlaudDeviceAgentDelegate protocol. This protocol provides feedback on everything from connection status to recording events.
It’s best practice to make your central manager class or a relevant view controller conform to this delegate protocol.
import Foundation
import PlaudDeviceBasicSDK // For BleDevice, BleFile types
import PenBleSDK

@objc public protocol PlaudDeviceAgentDelegate: AnyObject {
    
    // MARK: - Initialization & Connection
    
    /// Called with the result of the AppKey verification.
    @objc optional func plaud(didVerifyAppKeyWith result: Int)
    
    /// Reports the Bluetooth connection state.
    @objc optional func plaud(didUpdateConnectionState state: Int)
    
    /// Callback for the device binding (pairing) process.
    @objc optional func plaud(didBindDevice sn: String?, status: Int, protVersion: Int, timezone: Int)
    
    /// Unbind result.
    @objc optional func plaud(didUnbind status: Int)
    
    
    // MARK: - Device Discovery
    
    /// Called when Bluetooth devices are discovered during a scan.
    @objc optional func plaud(didDiscover devices: [BleDevice])
    
    /// Called when the device scan times out.
    @objc optional func plaudDidTimeoutScan()
    

    // MARK: - Device Status
    
    /// Provides the current state of the PLAUD device.
    @objc optional func plaud(deviceDidUpdateState state: Int, privacy: Int, keyState: Int, uDisk: Int, findMyToken: Int, hasSndpKey: Int, deviceAccessToken: Int)
    
    /// Provides the device's name.
    @objc optional func plaud(didReceiveDeviceName name: String?)
    
    /// Reports the device's storage capacity.
    @objc optional func plaud(didReceiveStorageInfo total: Int, free: Int, duration: Int)
    
    /// Reports a change in the device's battery level.
    @objc optional func plaud(didUpdateBatteryLevel power: Int, oldPower: Int)
    
    /// Reports the device's charging status.
    @objc optional func plaud(didUpdateChargingState isCharging: Bool, level: Int)
    
    /// Provides the microphone's current gain value.
    @objc optional func plaud(didReceiveMicGain value: Int)
    
    
    // MARK: - Recording Control
    
    /// Confirms that recording has started.
    @objc optional func plaud(didStartRecordingWithSessionId sessionId: Int, start: Int, status: Int, scene: Int, startTime: Int, reason: Int)
    
    /// Confirms that recording has stopped.
    @objc optional func plaud(didStopRecordingWithSessionId sessionId: Int, reason: Int, fileExist: Bool, fileSize: Int)
    
    /// Confirms that recording has been paused.
    @objc optional func plaud(didPauseRecordingWithSessionId sessionId: Int, reason: Int, fileExist: Bool, fileSize: Int)
    
    /// Confirms that a paused recording has resumed.
    @objc optional func plaud(didResumeRecordingWithSessionId sessionId: Int, start: Int, status: Int, scene: Int, startTime: Int)
    
    
    // MARK: - File Management
    
    /// Provides the list of files stored on the device.
    @objc optional func plaud(didReceiveFileList files: [BleFile])
    
    /// Reports the result of a file deletion attempt.
    @objc optional func plaud(didDeleteFileWithSessionId sessionId: Int, status: Int)
    
    
    // MARK: - Data Transfer (Download)
    
    /// Indicates the start of a file download.
    @objc optional func plaud(didStartSyncingFileHead sessionId: Int, status: Int)
    
    /// Indicates the end of a file download.
    @objc optional func plaud(didFinishSyncingFileTail sessionId: Int, crc: Int)
    
    /// Delivers chunks of file data during a download.
    @objc optional func plaud(didReceiveDataChunk sessionId: Int, start: Int, data: Data)

    /// Reports download progress for a composite file.
    @objc optional func plaud(didDownloadFile sessionId: Int, desiredOutputPath: String, status: Int, progress: Int, tips: String)
    
    /// Called when a file download is manually stopped.
    @objc optional func plaudDidStopDownloadingFile()
    
    
    // MARK: - Wi-Fi Configuration
    
    /// Reports the result of setting a Wi-Fi configuration.
    @objc optional func plaud(didSetConfigForWifiSync result: Int)

    /// Provides the list of saved Wi-Fi networks.
    @objc optional func plaud(didReceiveWifiSyncList list: [UInt32])
    
    /// Reports the result of a Wi-Fi connection test.
    @objc optional func plaud(didReceiveWifiTestResult index: UInt32, result: Int, rawCode: Int)
    
    
    // MARK: - Firmware Updates (OTA)
    
    /// Reports the result of a firmware update (OTA).
    @objc optional func plaud(didFinishOtaWithResult uid: Int, status: Int, errmsg: String?)
    
    /// The device is requesting a specific chunk of the firmware file.
    @objc optional func plaud(deviceDidRequestOtaPacket uid: Int, start: Int, end: Int)
}
See all 113 lines
​
Methods

Here are some of the most common methods you’ll use to interact with the device.
​
Device Scanning

To discover nearby PLAUD devices, start a scan. The results will be delivered to the plaud(didDiscover:) delegate method.
// Get the shared agent instance
guard let deviceAgent = self.deviceAgent else { return }

// Start scanning for devices
deviceAgent.startScan()

// Implement the delegate method in your class to receive the results
// extension YourViewController: PlaudDeviceAgentDelegate {
//     func plaud(didDiscover devices: [BleDevice]) {
//         // Sort devices by signal strength (strongest first)
//         let sortedDevices = devices.sorted { $0.rssi > $1.rssi }
        
//         // Update your UI with the sorted list of devices
//         self.discoveredDevices = sortedDevices
//         self.tableView.reloadData()
//         print("Discovered devices: \(sortedDevices.map { $0.name ?? "Unknown" })")
//     }
// }
​
Connect to a Device

Once a device has been discovered, you can connect to it using its BleDevice object.
// Assume `selectedDevice` is a `BleDevice` object from the scan results
func connect(to device: BleDevice) {
    guard let deviceAgent = self.deviceAgent else { return }
    deviceAgent.connect(device)
}

// Implement the delegate method to handle connection state changes
// func plaud(didUpdateConnectionState state: Int) {
//     let message: String
//     switch state {
//     case 0:
//         message = "Device disconnected or failed to connect."
//     case 1:
//         message = "Device connected successfully!"
//     case 2:
//         message = "Device connection failed."
//     default:
//         message = "Unknown connection state."
//     }
//     print(message)
//     // Update your UI to reflect the new connection status
// }
​
Start Recording

Send a command to the connected device to begin recording.
guard let deviceAgent = self.deviceAgent else { return }
deviceAgent.startRecord()

// Implement the delegate method to confirm recording has started
// func plaud(didStartRecordingWithSessionId sessionId: Int, status: Int, ...) {
//     if status == 0 {
//         print("Recording started successfully with session ID: \(sessionId)")
//     } else {
//         print("Failed to start recording. Status code: \(status)")
//     }
//     // Update your recording UI
// }
​
Stop Recording

Send a command to the connected device to stop the current recording.
guard let deviceAgent = self.deviceAgent else { return }
deviceAgent.stopRecord()

// Implement the delegate method to confirm recording has stopped
// func plaud(didStopRecordingWithSessionId sessionId: Int, reason: Int, ...) {
//     // Reason codes indicate why recording stopped:
//     // 1: Stopped by device button
//     // 2: Stopped by app command
//     // 3: Stopped due to auto-split feature
//     // 4: Stopped by physical mode switch
//     print("Recording stopped for session \(sessionId). Reason code: \(reason)")
//     // Update your UI and potentially start file download
// }
​
Example Implementation

A complete example iOS app using this SDK can be found in the official repository.
For a full, working example, please refer to the demo app in the Plaud iOS SDK repository. The app demonstrates:
Device scan, connect, and disconnect flows
Recording control (start, stop, pause, resume)
Recording file management (list, download, delete)
Device Wi-Fi configuration and testing
Triggering transcription and summary services