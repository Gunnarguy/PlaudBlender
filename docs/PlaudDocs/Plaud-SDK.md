Welcome to PLAUD SDK

https://github.com/Plaud-AI/plaud-sdk

License Platform API iOS

This is the official hub for the PLAUD SDK.

Table of Contents

Introduction
System Architecture
Supported Platforms
Core Features
Integration
Examples
FAQ
Support
License
Introduction

You can use this SDK to integrate PLAUD's AI-powered hardware and software services into your products, including but not limited to:

🔌 Connecting to PLAUD devices
📊 Monitoring device settings and status
🎮 Controlling device operations such as starting, stopping, and more
⚙️ Modifying device configurations to meet your requirements
📥 Retrieving recordings produced by devices after obtaining authorization
🔄 Integrating with your workflow via PLAUD Cloud or self-hosted services
System Architecture

The PLAUD system architecture is designed as a modular, AI-powered platform that seamlessly integrates hardware devices with intelligent software services to capture, process, and extract insights from audio recordings. The architecture consists of three interconnected layers that work together to deliver a comprehensive audio intelligence solution. Several key concepts are explained below:

PLAUD SDK: A software development kit enabling third-party applications to integrate with PLAUD's AI-powered hardware and software services. The SDK acts as the integration layer.
Client Host App: The client's app that integrates the PLAUD SDK to access PLAUD's hardware and software services. This enables organizations to incorporate PLAUD's audio intelligence capabilities into their existing applications across various domains.
PLAUD Template App: A pre-built template app provided by PLAUD, enabling enterprises to rapidly customize and deploy private-branded solutions via APIs and Apps. This out-of-the-box solution accelerates time-to-market for custom audio intelligence applications.
PLAUD system architecture diagram

Supported Platforms

PLAUD SDK is designed to work across all platforms. ​Support for Android and iOS is now available, and additional platforms are currently under development.

Platform	Support Status	Min OS Version
iOS	✅ Available	iOS 13.0+
Android	✅ Available	API 21+
ReactNative	✅ Available	API 21+、iOS 13.0+
Web	🚧 Under Development	-
macOS	🚧 Under Development	-
Windows	🚧 Under Development	-
Core Features

Device Management

📡 Scan and discover nearby devices
🔗 Bind and connect to target devices
📊 Retrieve real-time device status
💾 Check device storage capacity
🔋 Monitor battery level and charging status
Recording Operations

⏺️ Start/pause/resume/stop recordings
📋 Get recording item lists with metadata
⬇️ Download recordings with progress tracking
🗑️ Delete specific recordings
🧹 Clear all recording items
Network Configuration

➕ Add Wi-Fi networks
➖ Remove Wi-Fi networks
🔧 Update Wi-Fi configurations
📝 Retrieve saved Wi-Fi list
☁️ Configure upload destinations
Cloud Services

🤖 AI-powered processing:
🎯 Automatic transcription
📝 Smart summarization
⚙️ Customizable templates
🌐 Multi-language support
Integration

See the SDK Integration Guide

Examples

iOS Demo App

 PLAUD_SDK_Demo_iOS.mp4 

Android Demo App

 PLAUD_SDK_Demo_Android.mp4 

Check out the example applications:

iOS Demo App
Android Demo App
Install app

Android App
FAQ

What audio formats are supported?

The SDK supports MP3, WAV formats currently.

Is offline mode supported?

Yes, basic device operations work offline. Cloud features require an internet connection.

How to get a PLAUD device?

PLAUD devices are available for purchase through our official website or other authorized online retailers.

How can I test PLAUD devices with my system?

An appKey is required for secure device connection and access. If you don't have one, contact support@plaud.ai to request a test key, including your company name and intended use case. Then begin testing by following steps:

Build and run the Demo app in the examples folder.
Enter your appKey (generated above).
Scan for available devices and connect to yours.
Can I test without a physical device?

No, ​a physical device is currently a necessity.

Try Playground App

Coming soon

Support

📧 Email: support@plaud.ai
📖 Documentation: plaud.mintlify.app
🐛 Issues: GitHub Issues
License

This project is licensed under the MIT License

