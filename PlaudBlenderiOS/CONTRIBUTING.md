# Contributing Guidelines: PlaudBlenderiOS

This document outlines development procedures, coding conventions, testing guidelines, and standards for contributing to the PlaudBlenderiOS project.

---

## 1. Project Status

PlaudBlenderiOS is an active, native Swift application. We welcome structural refinements, networking upgrades, and diagnostic tool improvements that enhance the developer dashboard experience.

---

## 2. Development Prerequisites

To set up your local development workspace, ensure you meet these targets:
- **macOS**: Version Sequoia or newer.
- **IDE**: Xcode 26.3+ installed.
- **Target SDK**: iOS 26.2+ deployment target SDK.
- **Local Server**: A running instance of the [Chronos FastAPI backend](https://github.com/Gunnarguy/PlaudBlender).

---

## 3. Git and Branch Workflow

We follow standard branch coordination policies:
1. **Branch Names**: Format branch paths with descriptive category prefixes:
   - `feature/new-view-name`
   - `bugfix/issue-identifier`
   - `docs/update-naming`
2. **Commit Style Guidelines**: Prefix commit descriptions with lowercase tags followed by clear, present-tense summaries:
   - `feat: add WKWebView static content loaders`
   - `fix: correct token validation cache logic`
   - `docs: update setup commands in README`

---

## 4. Coding Conventions

All code must conform to Swift standard guidelines and iOS structural conventions:

### SwiftUI Design Conventions
- **ViewModel Lifecycle Caching**: Never allocate ViewModels directly inside View initializers. Always route dependencies through `ViewModelCache` allocations inside ContentView or store them within the Environment context.
- **Modifiers Order**: Position layout and frames properties before background colors or shapes to ensure expected view sizing behaves predictably.
- **Previews Isolation**: Wrap SwiftUI previews in static mock bindings. Avoid executing live network queries inside preview builds.

### Swift Concurrency Standards
- **Thread Safety**: Ensure views modifications are decorated with the `@MainActor` attribute.
- **Task Management**: Use native `.task` blocks inside view declarations to tie async network tasks to the view's layout lifecycle, preventing memory leaks when views disappear.
- **Isolated State**: Limit shared singletons. Inject network engines and authentication managers using SwiftUI's `.environment()` decorators.

---

## 5. Testing Expectations

Before opening a pull request, verify that:
- [ ] The app compiles successfully inside Xcode without syntax warnings.
- [ ] All unit tests in the `PlaudBlenderiOSTests` target execute and pass.
- [ ] Network events log entries mask credentials correctly.

---

## 6. Guidelines for Autonomous AI Agents

If you are an AI coding assistant (such as Antigravity, Copilot, or Cursor) working on this codebase, you must adhere to these directives:
- **Prioritize Real Architectures**: Never invent APIs, models, or views that are not already present in the workspace. Refer to existing implementation patterns in [APIClient.swift](PlaudBlenderiOS/Services/APIClient.swift) and [SyncViewModel.swift](PlaudBlenderiOS/ViewModels/SyncViewModel.swift).
- **Maintain Diffs Observability**: Present clear differences before modifying files.
- **Document Changes Chronologically**: If refactoring views, update the respective API route maps in [README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md) to keep documentation in sync with code.
