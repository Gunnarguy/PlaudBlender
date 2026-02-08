CAPABILITIES
File Management
Learn how to manage files and recordings under your account
​
Overview

Plaud File Whole Lifecycle
Plaud’s file API handles the entire lifecycle of your recording, from raw audio to AI-generated assets, like summaries and structured extractions. They can be used to:
Generate Actionable Insights: Transform recordings into clear action items, decisions, and follow-ups.
Automate Information Discovery: Intelligently search and pinpoint key information across all your conversations.
Power Custom Workflows: Feed structured, AI-enriched data directly into your business applications and workflows.
​
File Types

Raw Recordings: Original recording files in opus and mp3 formats from Plaud devices, direct API or SDK uploads
Audio Transcripts: Speech-to-text conversion results generated from audio files
Summary Files: Intelligent summaries created from transcripts or text content
ETL Outputs: Structured data extraction and transformation results for different industries and customized configuration
​
Ownership

In the Plaud ecosystem, each file is assigned an owner. The ownership is inherited from the device that created the recording; the owner can be an individual user, a team, or an organization. Therefore, a device must first be bound to an owner account before any file operations can be performed.
Device Bind Quickstart
Bind your device and start a developing journey in minutes.
​
Recording Upload

After binding your device successfully, you can start yo upload recordings.
File Upload Quickstart
Learn upload process for recording files.
​
AI Capabilities

Enhance your recordings with intelligent processing capabilities including automated summaries and structured data extraction workflows.
AI Summary
Generate intelligent summaries from transcripts.
​
FAQ

What file formats are supported for upload?
The platform supports opus and mp3 audio formats. Opus is the preferred format for original recordings from Plaud devices, while mp3 is supported for broader compatibility.
What's the maximum file size for uploads?
The platform uses multipart upload supporting files up to several GB in size. Files are uploaded in 5MB chunks for reliability and can be resumed if interrupted.
How secure is my file data?
All files are stored in encrypted cloud storage with secure access controls. Download links are pre-signed with automatic expiration, and all API access requires authentication tokens.
How long does transcription take?
Transcription processing time depends on audio length and quality. Typical processing is 1:4 ratio (1 minute audio = ~15 seconds processing). You’ll receive webhook notifications when complete.
Can I process multiple files simultaneously?
Yes, the platform supports batch operations and concurrent processing. You can submit multiple transcription tasks and monitor their progress through the task status API.
What happens to processed data like transcripts?
Processed data (transcripts, summaries, ETL results) are stored as structured JSON data linked to the original audio file. You can retrieve, update, or delete this data independently.
Do you support webhook notifications?
How do I organize files by project or category?
Files support metadata fields including custom names, device associations, and time-based organization. You can also use group_id for logical file grouping and parent-child relationships for file splitting/merging.