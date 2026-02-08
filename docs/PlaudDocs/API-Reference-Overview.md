API REFERENCE
Overview
Welcome to the Plaud API Reference
​
Introduction

This API reference describes the RESTful, streaming, and realtime APIs you can use to interact with Plaud Platform. REST APIs are usable via HTTP in any environment that supports HTTP requests. Language-specific SDKs are listed on the library page.
​
Authentication

All requests to the Plaud API must include an Authorization header with an API Token. For detailed instructions on how to generate an API token, please refer to Authorization.
API Token should be provided via HTTP Bearer authentication.
We will release the Client SDKs in coming releases. The SDK will send the header on your behalf with every request.
​
Content types

The Plaud API accepts JSON in request bodies and returns JSON in all response bodies. You must include the Content-Type: application/json header in all requests that send a JSON payload (e.g., POST, PUT, PATCH).