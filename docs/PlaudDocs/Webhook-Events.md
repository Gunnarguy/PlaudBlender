DEVELOPER GUIDES
Webhook Events
Learn how to use webhooks to receive asynchronous notifications when events happen.
​
Overview

Webhooks allow you to receive real-time notifications about events happening within the Plaud ecosystem, such as when a transcription is completed or a new file is uploaded. Instead of continuously polling the API for status changes, you can subscribe to events. This is particularly useful for long-running transcription and extraction jobs.
When an event happen, Plaud will send a POST request to your specified webhook URL with the relevant results.
​
Using Webhooks

1
1. Create a webhook endpoint

First, create a webhook in the Plaud Developer Portal. Navigate to your event subscribe and click “Add Subscription”. You will need to configure the following:
Name: A descriptive name for your webhook
Callback URL: Your publicly accessible HTTPS endpoint
Events to subscribe: Select the specific events you want to receive notifications for.
2
2. Implement and Verify Your Endpoint

Here is an example of how to implement a webhook endpoint using Python and Flask. This is a complete, runnable example that includes signature verification.
Python
import os
import hmac
import hashlib
from flask import Flask, request, abort

app = Flask(__name__)

# Your webhook signing secret, stored as an environment variable
WEBHOOK_SECRET = os.environ.get('PLAUD_WEBHOOK_SECRET')

def verify_signature(payload_body, signature_header):
    """Verify that the payload was sent from Plaud."""
    if not signature_header:
        raise ValueError("Signature header is missing.")
    
    hash_object = hmac.new(WEBHOOK_SECRET.encode('utf-8'), msg=payload_body, digestmod=hashlib.sha256)
    expected_signature = hash_object.hexdigest()

    if not hmac.compare_digest(expected_signature, signature_header):
        raise ValueError("Signatures do not match.")

@app.route('/webhook', methods=['POST'])
def webhook():
    # 1. Verify the signature
    try:
        signature = request.headers.get('Plaud-Signature')
        verify_signature(request.data, signature)
    except ValueError as e:
        print(f"Error verifying signature: {e}")
        abort(400, 'Invalid signature')
    
    # 2. Handle the event
    event = request.get_json()
    event_type = event.get('event_type')
    
    if event_type == 'audio_transcribe.completed':
        data = event.get('data', {})
        print(f"Transcription completed for file: {data.get('file_id')}")
        # Add your business logic here...
    else:
        print(f"Received an unhandled event type: {event_type}")
        
    # 3. Acknowledge receipt of the event
    return {'status': 'success'}, 200

if __name__ == '__main__':
    app.run(port=3000)
​
Security Considerations

HTTPS Requirement: Your webhook endpoint URL must use HTTPS to ensure data is encrypted in transit.
Signature Verification: As detailed above, always verify the signature of incoming requests.
Idempotency: Your endpoint might receive the same event more than once. Design your event processing to be idempotent to prevent duplicate processing.