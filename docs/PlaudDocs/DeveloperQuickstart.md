GET STARTED
Developer quickstart
Learn how to make your first Plaud API request
The Plaud API provides a simple interface to state-of-the-art recording devices and features. Follow this guide to learn how to bind your Plaud device. See the developer guides for more examples with our other products.
​
Prerequisites

You need to have an iOS/Android app that your users can download, OR
Your users need to have bluetooth-enabled computers/laptops that they can use to transfer recordings over to your web app
​
Using the API

1
Create an APP

Create an APP in the dashboard here, which you’ll use to securely access the API.
Don’t have access yet? Book a call with our Submit Request Form here.
Store Client ID and Secret Key as managed secrets as a environment variable via an .env file, or directly in your app’s configuration depending on your preference.
.env
PLAUD_CLIENT_ID=<PLAUD_CLIENT_ID>
PLAUD_CLIENT_SECRET_KEY=<PLAUD_CLIENT_SECRET_KEY>
2
Install Dependencies

We’ll also use the dotenv library to load our Client ID and Secret Key from environment variables.

Python
pip install python-dotenv
3
Make your first request

Create a new file named example.py, depending on your language of choice and add the following code:

Python
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

# Step 1: Generate API token
client_id = os.getenv("PLAUD_CLIENT_ID")
secret_key = os.getenv("PLAUD_CLIENT_SECRET_KEY")

# Create base64 encoded credentials
credentials = base64.b64encode(f"{client_id}:{secret_key}".encode()).decode()

# Request API token
token_response = requests.post(
    'https://api.plaud.ai/apis/oauth/api-token',
    headers={
        'Authorization': f'Bearer {credentials}',
        'Content-Type': 'application/json'
    }
)

token_data = token_response.json()
api_token = token_data['api_token']

# Step 2: Use token to get device list
devices_response = requests.get(
    'https://api.plaud.ai/devices/',
    headers={
        'Authorization': f'Bearer {api_token}',
        'Content-Type': 'application/json'
    }   
)

devices = devices_response.json()
print(devices)
4
Run the code


Python
python example.py
You can get a device list managed under your app.