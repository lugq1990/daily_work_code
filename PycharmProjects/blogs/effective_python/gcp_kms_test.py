import requests
import os

url = "https://us-central1-buoyant-sum-302208.cloudfunctions.net/function-1"

key_file_name = "buoyant-sum-302208-d86625c05132.json"
key_file_path = r"C:\Users\guangqiiang.lu\Downloads"

key_file_path = os.path.join(key_file_path, key_file_name)

# Authenticate with service account 
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession
from google.cloud import storage

creds = service_account.IDTokenCredentials.from_service_account_file(key_file_path, target_audience=url)

auth_session = AuthorizedSession(creds)

# use credential to call url
import json

# response = auth_session.post(url, data=json.dumps({"message": "hi world"}))
response =auth_session.request("POST", url, json={"message":"hi, world!"})

print(response.status_code)
print(response.text)


class A:

    def __init__(self, a) -> None:

        self.a = a