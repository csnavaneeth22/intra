import os
import webbrowser
from fyers_apiv3 import fyersModel

# Replace these with your actual Fyers App details
CLIENT_ID = "4L5T10AYM2-100"
SECRET_KEY = "OL6KNXRQOV"
REDIRECT_URI = "https://127.0.0.1"

def get_auth_url():
    """Generates the Fyers Auth URL where you need to login."""
    session = fyersModel.SessionModel(
        client_id=CLIENT_ID,
        secret_key=SECRET_KEY,
        redirect_uri=REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code"
    )
    return session.generate_authcode()

def generate_access_token(auth_code):
    """
    Exchanges the auth code (from the URL after login) for an access_token.
    """
    session = fyersModel.SessionModel(
        client_id=CLIENT_ID,
        secret_key=SECRET_KEY,
        redirect_uri=REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code"
    )
    session.set_token(auth_code)
    response = session.generate_token()
    
    if "access_token" in response:
        with open("access_token.txt", "w") as f:
            f.write(response["access_token"])
        print("Generated and saved access_token.txt successfully.")
        return response["access_token"]
    else:
        print("Failed to generate token:", response)
        return None

def get_fyers_client():
    if not os.path.exists("access_token.txt"):
        raise Exception("Access token missing. Please run auth process.")
    
    with open("access_token.txt", "r") as f:
        token = f.read().strip()
        
    fyers = fyersModel.FyersModel(client_id=CLIENT_ID, is_async=False, token=token, log_path="")
    return fyers

if __name__ == "__main__":
    auth_url = get_auth_url()
    print("1. Automatically opening the browser to login...")
    try:
        webbrowser.open(auth_url)
    except:
        print("Could not open browser. Go to this URL to Login:\n", auth_url)
        
    code = input("2. Upon login, the browser will redirect to a broken page. Look at the URL bar, copy the 'auth_code=' parameter, and paste it here: ")
    generate_access_token(code)
