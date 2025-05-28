from fastapi import Header, HTTPException, status
from config import Config

def api_key_auth(x_api_key: str = Header(None)):
    if Config.API_KEY and x_api_key != Config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return x_api_key