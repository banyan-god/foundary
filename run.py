import logging
import uvicorn
from config import Config
from api.main import app

if __name__ == "__main__":
    # configure root logger level from settings
    level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=level)
    # start server with matching log level
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level=Config.LOG_LEVEL.lower())