import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("KEY")
secret = os.getenv("SECRET")

symbol = "BTCUSDT"
