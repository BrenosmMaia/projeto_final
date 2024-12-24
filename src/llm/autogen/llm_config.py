import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

llama_bedrock = [
    {
        "api_type": "bedrock",
        "model": "meta.llama3-3-70b-instruct-v1:0",
        "aws_region": "us-east-2",
        "aws_access_key": os.getenv("AWS_ACCESS_KEY"),
        "aws_secret_key": os.getenv("AWS_SECRET_KEY"),
        "price": [0.0024, 0.0024],
        "temperature": 0.3,
        "cache_seed": None,
    }
]

claude_haiku = [
    {
        "api_type": "bedrock",
        "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "aws_region": "us-east-2",
        "aws_access_key": os.getenv("AWS_ACCESS_KEY"),
        "aws_secret_key": os.getenv("AWS_SECRET_KEY"),
        "price": [0.001, 0.005],
        "temperature": 0.3,
        "cache_seed": None,
    }
]
