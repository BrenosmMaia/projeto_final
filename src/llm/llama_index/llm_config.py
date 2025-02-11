import os

from dotenv import find_dotenv, load_dotenv

from llama_index.llms.bedrock import Bedrock

load_dotenv(find_dotenv())

llama = Bedrock(
    model="meta.llama3-3-70b-instruct-v1:0",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    region_name="us-east-2",
    context_size=128_000,
    temperature=0.0,
)
