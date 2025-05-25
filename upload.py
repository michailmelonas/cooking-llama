import os
import boto3


session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)
s3_client = session.client("s3")


s3_client.upload_file(
    "/Users/michailmelonas/.llama/checkpoints/Llama3.2-1B-Instruct.zip",
    "llama3.2-1b-instruct",
    "Llama3.2-1B-Instruct.zip"
)