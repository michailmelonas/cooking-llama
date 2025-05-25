import os
import zipfile

import boto3


def download_weights():
    """Download weights from S3 and save to /persistent-storage"""
    session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    s3_client = session.client("s3")
    s3_client.download_file(
        "llama3.2-1b-instruct",
        "Llama3.2-1B-Instruct.zip",
        "/persistent-storage/Llama3.2-1B-Instruct.zip"
    )
    with zipfile.ZipFile("/persistent-storage/Llama3.2-1B-Instruct.zip", "r") as f:
        f.extractall("/persistent-storage/")

    os.remove("/persistent-storage/Llama3.2-1B-Instruct.zip")
