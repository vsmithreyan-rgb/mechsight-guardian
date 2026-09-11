import os
from pathlib import Path
from datetime import datetime

import boto3
from dotenv import load_dotenv


load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")

s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
)


def upload_file(local_path, s3_key):
    local_path = Path(local_path)

    if not local_path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")

    s3_client.upload_file(
        str(local_path),
        AWS_S3_BUCKET,
        s3_key,
    )

    return {
        "bucket": AWS_S3_BUCKET,
        "key": s3_key,
        "s3_uri": f"s3://{AWS_S3_BUCKET}/{s3_key}",
    }


def upload_inspection_video(local_path):
    local_path = Path(local_path)

    if not local_path.exists():
        raise FileNotFoundError(f"Inspection video not found: {local_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    s3_key = (
        f"inspection-videos/"
        f"{timestamp}_{local_path.name}"
    )

    return upload_file(
        local_path,
        s3_key,
    )