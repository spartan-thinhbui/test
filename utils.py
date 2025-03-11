"""
Author: Lei Cao
Email: lei@resiquant.ai

Copyright: Resiquant Inc.
"""

import boto3
import io
import numpy as np
from PIL import Image

s3_client = boto3.client('s3')

def get_s3_object(bucket_name, file_relative_path):
    """Gets object as data bytes from S3 URI
    """
    response = s3_client.get_object(Bucket=bucket_name, Key=file_relative_path)
    data_bytes = response['Body'].read()
    return data_bytes


def convert_bytes_to_np_image(data_bytes):
    data_stream = io.BytesIO(data_bytes)
    # Open the image using PIL
    img = Image.open(data_stream)
    # Optionally, convert the image to RGB (if you want to remove any alpha channel)
    img = img.convert("RGB")
    # Convert the PIL image to a NumPy array
    image_np = np.array(img)
    return image_np


def get_np_image_from_s3(s3_bucket_name: str, file_path: str) -> np.ndarray:
    data_bytes = get_s3_object(s3_bucket_name, file_path)
    image_np = convert_bytes_to_np_image(data_bytes)
    return image_np


def parse_uri(file_uri):
    split = file_uri.split('//')
    names = split[1].split('/')
    bucket_name = names[0]
    file_path = '/'.join(names[1:])
    return bucket_name, file_path


if __name__ == '__main__':
    bucket_name = 'sagemaker-us-west-2-resiquant-ai-cv-dev' # Replace with your bucket name
    object_name = 'resiquant_200_addresses/1_front_st_san_francisco_ca_94111.jpg' # Replace with your object key
    
    data_bytes = get_s3_object(bucket_name, object_name)
    image_np = convert_bytes_to_np_image(data_bytes)

    print(image_np)

