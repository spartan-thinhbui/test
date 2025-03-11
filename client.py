"""
client.py for testing
Author: Lei Cao
Email: lei@resiquant.ai

Copyright: Resiquant Inc.
"""
import requests
import os 
import base64
from PIL import Image
import io

DEBUG = False


url = "http://127.0.0.1:8000/predict"      
# url = "https://8000-01jm7zdyw3d5w0jzqpwp1bkk8j.cloudspaces.litng.ai/predict"
API_TOKEN = os.environ.get('API_TOKEN')
headers = {"Authorization": f"Bearer {API_TOKEN}"}

payload_list = [
    {"file_uri": "s3://sagemaker-us-west-2-resiquant-ai-cv-dev/resiquant_200_addresses/1_front_st_san_francisco_ca_94111.jpg"},
    {"file_uri": "s3://resiquant-service-platform-private-dev/tmp/blank_white.jpg"},
    {"file_uri": ''}
]

def image_to_base64(image_path, format='JPEG', quality=85):
    if format is None:
        raise ValueError("Image format cannot be None. Please specify a valid format (e.g. 'JPEG', 'WEBP')")
        
    """Convert image to base64 string"""
    img = Image.open(image_path)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=format, quality=quality)  # format="WEBP" or "JPEG" for our use case
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode('utf-8')

def test_request(payload_list=None, image_path=None):
    if image_path:
        try:
            image_base64 = image_to_base64(image_path)
            payload = {"image": image_base64}
            if DEBUG:
                print(f"Payload length: {len(image_base64)}")
                print(f"First 100 chars of payload: {image_base64[:100]}")  # Debug print
                print(f"Full payload keys: {payload.keys()}")  # Debug print
                # print("Full payload:", payload)  # Add this line
            response = requests.post(
                url,
                json=payload,
                headers=headers
            )
            if response.status_code == 200:
                print(f"Testing image file: {image_path}")
                print("Response headers:", response.headers)
                print("Response text:", response.text)
            else:
                print(f"Request failed with status code {response.status_code}.")
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            
    if payload_list:
        # Test file URI payloads
        for payload in payload_list:
            print(f"Testing payload: {payload}")
            response = requests.post(
                url, 
                json=payload,
                headers=headers
            )
            if response.status_code == 200:
                print("Response:", response.text)
            else:
                print(f"Request failed with status code {response.status_code}.")


# Example usage:
if __name__ == "__main__":
    # Test with S3 URIs
    # test_request(payload_list=payload_list)
    
    # Test with local image file
    image_path="test_data/resiquant_200_addresses/10_almaden_blvd_san_jose_ca_95113.jpg"
    test_request(image_path=image_path)