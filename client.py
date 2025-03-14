"""
client.py for testing
Author: Lei Cao
Email: lei@resiquant.ai

Copyright: Resiquant Inc.
"""
import requests
import os 

if __name__ == "__main__":


    url = "http://127.0.0.1:8000/predict"      
    # url = "https://8000-01jmaf81kef4wvpry4q70jhz72.cloudspaces.litng.ai/predict"
    API_TOKEN = os.environ.get('API_TOKEN')
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    s3_path_prefix = "s3://resiquant-service-platform-private-dev/tmp/file_store_sample"
    file_path = "1_front_st_san_francisco_ca_94111/imagery/street_level/google_xefC4NNFlXbZq-OEDLb_aQ_0.jpg"

    payload = {"file_uri": f"{s3_path_prefix}/{file_path}", "focal_length": 400.0}
    response = requests.post(
        url, 
        json=payload,
        headers=headers
    )


    if response.status_code == 200:
        print("Response:", response.text)
    else:
        print(f"Request failed with status code {response.status_code}.")