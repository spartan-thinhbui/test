# Footprinter Service

## API Documentation

### Overview
The Footprinter API provides building footprint detection from aerial/satellite imagery. It accepts either a direct image upload (base64 encoded) or an S3 URI pointing to an image file.

### Base URL

#### Local testing
http://127.0.0.1:8000/predict

#### Deployment on Lightning.ai Studio
https://8000-01jm7zdyw3d5w0jzqpwp1bkk8j.cloudspaces.litng.ai/predict

### Authentication
The API requires Bearer token authentication.

Authorization: Bearer <API_TOKEN>

### Endpoints

#### POST /predict
Detects building footprints in the provided image.

##### Request Format
Content-Type: `application/json`

##### Request Body
The request body must contain one of the following:

| Field | Type | Description |
|-------|------|-------------|
| file_uri | string | (Optional) S3 URI pointing to an image file (e.g., "s3://bucket/path/to/image.jpg") |
| image | string | (Optional) Base64 encoded image string |

**Note:** At least one of `file_uri` or `image` must be provided and non-empty.

##### Example Requests

1. Using S3 URI:

```json
{
    "file_uri": "s3://resiquant-service-platform-private-dev/tmp/file_store_sample/1_front_st_san_francisco_ca_94111/"
}
```

2. Using Base64 encoded image:

```json
{
    "image": "base64_encoded_image_string"
}
```

##### Response Format
Content-Type: `application/json`

```python
{
    "polygons": [
        [
            [[x1, y1], [x2, y2], ..., [xn, yn]],  // First polygon
            [[x3, y3], [x4, y4], ..., [xj, yj]],  // First hole (if there are holes in the polygon)
            [[x5, y5], [x6, y6], ..., [xk, yk]],  // Second hole (if there are holes in the polygon)
        ],
        [
            [[x1, y1], [x2, y2], ..., [xn, yn]]  // Second polygon
        ]
        // ... more polygons
    ]
}
```


### Response Codes
* 200: Success
* 422: Validation Error (invalid or missing input)
* 500: Server Error

### Error Responses

```json
{
    "detail": "No valid input provided: need either a base64 encoded image or a non-empty S3 file URI"
}
```

### Example Usage (Python)

```python
import requests
import base64
from PIL import Image
import io

def image_to_base64(image_path, format='JPEG', quality=85):
    """Convert image to base64 string"""
    img = Image.open(image_path)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=format, quality=quality)
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode('utf-8')

# API configuration
url = "http://127.0.0.1:8000/predict"
# url = "https://8000-01jm7zdyw3d5w0jzqpwp1bkk8j.cloudspaces.litng.ai/predict"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Example 1: Using local image file
image_path = "path/to/image.jpg"
image_base64 = image_to_base64(image_path)
payload = {"image": image_base64}
response = requests.post(url, json=payload, headers=headers)

# Example 2: Using S3 URI
payload = {
    "file_uri": "s3://bucket/path/to/image.jpg"
}
response = requests.post(url, json=payload, headers=headers)
```

### Notes
* The API processes one image at a time
* Maximum image size and supported formats should be specified
* Processing time may vary depending on image size and complexity


### How to Test locally:
1. Run `python server.py` in one terminal
2. Run `python client.py` in another terminal

### How to Test on Lightning.ai Studio:
1. Run `python server.py` in one terminal
2. Run `python client.py` in another terminal

