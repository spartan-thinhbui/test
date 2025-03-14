<h1 align="center">StreetView Image Evaluator</h1>

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed.
- `pip`, the Python package installer.

### Setup

Follow these steps to get started:

```bash
# Clone the repository
git clone https://github.com/ResiQuant/image-filtering.git
cd image-filtering

# Install required dependencies
pip install -r requirements.txt

# Start the server
python server.py

```

> - **Server URL:** `http://localhost:8000`
> - **Server API Docs:** `http://localhost:8000/docs`

---

### Usage

#### Making API Requests

**Via cURL:**

```sh
curl http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "file_uri": "s3://sagemaker-us-west-2-resiquant-ai-cv-dev/resiquant_200_addresses/1_front_st_san_francisco_ca_94111.jpg",
    "focal_length": 400.0
  }'
```

or

```sh
curl http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image": "/9j/4AAQSkZJRg...base64_encoded_image_data...",
    "focal_length": 400.0
  }'
```

---

### Sample Response

```json
{
  "building_coverage": 0.652788,
  "sky_coverage": 0.21675466666666668,
  "building_visibility": 0.9984217277531684,
  "building_vertical_completeness": 1,
  "building_horizontal_completeness": 0,
  "image_quality": 0.8240751266479492,
  "pounding_distance": 13.523075943681578
}
```
