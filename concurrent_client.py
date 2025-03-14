import asyncio
import aiohttp
import os
from typing import List, Dict
import json
from pathlib import Path
import time
import base64
from tqdm import tqdm


url = "http://127.0.0.1:8000/predict"      
# url = "https://8000-01jmaf81kef4wvpry4q70jhz72.cloudspaces.litng.ai/predict"
API_TOKEN = os.environ.get('API_TOKEN')
headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}
the_path = "my_test_data/file_store_sample1"
# the_path = "test_data/file_store_2024_02_10"
# the_path = "my_test_data/file_store"
s3_path_prefix = "s3://resiquant-service-platform-private-dev/tmp/file_store_sample"

def get_address_images() -> Dict[str, List[str]]:
    """
    Scan through all address folders and collect street-level images.
    Returns a dictionary with address folder names as keys and lists of image paths as values.
    """
    address_images = {}
    
    # Get all immediate subdirectories (address folders)
    base_dir = Path(the_path)
    address_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    print(f"\nFound {len(address_dirs)} address directories")
    
    for address_dir in address_dirs:
        print(f"\nProcessing address: {address_dir.name}")
        
        # Construct path to street_level directory
        street_level_dir = address_dir / "imagery" / "street_level"
        if not street_level_dir.exists():
            print(f"No street_level directory found in {address_dir}")
            continue
        
        # Get all jpg files
        all_images = list(street_level_dir.glob("*.jpg"))
        print(f"Total JPG files found: {len(all_images)}")
        # # print("All images:", [f.name for f in all_images])
        
        # Filter images
        images = [
            str(f.relative_to(base_dir))
            for f in all_images
            # if not (f.name.endswith("wide.jpg") or f.name.endswith("zoom.jpg"))
        ]
        
        # print(f"After filtering: {len(images)} images")
        # print("Filtered images:", [Path(img).name for img in images])
        
        if images:
            address_images[str(address_dir.name)] = images
        else:
            print(f"No valid images found for {address_dir.name}")
            
    return address_images

def encode_image_to_base64(image_path: str) -> str:
    """Convert an image file to base64 string"""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_focal_length_from_json(image_path: str) -> float:
    """Read focal length from corresponding JSON file"""
    # Convert image path to JSON path (replace .jpg with .json)
    json_path = Path(the_path) / image_path.replace('.jpg', '.json')
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            return float(data.get('pproj_focal_length', 400.0))  # Default to 400.0 if not found
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not read focal length from {json_path}: {str(e)}")
        return 400.0  # Default focal length

async def process_single_image(
    session: aiohttp.ClientSession,
    image_path: str,
    semaphore: asyncio.Semaphore,
    use_base64: bool = False
) -> tuple[str, dict]:
    """Process a single image through the API"""
    async with semaphore:
        # Get focal length from corresponding JSON file
        focal_length = get_focal_length_from_json(image_path)
        
        if use_base64:
            # Read and encode the image file
            full_path = Path(the_path) / image_path
            image_base64 = encode_image_to_base64(str(full_path))
            payload = {
                "image": image_base64,
                "focal_length": focal_length
            }
        else:
            payload = {
                "file_uri": f"{s3_path_prefix}/{image_path}",
                "focal_length": focal_length
            }

        try:
            async with session.post(url, json=payload, headers=headers) as response:
                response_text = await response.text()
                
                if response.status == 200:
                    result = await response.json()
                    return image_path, {"status": "success", "response": result}
                else:
                    return image_path, {
                        "status": "error",
                        "error": f"Status code: {response.status}",
                        "details": response_text
                    }
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            return image_path, {
                "status": "error",
                "error": str(e)
            }


async def process_all_images(
    all_images: List[tuple[str, str]],
    max_concurrent: int = 4,
    use_base64: bool = False
) -> Dict[str, dict]:
    """Process all images concurrently with a global concurrency limit"""
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(max_concurrent)
        
        total_images = len(all_images)
        print(f"\nStarting to process {total_images} images across all addresses")
        
        # Initialize counters for real-time stats
        success_count = 0
        error_count = 0
        pbar = tqdm(total=total_images, desc="Processing images", unit="img")
        
        # Create tasks list
        tasks = [
            process_single_image(session, image_path, semaphore, use_base64)
            for address, image_path in all_images
        ]
        
        # Process results as they complete
        organized_results = {}
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            img_path, result = await task
            
            # Update progress and stats
            if result["status"] == "success":
                success_count += 1
            else:
                error_count += 1
                print(f"\nError processing {img_path}: {result.get('error', 'Unknown error')}")
            
            # Find corresponding address for this image
            for address, image_path in all_images:
                if image_path == img_path:
                    if address not in organized_results:
                        organized_results[address] = {}
                    organized_results[address][image_path] = result
                    break
            
            # Update progress bar description with current stats
            pbar.set_description(f"Processing images (Success: {success_count}, Errors: {error_count})")
            pbar.update(1)
        
        pbar.close()
        print(f"\nFinal Results - Processed {total_images} images: {success_count} successful, {error_count} failed")
            
        return organized_results

async def process_all_addresses(
    address_images: Dict[str, List[str]],
    max_concurrent: int = 4,
    use_base64: bool = False
) -> Dict[str, Dict[str, dict]]:
    """Process all images from all addresses concurrently"""
    start_time = time.time()
    
    # Flatten all images into a single list with their address
    all_images = [
        (address, image_path)
        for address, images in address_images.items()
        for image_path in images
    ]
    
    total_images = len(all_images)
    print(f"Total images to process across all addresses: {total_images}")
    
    results = await process_all_images(all_images, max_concurrent, use_base64)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    if total_images > 0:
        print(f"\nOverall Statistics:")
        print(f"Total images processed: {total_images}")
        print(f"Total time taken: {total_time:.2f} seconds")
        print(f"Average time per image: {total_time/total_images:.2f} seconds")
    
    return results

async def main(max_concurrent: int = 4, use_base64: bool = False, output_path: str = './'):
    address_images = get_address_images()
    print(f"Found {len(address_images)} addresses to process")
    print(f"Using {'base64 encoding' if use_base64 else 'S3 URIs'} for image transfer")
    
    results = await process_all_addresses(address_images, max_concurrent, use_base64)
    
    # Save final results

    output_path = Path(output_path)
    with open(output_path / 'final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    # Set use_base64=True to send images as base64 instead of S3 URIs
    print('Starting tests for image bytes inputs')
    results = asyncio.run(main(max_concurrent=320, output_path=the_path, use_base64=True))

    # print('Starting tests for file_uri inputs')
    # results = asyncio.run(main(max_concurrent=8, use_base64=False))