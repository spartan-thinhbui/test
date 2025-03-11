"""
server.py

Author: Lei Cao
Email: lei@resiquant.ai

Copyright: Resiquant Inc.
"""

import litserve as ls
from modules.efficientnet_sam_pipeline import SegmentationPipeline
import config
from utils import get_np_image_from_s3, parse_uri
from schemas import FootprinterRequest, FootprinterResponse
import numpy as np
import logging
import time
import io
from PIL import Image
import base64


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s UTC - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('server.log')
    ]
)
logger = logging.getLogger(__name__)

# STEP 1: DEFINE YOUR MODEL API
class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        logger.info(f"Initializing model pipeline on device: {device}")
        try:
            pipeline = SegmentationPipeline(
                unet_model_path=config.unet_model_path,
                sam_model_path=config.sam_model_path,
                sam_model_type=config.sam_model_type,
                device=device,
                unet_tta=1
            )
            self.model = pipeline
            logger.info("Model pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model pipeline: {str(e)}")
            raise

    def decode_request(self, request: FootprinterRequest) -> np.array:
        try:
            # The logic is if image is provided, use it regardless of file_uri presence. 
            # Otherwise, use file_uri.
            if request.image is not None:
                logger.info("Processing direct image input")
                try:
                    # Decode base64 string to bytes
                    image_bytes = base64.b64decode(request.image)
                    image = Image.open(io.BytesIO(image_bytes))
                    return np.array(image)
                except base64.binascii.Error:
                    logger.error("Invalid base64 string provided")
                    raise ValueError("Invalid base64 string provided")
                except Exception as e:
                    logger.error(f"Error processing image data: {str(e)}")
                    raise
            
            # Handle file URI input
            elif request.file_uri and len(request.file_uri) > 0:
                try:
                    bucket_name, file_path = parse_uri(request.file_uri)
                    logger.info(f"Processing request for file: {bucket_name}/{file_path}")
                    image_np = get_np_image_from_s3(bucket_name, file_path)
                    return image_np
                except Exception as e:
                    logger.error(f"Error processing S3 file {request.file_uri}: {str(e)}")
                    raise
            else:
                logger.error("No valid input provided: need either a base64 encoded image or a non-empty S3 file URI")
                raise ValueError("No valid input provided: need either a base64 encoded image or a non-empty S3 file URI")
        except Exception as e:
            logger.error(f"Error decoding request: {str(e)}")
            raise

    def predict(self, x: np.array) -> list:
        start_time = time.time()
        try:
            result = self.model.predict(x)[1]
            processing_time = time.time() - start_time
            logger.info(f"Prediction completed in {processing_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def encode_response(self, output) -> FootprinterResponse:
        try:
            return FootprinterResponse(polygons=output)
        except Exception as e:
            logger.error(f"Error encoding response: {str(e)}")
            raise


# STEP 2: START THE SERVER
if __name__ == "__main__":
    api = SimpleLitAPI()
    # server = ls.LitServer(api, accelerator="gpu")
    server = ls.LitServer(api, accelerator='gpu', devices=1, workers_per_device=1, timeout=180)
    server.run(port=8000)