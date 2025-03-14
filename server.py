import os
import logging

import litserve as ls
from modules import ImageEvaluationPipeline 

import config
from utils import parse_uri, get_np_image_from_s3

from schemas import ImageEvaluateRequest, ImageEvaluateResponse
import numpy as np

import time
import io
from PIL import Image
import base64

IS_LOCAL_ENV = os.environ.get('ENVIRONMENT', '').lower() == 'local'
if IS_LOCAL_ENV:
    num_devices = config.local_num_devices
    workers_per_device = config.local_workers_per_device
else:
    num_devices = config.server_num_devices
    workers_per_device = config.server_workers_per_device

# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s UTC - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('server.log')
    ]
)
logger = logging.getLogger(__name__)


class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        logging.info(f"Initializing ImageEvaluationPipeline on device: {device}")
        try:
            # Initialize pipeline
            self.model = ImageEvaluationPipeline(
                groundingdino_config_path=config.groundingdino_config_path,
                groundingdino_model_path=config.groundingdino_model_path,
                sam_model_path=config.sam_model_path,
                depth_model_path=config.depth_model_path,
                depth_model_type=config.depth_model_type,
                brisque_model_path=config.brisque_model_path,
                brisque_range_path=config.brisque_range_path,
                bbox_threshold=config.bbox_threshold,
                text_threshold=config.text_threshold,
                building_top_max_gap=config.building_top_max_gap,
                device=device
            )
            logging.info("ImageEvaluationPipeline initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize ImageEvaluationPipeline: {str(e)}")
            raise

    def decode_request(self, request: ImageEvaluateRequest) -> np.array:
        try:
            # The logic is if image is provided, use it regardless of file_uri presence. 
            # Otherwise, use file_uri.
            if request.image is not None:
                logger.info("Processing direct image input")
                try:
                    # Decode base64 string to bytes
                    image_bytes = base64.b64decode(request.image)
                    image = Image.open(io.BytesIO(image_bytes))
                    return np.array(image), request.focal_length
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
                    return image_np, request.focal_length
                except Exception as e:
                    logger.error(f"Error processing S3 file: {str(e)}")
                    raise
            else:
                logger.error("No valid input provided: need either a base64 encoded image or a non-empty S3 file URI")
                raise ValueError("No valid input provided: need either a base64 encoded image or a non-empty S3 file URI")
        except Exception as e:
            logger.error(f"Error decoding request: {str(e)}")
            raise

    def predict(self, input: any) -> dict:
        img, focal_length = input

        logging.info("Starting prediction")
        start_time = time.time()
        try:
            result = self.model.predict(img, focal_length)
            processing_time = time.time() - start_time
            logger.info(f"Prediction completed in {processing_time:.2f} seconds")
            return result
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise

    def encode_response(self, output: dict) -> ImageEvaluateResponse:
        try:
            response = ImageEvaluateResponse(**output)
            logging.debug("Response encoded successfully")
            return response
        except Exception as e:
            logging.error(f"Failed to encode response: {str(e)}")
            raise


if __name__ == "__main__": 
    logging.info("Starting SimpleLitAPI server")
    api = SimpleLitAPI()
    server = ls.LitServer(
        api, 
        accelerator='gpu', 
        devices=num_devices, 
        workers_per_device=workers_per_device, 
        timeout=180, 
        track_requests=True)

    logging.info(f"Server configured with GPU acceleration, {num_devices} devices, {workers_per_device} workers per device")
    server.run(port=8000)