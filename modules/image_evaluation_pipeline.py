# %% [markdown]
# ## Install dependencies

# %%
# !pip install segment-anything groundingdino-py opencv-python-headless

# %% [markdown]
# # Prerequisites for Apple Silicon
# 
# Apply this fix to segment-anything package that you just installed in your environment:
# https://github.com/facebookresearch/segment-anything/pull/122/commits/cd507390ca9591951d0bfff2723d1f6be8792bb8

# %%
from enum import Enum
import torch
import numpy as np
import cv2
import os
import time

if __name__ == '__main__':
    from models import GroundingDINOModel, SAM2Model, DepthProModel
    from utils import *
else:
    from .models import GroundingDINOModel, SAM2Model, DepthProModel
    from .utils import *

class DetectionObjects(str, Enum):
    BUILDING = "building"
    SKY = "sky"
    TREE = "tree"
    VEHICLE = "vehicle"
    POLE = "pole"

class ImageEvaluationPipeline:
    def __init__(
        self,
        groundingdino_model_path: str,
        groundingdino_config_path: str,
        sam_model_path,
        depth_model_type,
        depth_model_path,
        brisque_model_path: str,
        brisque_range_path: str,
        bbox_threshold: float = 0.3,
        text_threshold: float = 0.25,
        building_top_max_gap: int = 10,
        device: str = "cpu",
    ):
        self.device = device

        self.object_detect_model = GroundingDINOModel(
            model_path=groundingdino_model_path,
            model_config_path=groundingdino_config_path,
            device=device
        )
        
        self.segmentation_model = SAM2Model(
            model_path=sam_model_path,
            device=device
        )

        self.depth_estimation_model = DepthProModel(
            model_type=depth_model_type,
            model_path=depth_model_path,
            device=device
        )

        self.bbox_threshold = bbox_threshold
        self.text_threshold = text_threshold

        self.building_top_max_gap = building_top_max_gap

        self.brisque_model_path=brisque_model_path
        self.brisque_range_path=brisque_range_path

        self.target_objects = [
            DetectionObjects.BUILDING,
            DetectionObjects.SKY,
            DetectionObjects.TREE,
            DetectionObjects.VEHICLE, 
            DetectionObjects.POLE
        ]


    @staticmethod
    def _compute_building_coverage(building_masks: np.array) -> float:
        """
            Compute building coverage
            Args:
                building_masks: a list of building masks with shape (1xHxW)
            Returns:
                float: building coverage score (0 to 1)
        """
        if len(building_masks) > 0:
            building_mask = np.sum(building_masks, axis=0)[0]
            building_coverage = compute_mask_area_ratio(building_mask)
            return building_coverage
        else:
            return 0.0
    
    
    @staticmethod
    def _compute_sky_coverage(sky_masks: np.array) -> float:
        """
            Compute sky coverage
            Args:
                sky_masks: a list of sky masks with shape (1xHxW)
            Returns:
                float: sky coverage score (0 to 1)
        """
        if len(sky_masks) > 0:
            sky_mask = np.sum(sky_masks, axis=0)[0]
            sky_coverage = compute_mask_area_ratio(sky_mask)
            return sky_coverage
        else:
            return 0.0
    

    @staticmethod
    def _compute_building_visibility(building_bboxes: list[np.array], obstruction_masks: np.array) -> float:
        """
            Compute building visiblity
            Args:
                building_bboxes: a list of building bounding boxes
                obstruction_masks: a list of obstruction (sky, ground, sidewalk) masks with shape (1xHxW)
            Returns:
                float: sky coverage score (0 to 1)
        """
        if len(building_bboxes) == 0:
            return 0.0
        
        if len(obstruction_masks) == 0:
            return 1.0

        obstruction_mask = np.sum(obstruction_masks, axis=0)[0]
        obstruction_mask = obstruction_mask > 0
        building_bbox_mask = create_bbox_mask(obstruction_mask.shape, building_bboxes)
        building_visibility = 1 - compute_mask_intersect_ratio(building_bbox_mask, obstruction_mask)

        return building_visibility


    def _compute_image_quality(self, image: np.array) -> float:
        """
            Compute image quality score by using BRISQUE
            Args:
                image: np.array with shape (H, W, 3)
            Returns:
                float: image quality score (0 to 1)
        """
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
        
        # Compute BRISQUE score
        brisque_score = cv2.quality.QualityBRISQUE_compute(gray_image, self.brisque_model_path, self.brisque_range_path)[0]

        # invert BRISQUE score and scale to [0, 1]
        image_quality = max(0, min(1, 1 - (brisque_score / 100)))
        return image_quality
        

    @staticmethod
    def compute_building_completeness(building_bboxes: list[np.array], image_shape: np.array) -> tuple[bool, bool]:
        """
            Compute building completeness
            Args:
                building_bboxes: building bounding boxes
                image_shape: image shape
            Returns:
                bool: Building completeness vertically (True or False)
                bool: Building completeness horizontally (True or False)
        """
        if len(building_bboxes) > 0:
            # only select building with largest area
            areas = [compute_bbox_area(bbox) for bbox in building_bboxes]
            target_building_bbox = building_bboxes[np.argmax(areas)]

            vertical_complete = is_bbox_inside_image_vertical(target_building_bbox, image_shape, 5) # offset 5 pixels
            horizontal_complete = is_bbox_inside_image_horizontal(target_building_bbox, image_shape, 5) # offset 5 pixels
            return vertical_complete, horizontal_complete
        else:
            return False, False
    

    def _get_building_top(
        self, 
        image_shape: tuple,
        building_masks: np.ndarray,
        sky_masks: np.ndarray,
        building_top_max_gap: int = 10,
        debug_image: np.ndarray = None,
        output_dir: str = None,
        image_filename: str = None
    ) -> tuple[int, np.ndarray]:
        """
        Find the highest point of the building at the center of the image where it meets the sky.
        Only counts if sky is visible from the very top of the image.
        
        Args:
            image_shape: tuple of (height, width, channels)
            building_masks: numpy array of building segmentation masks
            sky_masks: numpy array of sky segmentation masks
            building_top_max_gap: maximum allowed gap between sky and building
            debug_image: original image for visualization if debug is True
            output_dir: directory to save debug visualization
            image_filename: original image filename to use for debug image naming
        
        Returns:
            int: y-coordinate of highest building point at center (-1 if no sky found)
            np.ndarray: debug visualization if debug_image is provided, None otherwise
        """
        building_top = -1
        
        # Get image dimensions
        height, width = image_shape[:2]
        center_x = width // 2
        
        if len(building_masks) > 0 and len(sky_masks) > 0:
            # Combine all building masks and sky masks
            building_mask = np.any(building_masks, axis=0)[0]
            sky_mask = np.any(sky_masks, axis=0)[0]
            
            # Get vertical profile at center of image
            building_profile = building_mask[:, center_x]
            sky_profile = sky_mask[:, center_x]
            
            # Check if sky is present at the top of the image
            if sky_profile[0]:
                # Find the first non-sky pixel from top, which should be building
                building_top_point = None
                last_sky_point = None

                for y in range(height):
                    # If we hit a non-sky pixel and we're still in continuous sky region
                    if sky_profile[y]:
                        last_sky_point = y
                    elif building_profile[y]:
                        building_top_point = y
                        break

                if building_top_point and last_sky_point and \
                    (building_top_point - last_sky_point <= building_top_max_gap + 1):
                    building_top = (building_top_point + last_sky_point) // 2

        # Create debug visualization if debug_image is provided
        if debug_image is not None and output_dir is not None:
            try:
                # Create a copy of the image for visualization
                debug_vis = debug_image.copy()
                
                # Create visualization masks
                sky_overlay = np.zeros_like(debug_image)
                sky_overlay[sky_mask] = [135, 206, 235]  # Light blue for sky
                building_overlay = np.zeros_like(debug_image)
                building_overlay[building_mask] = [139, 69, 19]  # Brown for building
                
                # Blend masks with original image
                alpha = 0.5
                debug_vis = cv2.addWeighted(debug_vis, 1, sky_overlay, alpha, 0)
                debug_vis = cv2.addWeighted(debug_vis, 1, building_overlay, alpha, 0)
                
                # Draw center line
                cv2.line(debug_vis, (center_x, 0), (center_x, height), (0, 255, 0), 1)
                
                # Draw building top point if found
                if building_top != -1:
                    cv2.circle(debug_vis, (center_x, building_top), 10, (0, 0, 255), -1)
                
                # Add text to show detection status
                status_text = "Detection Failed: "
                if len(building_masks) == 0:
                    status_text += "No building detected"
                elif len(sky_masks) == 0:
                    status_text += "No sky detected"
                elif not np.any(sky_mask[0, :]):
                    status_text += "No sky at top of image"
                elif building_top == -1:
                    status_text += "Gap too large between sky and building"
                else:
                    status_text = f"Building top detected at y={building_top}"
                
                # Put status text on image
                cv2.putText(debug_vis, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          1, (0, 0, 0), 4, cv2.LINE_AA)  # Thicker black outline
                cv2.putText(debug_vis, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          1, (255, 255, 255), 1, cv2.LINE_AA)  # White inner text
                
                # Save debug visualization
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                # Create unique filename for debug image
                if image_filename:
                    base_name = os.path.splitext(os.path.basename(image_filename))[0]
                    debug_filename = f'{base_name}_building_top_debug.jpg'
                else:
                    # Fallback to timestamp if no filename provided
                    debug_filename = f'building_top_debug_{int(time.time())}.jpg'
                    
                output_path = os.path.join(output_dir, debug_filename)
                cv2.imwrite(output_path, cv2.cvtColor(debug_vis, cv2.COLOR_RGB2BGR))
                print(f"Debug image for {os.path.basename(image_filename)} saved to {output_path}")
            except Exception as e:
                print(f"Error saving debug image for {image_filename}: {str(e)}")

        return building_top

    def get_building_pounding_distance(self, image: np.array, focal_length: float, building_masks: np.array, building_bboxes: np.array):
        """
        Get nearest distance of the center building in the image with its neighbor
        Args:
            image: numpy array with shape HxWxC represents the image
            focal_length: focal length of the image
            building_masks: detected buildings segmentation masks
            building_bboxes: detected building bounding boxes
        Returns:
            float: minimum distance between the target building and its nearest neighbor,
            if there's no target building nor neighbor building, -1 will be returned
        """

        h, w = image.shape[:2]
        cx, cy = int(w/2), int(h/2)

        target_buildings = []
        neighbor_buildings = []

        # get target building and its nearest neighbor
        for i, building in enumerate(building_bboxes):
            xmin, ymin, xmax, ymax = building
            if xmin < cx < xmax and ymin < cy < ymax:
                target_buildings.append(i)
            else:
                neighbor_buildings.append(i)
        
        # return -1 if there's no target building nor neighbor building
        if len(target_buildings) == 0 or len(neighbor_buildings) == 0:
            print("There's no target building and neighbor, returning -1")
            return -1

        # estimate depth map
        print("Calculating depth map...")
        depth_map = self.depth_estimation_model.predict(image, focal_length=focal_length)


        print("Calculate distance between target building and its nearest neighbor...")
        
        # select the smallest target building to avoid overlapping
        min_building = np.argmin([compute_bbox_area(building_bboxes[i]) for i in target_buildings])
        target_bbox = building_bboxes[target_buildings[min_building]]
        target_mask = building_masks[target_buildings[min_building]]

        # find the closest neighbor
        neighbor_buildings_bboxes = [building_bboxes[i] for i in neighbor_buildings]
        _, nearest_idx = find_nearest_bbox(neighbor_buildings_bboxes, target_bbox)
        neighbor_mask = building_masks[nearest_idx]

        # calculate 3d distance between 2 buildings
        fx = fy = focal_length
        min_distance = compute_3d_distance(depth_map, target_mask[0], neighbor_mask[0], fx, fy, cx, cy)

        return min_distance


    def predict(self, image: np.ndarray, focal_length: float = None, output_dir: str = None, image_filename: str = None) -> dict:
        """
        Evaluate image by the Building Coverage, Sky Coverage, Building Completeness, and Image Quality
        
        Args:
            image: numpy array of shape (height, width, channels) in RGB format.
            focal_length: float image focal length
            output_dir: directory to save debug visualizations
            image_filename: original image filename to use for debug image naming
        Returns:
            dict: evaluated scores as key-value pair with key is the metric name and value is the score
        """
        target_bounding_boxes = self.object_detect_model.predict(image, self.target_objects, self.bbox_threshold, self.text_threshold)

        building_bboxes = target_bounding_boxes[DetectionObjects.BUILDING]
        sky_bboxes = target_bounding_boxes[DetectionObjects.SKY]

        nbuilding = len(building_bboxes)
        nsky = len(sky_bboxes)

        all_bboxes = [target_bounding_boxes[object] for object in self.target_objects]
        all_bboxes = sum(all_bboxes, []) # flatten array
        
        scores = {
            "building_coverage": 0,
            "sky_coverage": 0,
            "building_visibility": 0,
            "building_vertical_completeness": 0,
            "building_horizontal_completeness": 0,
            "image_quality": 0,
            "building_top": -1,  # Initialize to -1 instead of 0
            "pounding_distance": -1
        }

        # Initialize empty masks
        building_masks = np.zeros((0, image.shape[0], image.shape[1]), dtype=bool)
        sky_masks = np.zeros((0, image.shape[0], image.shape[1]), dtype=bool)

        if len(all_bboxes) > 0:
            segmented_masks = self.segmentation_model.predict(image, bboxes=all_bboxes, erode=True)

            # Building Coverage
            building_masks = segmented_masks[:nbuilding]
            scores["building_coverage"] = self._compute_building_coverage(building_masks)

            # Sky coverage
            sky_masks = segmented_masks[nbuilding:nbuilding+nsky]
            scores["sky_coverage"] = self._compute_sky_coverage(sky_masks)

            # Building Visiblity
            scores["building_visibility"] = self._compute_building_visibility(
                building_bboxes, 
                segmented_masks[nbuilding+nsky:]
            )
            
            # building completeness
            vertical_complete, horizontal_complete = self.compute_building_completeness(
                building_bboxes, 
                image.shape
            )
            scores["building_vertical_completeness"] = float(vertical_complete)
            scores["building_horizontal_completeness"] = float(horizontal_complete)

            # Image quality
            scores["image_quality"] = self._compute_image_quality(image)

            # Always call building top detection, even with empty masks
            building_top = self._get_building_top(
                image_shape=image.shape,
                building_masks=building_masks,
                sky_masks=sky_masks,
                building_top_max_gap=self.building_top_max_gap,
                debug_image=image,
                output_dir=output_dir,
                image_filename=image_filename
            )
            scores["building_top"] = building_top
        
            if focal_length and nbuilding > 0:
                scores["pounding_distance"] = self.get_building_pounding_distance(image, focal_length, building_masks, building_bboxes)

        return scores


# %%
if __name__ == "__main__":
    import time
    from pathlib import Path

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    pipeline = ImageEvaluationPipeline(
        groundingdino_config_path="../configs/GroundingDINO_SwinB.cfg.py",
        groundingdino_model_path="../models/groundingdino_swinb_cogcoor.pth",
        depth_model_path="../models/depth_pro.pt",
        depth_model_type="dinov2l16_384",
        sam_model_path="../models/sam2.1_hiera_large.pt",
        brisque_model_path="../configs/brisque_model_live.yml",
        brisque_range_path="../configs/brisque_range_live.yml",
        bbox_threshold=0.3,
        text_threshold=0.25,
        building_top_max_gap=10,
        device=device
    )

    path = "../test_data/file_store_2024_02_10/"
    dataset_dir = Path("../my_test_data/file_store/")
    output_dir = "../analysis/debug_output"
    # dataset_dir = Path("../file_store/")
    output_json = {}

    for filename in dataset_dir.glob("**/street_level/*.jpg"):
        image = cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)

        tick = time.time()
        # Get predictions from the pipeline
        scores = pipeline.predict(
            image, 
            output_dir=output_dir,
            image_filename=str(filename)
        )
        tock = time.time()
        print(f"Time taken: {tock - tick} seconds")
        print(filename, "/n", scores)
        output_json[str(filename)] = scores 
    import json
    with open('image_scores.json', 'w') as f:
        json.dump(output_json, f, indent=2)