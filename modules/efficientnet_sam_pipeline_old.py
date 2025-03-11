# %% [markdown]
# ## Install dependencies

#!pip install rasterio shapely scikit-learn scikit-image einops timm geopandas segment-anything matplotlib opencv-python-headless albumentations ipywidgets

# %% [markdown]
# # Prerequisites for Apple Silicon
# 
# Apply this fix to segment-anything package that you just installed in your environment:
# https://github.com/facebookresearch/segment-anything/pull/122/commits/cd507390ca9591951d0bfff2723d1f6be8792bb8



"""
This module is the original pipeline where EfficientNet and SAM provides their own predicted masks, which are further combined into on mask as output
"""


import json
import os
from abc import ABC, abstractmethod

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import rasterio
from rasterio import features
from matplotlib import pyplot as plt
import geopandas as gpd
import shapely
import numpy as np
import cv2


# Utils functions

# %%
def find_files(directory, file_name):
    """
    Search for list of files by name in a directory and its subdirectories
    """
    for root, _, files in os.walk(directory):
        if file_name in files:

            file_path = os.path.join(root, file_name)
            yield file_path

def find_key_in_nested_dict(d, target_key):
    """
    Recursively search for a key in a nested dictionary or list and return the key-value pair.
    """
    if isinstance(d, dict):
        # If the current level is a dictionary, search through its key-value pairs
        for key, value in d.items():
            if key == target_key:
                yield key, value
            elif isinstance(value, (dict, list)):  # If the value is a dict or list, recurse into it
                yield from find_key_in_nested_dict(value, target_key)
    elif isinstance(d, list):
        # If the current level is a list, iterate through its elements
        for item in d:
            yield from find_key_in_nested_dict(item, target_key)

def show_anns(anns, borders=True):
    """
    Plot predicted masks
    """

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)


def export_vector_data_from_mask(mask, output):
    """
    Plot predicted masks
    """

    if mask.max() < 1:
        mask = mask * 255

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    vector_data = {
      "type": "FeatureCollection",
      "features": [{
        "type": "Feature",
        "geometry": {
          "type": "Polygon",
          "coordinates": [[(int(point[0][0]), int(point[0][1])) for point in contour] for contour in contours]
        },
        "properties": {
          "name": "Shape"
        }
      }]
    }

    with open(output, "w") as json_file:
        json.dump(vector_data, json_file)

    return vector_data

# %%
def check_mask_intersection(mask1, mask2, ratio=0.0):
    """
    Check if two segmentation masks intersect over a area ratio

    Args:
        mask1 (numpy.ndarray): First binary mask (same shape as mask2).
        mask2 (numpy.ndarray): Second binary mask.
        ratio (float): Area percentage that aleast 2 masks must intersect. Default to 0%.

    Returns:
        bool: True if there is an intersection, False otherwise.
    """
    intersection = np.logical_and(mask1, mask2)  # Find overlapping pixels
    union = np.logical_or(mask1, mask2)  # Find all pixels

    union_area = np.sum(union)
    intersect_area = np.sum(intersection)

    if union_area > 0:
        return intersect_area / union_area > ratio

    return False


def combine_masks(anns, output_mask, bounding_mask=None, resize_mask=None):
    """
    Combine predicted masks into one
    """

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    masks = []

    for ann in sorted_anns:
        if bounding_mask is None or check_mask_intersection(ann['segmentation'], bounding_mask, 0.01):
            masks.append(ann['segmentation'])

    if len(masks) > 0:
        mask = np.sum(masks, axis=0)
        mask[mask > 0] = 1
        mask = mask.astype(np.uint8)
    else:
        mask = np.zeros_like(sorted_anns[0]['segmentation'], dtype=np.uint8)

    if resize_mask:
        mask = cv2.resize(mask, resize_mask, interpolation=cv2.INTER_LINEAR)

    if output_mask:
        cv2.imwrite(output_mask, (mask * 255))

    return mask



def raster_to_vector(source, output, simplify_tolerance=None, dst_crs=None, **kwargs):
    """Vectorize a raster dataset.

    Args:
        source (str): The path to the tiff file.
        output (str): The path to the vector file.
        simplify_tolerance (float, optional): The maximum allowed geometry displacement.
            The higher this value, the smaller the number of vertices in the resulting geometry.
    """
    

    with rasterio.open(source) as src:
        band = src.read()

        mask = band != 0
        shapes = features.shapes(band, mask=mask, transform=src.transform)

    fc = [
        {"geometry": shapely.geometry.shape(shape), "properties": {"value": value}}
        for shape, value in shapes
    ]
    if simplify_tolerance is not None:
        for i in fc:
            i["geometry"] = i["geometry"].simplify(tolerance=simplify_tolerance)

    gdf = gpd.GeoDataFrame.from_features(fc)
    if src.crs is not None:
        gdf.set_crs(crs=src.crs, inplace=True)

    if dst_crs is not None:
        gdf = gdf.to_crs(dst_crs)

    gdf.to_file(output, **kwargs)


albu_dev = A.Compose([
    A.Normalize(),
    ToTensorV2(),
])

def dev_transform(img):
    mask = np.zeros_like(img)[..., [0]]

    data = albu_dev(image=img, mask=mask)
    img = data['image']

    return img

def preprocess_image(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Preprocess single image to target size
    Args:
        image: numpy array of shape (height, width, channels) in RGB format
        target_size: tuple of (height, width)
    Returns:
        resized_image: numpy array of shape (target_height, target_width, channels)
    """
    return cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

def preprocess_mask(mask: np.ndarray, target_size: tuple = None) -> np.ndarray:
    """
    Preprocess mask to ensure binary values and optionally resize
    Args:
        mask: numpy array of any shape with boolean or numeric values
        target_size: optional tuple of (height, width) for resizing
    Returns:
        processed_mask: numpy array with binary values (0 and 1)
    """
    # Convert to binary values first
    processed_mask = (mask > 0.5).astype(np.uint8)
    
    # Resize if target size is provided
    if target_size is not None:
        processed_mask = cv2.resize(
            processed_mask,
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_LINEAR
        )
        processed_mask = (processed_mask > 0.5).astype(np.uint8)
    
    return processed_mask


# %%
def find_center_contour(contours, image_shape):
    """
    Find the contour closest to the image center
    Args:
        contours: list of contours from cv2.findContours
        image_shape: tuple of (height, width)
    Returns:
        center_contour: the contour closest to center
    """
    if not contours:
        return None
        
    # Image center point
    image_center = np.array([image_shape[1] / 2, image_shape[0] / 2])
    
    min_distance = float('inf')
    center_contour = None
    
    for contour in contours:
        # Calculate contour center (centroid)
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
            
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        contour_center = np.array([center_x, center_y])
        
        # Calculate distance to image center
        distance = np.linalg.norm(contour_center - image_center)
        
        if distance < min_distance:
            min_distance = distance
            center_contour = contour
    
    return center_contour


def annotate_image(original_image, mask, output_image):
    """
    Create an annotated image with an overlay applied only to the part of the mask 
    inside the central contour, along with a bold pink contour.
    
    Args:
        original_image (np.ndarray): Image array of shape (height, width, channels) in RGB format.
        mask (np.ndarray): Binary mask of shape (height, width) with values 0 or 1 (or booleans).
        output_image (str): File path to save the annotated image.
        
    Returns:
        np.ndarray: Annotated image with the overlay and drawn bold pink contour.
    """
    # Ensure mask is binary (0 and 1)
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Prepare mask for contour detection by converting to 0-255 uint8
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the central contour within the overall mask
    central_contour = find_center_contour(contours, mask.shape)
    
    # Create an empty mask for the central contour overlay region
    central_overlay_mask = np.zeros_like(binary_mask)
    if central_contour is not None:
        # Fill in the central contour area with 1's
        cv2.drawContours(central_overlay_mask, [central_contour], contourIdx=-1, color=1, thickness=-1)
    
    # Determine the overlay region: apply overlay only where both the mask and central contour are true
    overlay_region = cv2.bitwise_and(binary_mask, central_overlay_mask)
    
    # Create an overlay image with a specific color (green in this case)
    overlay_color = np.array([0, 255, 0], dtype=np.uint8)
    overlay = np.zeros_like(original_image, dtype=np.uint8)
    overlay[overlay_region == 1] = overlay_color
    
    # Blend the overlay onto the original image with a transparency factor
    annotated = cv2.addWeighted(original_image, 1.0, overlay, 0.5, 0)
    
    # Draw the central contour in bold pink if available
    if central_contour is not None:
        pink_color = (255, 105, 180)  # Pink color in RGB
        red_color = (255, 0, 0)
        cv2.drawContours(annotated, [central_contour], contourIdx=-1, color=pink_color, thickness=5)
        M = cv2.moments(central_contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            # Draw the centroid as a small yellow circle
            cv2.circle(annotated, (center_x, center_y), 5, (255, 255, 0), -1)
    
    # Save the annotated image if an output path is provided (convert from RGB to BGR for saving)
    if output_image:
        cv2.imwrite(output_image, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    
    return annotated

def annotate_and_save_image(original_image, mask, contours, output_image):
    """
    Create an annotated image with overlays for all polygons and their holes.
    
    Args:
        original_image (np.ndarray): Image array of shape (height, width, channels) in RGB format.
        mask (np.ndarray): Binary mask of shape (height, width) with values 0 or 1.
        contours: List of lists where each sublist contains polygon and its holes as point lists.
        output_image (str): File path to save the annotated image.
        
    Returns:
        np.ndarray: Annotated image with overlays and drawn contours.
    """
    # Ensure mask is binary (0 and 1)
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Create an empty mask for the overlay regions
    overlay_mask = np.zeros_like(binary_mask)
    
    # Create annotated image starting with the original
    annotated = original_image.copy()
    
    for polygon_with_holes in contours:
        if not polygon_with_holes:  # Skip empty contours
            continue
            
        # Convert exterior polygon points to numpy array format for OpenCV
        exterior = np.array(polygon_with_holes[0], dtype=np.int32)
        exterior = exterior.reshape((-1, 1, 2))
        
        # Draw filled exterior polygon
        cv2.drawContours(overlay_mask, [exterior], -1, 1, -1)
        
        # Draw exterior contour in pink
        pink_color = (255, 105, 180)  # Pink color in RGB
        cv2.drawContours(annotated, [exterior], -1, pink_color, 2)
        
        # Draw holes if any exist
        for hole in polygon_with_holes[1:]:
            hole_contour = np.array(hole, dtype=np.int32).reshape((-1, 1, 2))
            # Remove hole area from overlay
            cv2.drawContours(overlay_mask, [hole_contour], -1, 0, -1)
            # Draw hole contour in different color
            cv2.drawContours(annotated, [hole_contour], -1, (0, 255, 255), 2)
        
        # Calculate and draw centroid for the exterior polygon
        M = cv2.moments(exterior)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            # Draw the centroid as a small yellow circle
            cv2.circle(annotated, (center_x, center_y), 5, (255, 255, 0), -1)
    
    # Create green overlay for all valid regions
    overlay = np.zeros_like(original_image, dtype=np.uint8)
    overlay[overlay_mask == 1] = [0, 255, 0]  # Green color
    
    # Blend the overlay with the annotated image
    annotated = cv2.addWeighted(annotated, 1.0, overlay, 0.3, 0)
    
    # Save the annotated image
    if output_image:
        cv2.imwrite(output_image, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    
    return annotated

def save_mask(mask, output_path):
    cv2.imwrite(
        output_path,
        mask * 255
    )





class Model(ABC):
    def __init__(self, model_path, device):
        self.device = device
        self.model_path = model_path
        self.model = self._load_model()
    
    @abstractmethod
    def _load_model(self):
        """Load the model from the specified path"""
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray, output_dir=None):
        """
        Run inference on single image
        Args:
            image: numpy array of shape (height, width, channels)
            output_dir: optional directory to save outputs
        Returns:
            mask: numpy array of shape (height, width)
        """
        pass


class UNetModel(Model):
    def __init__(self, model_path, device, tta=1):
        self.tta = tta
        super().__init__(model_path, device)
    
    def _load_model(self):
        model = torch.jit.load(self.model_path, map_location=self.device)
        return model.to(self.device).eval()
    
    def predict(self, image: np.ndarray, output_dir=None):
        """
        Args:
            image: numpy array of shape (1024, 1024, channels) in RGB format
            output_dir: optional directory to save visualization outputs
        Returns:
            mask: numpy array of shape (1024, 1024) with values 0 or 1
        """
        # Initialize prediction tensor
        pred = torch.zeros((1, 2, image.shape[0], image.shape[1]), 
                          dtype=torch.float32, device=self.device)
        
        # Preprocess image
        processed_image = dev_transform(image)
        img_tensor = torch.unsqueeze(processed_image, 0).to(self.device)
        
        with torch.no_grad():
            # Normal prediction
            mask = self.model(img_tensor)
            pred += torch.softmax(mask, dim=1)
            
            # TTA if enabled
            if self.tta > 1:
                # Horizontal flip
                mask = self.model(torch.flip(img_tensor, dims=[-1]))
                pred += torch.flip(torch.softmax(mask, dim=1), dims=[-1])
            
            if self.tta > 2:
                # Vertical flip
                mask = self.model(torch.flip(img_tensor, dims=[-2]))
                pred += torch.flip(torch.softmax(mask, dim=1), dims=[-2])
            
            if self.tta > 3:
                # Both flips
                mask = self.model(torch.flip(img_tensor, dims=[-1, -2]))
                pred += torch.flip(torch.softmax(mask, dim=1), dims=[-1, -2])
            
            pred /= self.tta
            
            # Convert to binary mask
            mask = (pred.argmax(1) > 0.5).cpu().numpy()[0]
            
            # Save visualization if output_dir provided
            if output_dir:
                os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
                
                cv2.imwrite(
                    os.path.join(output_dir, "masks", "mask.jpg"),
                    mask * 255
                )
                annotate_image(
                    image, 
                    mask,
                    os.path.join(output_dir, "annotations", "annotated.jpg")
                )
            
            return mask


from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


class SAMModel(Model):
    def __init__(self, model_path, device, model_type="vit_h"):
        self.model_type = model_type
        super().__init__(model_path, device)
        
    def _load_model(self):
        sam = sam_model_registry[self.model_type](checkpoint=self.model_path)
        sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        return sam
    
    def predict(self, image: np.ndarray, output_dir=None, bounding_mask=None):
        """
        Args:
            image: numpy array of shape (512, 512, channels) in RGB format
            output_dir: optional directory to save visualization outputs
            bounding_mask: optional numpy array of shape (512, 512) with values 0 or 1
        Returns:
            mask: numpy array of shape (512, 512) with values 0 or 1
        """
        try:
            # Generate SAM masks
            sam_masks = self.mask_generator.generate(image)
            
            # Save visualization if output_dir provided
            if output_dir:
                os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
                
                annotate_image(
                    image,
                    sam_masks,
                    os.path.join(output_dir, "annotations", "annotated.jpg")
                )
            
            return sam_masks
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return np.zeros(image.shape[:2], dtype=np.uint8)


def get_all_contours(mask):
    """
    Extract all contours from a binary mask, organizing them into a hierarchical structure
    of polygons and their holes.
    
    Args:
        mask: Binary mask as numpy array
    
    Returns:
        List of lists where each sublist contains:
        - First element: exterior contour points
        - Subsequent elements (if any): hole contour points
    """
    # Ensure mask is binary and convert to uint8
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    
    # Find contours with hierarchy information
    contours, hierarchy = cv2.findContours(
        mask_uint8, 
        cv2.RETR_CCOMP,  # Retrieves all contours and organizes them into a two-level hierarchy
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0:
        return []
    
    # Initialize result structure
    result = []
    processed_holes = set()
    
    # Hierarchy structure: [Next, Previous, First_Child, Parent]
    if hierarchy is not None:
        hierarchy = hierarchy[0]  # Remove single-dimensional entry
        
        # Process each contour
        for i, contour in enumerate(contours):
            # Skip if this contour is a hole and we've already processed it
            if i in processed_holes:
                continue
            
            # If this is an exterior contour (parent == -1)
            if hierarchy[i][3] == -1:
                # Convert contour points to list format
                polygon = [point[0].tolist() for point in contour]
                polygon_with_holes = [polygon]
                
                # Find all holes for this contour
                child_idx = hierarchy[i][2]
                while child_idx != -1:
                    hole = [point[0].tolist() for point in contours[child_idx]]
                    polygon_with_holes.append(hole)
                    processed_holes.add(child_idx)
                    child_idx = hierarchy[child_idx][0]  # Move to next hole
                
                result.append(polygon_with_holes)
    
    return result


class SegmentationPipeline:
    def __init__(
        self,
        unet_model_path: str,
        sam_model_path: str,
        device: str,
        unet_tta: int = 1,
        sam_model_type: str = "vit_h"
    ):
        self.device = device
        
        # Initialize models
        self.unet = UNetModel(
            model_path=unet_model_path,
            device=device,
            tta=unet_tta
        )
        
        self.sam = SAMModel(
            model_path=sam_model_path,
            device=device,
            model_type=sam_model_type
        )
    
    def predict(self, image: np.ndarray, return_intermediate: bool = False):
        """
        Run full segmentation pipeline on a single image.
        
        Args:
            image: numpy array of shape (height, width, channels) in RGB format.
            return_intermediate: whether to return the UNet mask along with final mask and contours.
        
        Returns:
            If return_intermediate is False:
                (final_mask, contours): tuple where final_mask is a binary segmentation mask
                of shape (height, width), and contours is a list of polygon contours with holes.
            If return_intermediate is True:
                (unet_mask, final_mask, contours) with the UNet mask resized back to original dimensions.
        """
        original_shape = image.shape[:2]
        
        # Preprocess for UNet (1024x1024)
        unet_image = preprocess_image(image, (1024, 1024))
        unet_mask = self.unet.predict(
            unet_image, 
            output_dir=None
        )
        
        # Preprocess for SAM (512x512)
        sam_image = preprocess_image(image, (512, 512))
        sam_bound_mask = preprocess_mask(unet_mask, (512, 512))
        
        # Run SAM prediction
        sam_mask = self.sam.predict(
            sam_image,
            output_dir=None,
            bounding_mask=sam_bound_mask
        )
        
        # Combine SAM masks into one binary mask
        combined_mask = combine_masks(
            sam_mask,
            output_mask=None,
            bounding_mask=sam_bound_mask,
            resize_mask=None
        )

        # Restore original size
        final_mask = preprocess_mask(combined_mask, original_shape)
        final_mask = (final_mask > 0.5).astype(np.uint8)
        
        # Extract all contours with holes from final_mask
        contours = get_all_contours(final_mask)
        
        if return_intermediate:
            unet_mask_orig = preprocess_mask(unet_mask, original_shape)
            unet_mask_orig = (unet_mask_orig > 0.5).astype(np.uint8)
            return unet_mask_orig, final_mask, contours
        
        return final_mask, contours


if __name__ == "__main__":
    # This __main__ entrypoint is only for testing
    import time
    import json

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu"
    # device = torch.device(device)
    print(device)
    # Initialize pipeline
    pipeline = SegmentationPipeline(
        unet_model_path="../models/efficientnet.pt",
        # sam_model_path="../models/sam_vit_l_0b3195.pth", 
        sam_model_path="../models/sam_vit_h_4b8939.pth",
        # sam_model_type="vit_l", 
        sam_model_type="vit_h",
        device=device,
        unet_tta=1
    )

    data_dir = "../test_data/resiquant_200_addresses"
    data_dir = "../test_data/test"
    # Load and process single image

    def get_files(folder_path, file_extension=".jpg"):

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(file_extension):
                    yield file


    for filename in get_files(data_dir):
        full_file_path = os.path.join(data_dir, filename)
        image = cv2.cvtColor(cv2.imread(full_file_path), cv2.COLOR_BGR2RGB)
        
        tick = time.time()
        # Get predictions from the pipeline
        final_mask, contours = pipeline.predict(
            image,
            return_intermediate=False
        )
        tock = time.time()
        print(f"Time taken: {tock - tick} seconds")
        
        # Define output folders for annotation and mask saving
        output_folder = "../prediction/pipeline"
        annotations_dir = os.path.join(output_folder, "annotations")
        masks_dir = os.path.join(output_folder, "masks")
        polygon_dir = os.path.join(output_folder, "polygons")
        os.makedirs(annotations_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(polygon_dir, exist_ok=True)
        
        json.dump(contours, open(os.path.join(polygon_dir, filename.replace(".jpg", ".json")), "w"))

        # Annotate the original image using the final mask and the provided contours.
        annotate_and_save_image(
            image,
            final_mask,
            contours,
            os.path.join(annotations_dir, filename)
        )
        
        # Save the final mask as an image (scaling binary mask to 0-255 for saving)
        save_mask(final_mask, os.path.join(masks_dir, filename))

