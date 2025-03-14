from typing import Union
from enum import Enum
import numpy as np
from scipy.spatial.distance import cdist
import cv2

def get_contour_points(mask):
    """
    Extracts contour points from a binary segmentation mask.

    :param mask: 2D numpy array (binary segmentation mask)
    :return: List of (x, y) coordinates of contour points
    """
    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_points = np.vstack([c.squeeze() for c in contours if len(c) > 0])  # Flatten list of contours
    return contour_points if len(contour_points) > 0 else np.array([])


def bbox_center(bbox):
    """Calculate the center (x, y) of a bounding box."""
    x1, y1, x2, y2 = bbox  # (xmin, ymin, xmax, ymax)
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def find_nearest_bbox(bboxes, target_bbox):
    """Find the nearest bounding box to the target."""
    target_center = bbox_center(target_bbox)
    
    distances = []
    for bbox in bboxes:
        center = bbox_center(bbox)
        distance = np.linalg.norm(np.array(target_center) - np.array(center))
        distances.append(distance)
    
    nearest_index = np.argmin(distances)  # Get index of the closest bbox
    return bboxes[nearest_index], nearest_index


def compute_3d_distance(depth_map, mask1, mask2, fx, fy, cx, cy, k=1000):
    """Finds the minimum 3D distance between two segmented objects in a depth map.
    
    Parameters:
    - depth_map: Absolute depth map (meters)
    - fx, fy: Focal lengths of the camera
    - cx, cy: Principal point of the camera
    - k: top-k neareast contours to be compared

    Returns:
    - Minimum 3D distance between the two objects
    """
    # Find contours
    contours1 = get_contour_points(mask1)
    contours2 = get_contour_points(mask2)

    if len(contours1) < 2 or len(contours2) < 2:
        print("Not enough objects detected!")
        return None

    # Convert 2D contour points to 3D coordinates
    def contour_to_3d(contour, depth_map):
        points_3d = []
        for pt in contour.reshape(-1, 2):
            x, y = pt

            z = depth_map[y, x]  # Depth value at (x, y)
            if z > 0:  # Ignore invalid depth
                X = (x - cx) * z / fx  # Convert pixel to real-world coordinates
                Y = (y - cy) * z / fy
                points_3d.append([X, Y, z])
        return np.array(points_3d)

    obj1_3d = contour_to_3d(contours1, depth_map)
    
    obj2_3d = contour_to_3d(contours2, depth_map)

    if obj1_3d.size == 0 or obj2_3d.size == 0:
        print("No valid 3D points found.")
        return None

    # Compute minimum distances between the two sets of 3D points
    distances = cdist(obj1_3d, obj2_3d)

    sorted_indices = np.argsort(distances, axis=None)[:k] # Get top K indices
    row_indices, col_indices = np.unravel_index(sorted_indices, distances.shape)

    # Get the minimum distances
    min_distances = [distances[i, j] for i, j in zip(row_indices, col_indices)]
    min_distance = np.mean(min_distances)

    return min_distance


def is_bbox_inside_image_vertical(bbox: Union[np.array,list,tuple], image_shape: Union[np.array,list,tuple], offset: int=0) -> bool:
    """
    Check if a bounding box stays within an image height by an offset.
    
    Args:
        bbox: np.array, list or tuple (x_min, y_min, x_max, y_max) - Bounding box coordinates
        image_shape: tuple (height, width) - Image dimensions
    
    Returns:
        bool: True if the bounding box is inside the image vertically, False otherwise
    """
    x_min, y_min, x_max, y_max = bbox
    height, width = image_shape[:2]
    
    return offset < y_min and y_max < height - offset


def is_bbox_inside_image_horizontal(bbox: Union[np.array,list,tuple], image_shape: Union[np.array,list,tuple], offset: int=0) -> bool:
    """
    Check if a bounding box stays within an image width by an offset.
    
    Args:
        bbox: tuple (x_min, y_min, x_max, y_max) - Bounding box coordinates
        image_shape: tuple (height, width) - Image dimensions
    
    Returns:
        bool: True if the bounding box is inside the image horizontally, False otherwise
    """
    x_min, y_min, x_max, y_max = bbox
    height, width = image_shape[:2]
    
    return offset < x_min and x_max < width - offset

def compute_mask_area_ratio(mask: np.array) -> float:
    """
    Check if two segmentation masks intersect over a area ratio

    Args:
        mask (numpy.ndarray): First binary mask (same shape as mask2).

    Returns:
        bool: True if there is an intersection, False otherwise.
    """
    object_area = np.sum(mask > 0)
    image_area = mask.shape[0] * mask.shape[1]

    return object_area / image_area


def compute_mask_intersect_ratio(mask1: np.array, mask2: np.array) -> float:
    """
    Compute mask intersection ratio of mask2 over mask1

    Args:
        mask1 (numpy.ndarray): First binary mask (same shape as mask2).
        mask2 (numpy.ndarray): Second binary mask.
    Returns:
        bool: True if there is an intersection, False otherwise.
    """
    intersection = np.logical_and(mask1, mask2)
    intersect_area = np.sum(intersection)

    mask1_area = np.sum(mask1 > 0)

    return intersect_area / mask1_area


def compute_bbox_area(bbox: Union[np.array,list,tuple]) -> int:
    """
    Compute a bounding box area

    Args:
        bbox (np.array|list|tuple): bounding box as x_min, y_min, x_max, y_max
    Returns:
        float: area
    """

    x_min, y_min, x_max, y_max = bbox

    width = max(0, x_max - x_min)  # Ensure non-negative width
    height = max(0, y_max - y_min)  # Ensure non-negative height
    return width * height


def compute_bboxes_intersect_ratio(bboxes1: list, bboxes2: list) -> float:
    """
    Compute bounding boxes intersection ratio of bounding boxes of object1 and bounding boxes of object2

    Args:
        bboxes1 (list[tuple|list|np.array]): list of bounding boxes for object 1
        bboxes2 (list[tuple|list|np.array]): list of bounding boxes for object 2
    Returns:
        bool: True if there is an intersection, False otherwise.
    """

    intersect_area = 0
    object1_area = 0

    for i, bbox1 in enumerate(bboxes1):
        object1_area += compute_bbox_area(bbox1)
        for j, bbox2 in enumerate(bboxes2):
            x1_min, y1_min, x1_max, y1_max = bbox1
            x2_min, y2_min, x2_max, y2_max = bbox2

            # Compute intersection coordinates
            x_int_min = max(x1_min, x2_min)
            y_int_min = max(y1_min, y2_min)
            x_int_max = min(x1_max, x2_max)
            y_int_max = min(y1_max, y2_max)

            # Compute intersection area
            inter_width = max(0, x_int_max - x_int_min)
            inter_height = max(0, y_int_max - y_int_min)
            intersect_area += inter_width * inter_height

    return intersect_area / object1_area if object1_area > 0 else 0.0


def create_bbox_mask(image_shape: Union[np.array,list,tuple], bbox_list: list):
    """
    Create a binary mask with multiple bounding boxes.

    Args:
        image_shape (tuple): (height, width) of the image.
        bbox_list (list): List of bounding boxes [(x_min, y_min, x_max, y_max), ...]

    Returns:
        mask (numpy array): Binary mask with bounding boxes filled with 1.
    """
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for bbox in bbox_list:
        x_min, y_min, x_max, y_max = bbox

        # Ensure coordinates are within image boundaries
        x_min, x_max = max(0, int(x_min)), min(width, int(x_max))
        y_min, y_max = max(0, int(y_min)), min(height, int(y_max))

        # Fill the bounding box region with 1
        mask[y_min:y_max, x_min:x_max] = 1

    return mask