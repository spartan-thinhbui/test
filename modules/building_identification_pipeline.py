# %% [markdown]
# ## Install dependencies

# %%
# !pip install groundingdino-py kornia

# %%
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.distance import cdist
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import time

if __name__ == '__main__':
    from models import GroundingDINOModel, SuperPoint, SAM2Model, DepthProModel, BaseModel
    from utils import compute_3d_distance, compute_bbox_area, find_nearest_bbox
else:
    from .models import GroundingDINOModel, SuperPoint, SAM2Model, DepthProModel, BaseModel
    from .utils import compute_3d_distance, compute_bbox_area, find_nearest_bbox

class BuildingIdentificationPipeline:
    def __init__(
        self,
        groundingdino_model_path: str,
        groundingdino_config_path: str,
        sam_model_path: str,
        depth_model_type: str,
        depth_model_path: str,
        superpoint_model_path: str,
        bbox_threshold: float = 0.35,
        text_threshold: float = 0.3,
        keypoint_feature_dim: int = 2048,
        building_feature_dim: int = 64,
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

        self.feature_dim = building_feature_dim

        self.superpoint = SuperPoint(model_path=superpoint_model_path, max_num_keypoints=keypoint_feature_dim).eval().to(device)
        self.bovw_cluster = KMeans(n_clusters=building_feature_dim, random_state=42)


    def get_bovw_features(self, descriptors: np.array) -> np.array:
        """
        Get batch of Bag of Visual Word features
        Args:
            descriptors: numpy array with shape (NxKxF) where N is the number images, K is the number of key points, F is the descriptor feature dimention
        Returns:
            numpy array (NxD): bag of visual words features of images
        """

        # Stack all descriptors together for clustering
        all_descriptors = np.vstack(descriptors)
        # Perform clustering
        self.bovw_cluster.fit(all_descriptors)

        # Function to compute histogram for each image
        def compute_histogram(descriptors):
            labels = self.bovw_cluster.predict(descriptors)  # Assign each descriptor to a cluster
            hist = np.bincount(labels, minlength=self.feature_dim)  # Count occurrences
            return hist / hist.sum() # Normalize histogram

        # Compute histograms for each image
        histogram_features = np.array([compute_histogram(des) for des in descriptors])
        return histogram_features


    @staticmethod
    def _compute_cluster_confidence(cluster_labels: np.array, features: np.array, n_clusters: int):
        """
        Compute confidence score for each feature in the cluster
        Args:
            cluster_labels: numpy array, clustering results
            features: numpy array of shape (NxD), feature
            n_clusters: number of clusters
        Returns:
            numpy array of floats: confidence scores
        """
        # Compute cluster centroids (mean of assigned points)
        centroids = np.array([features[cluster_labels == i].mean(axis=0) for i in range(n_clusters)])

        # Compute distance to cluster centroid for each point
        distances = np.array([cdist([f], [centroids[cluster_labels[i]]], metric='euclidean')[0][0] for i, f in enumerate(features)])

        # Normalize distances
        max_dist = distances.max()

        if max_dist != 0:
            confidence_scores = 1 - (distances / max_dist)
        else:
            confidence_scores = [1.0] * len(distances)

        return confidence_scores


    def classify_buildings(self, features: np.array):
        """
        Classify building features whether they are building of interest
        Args:
            features: np.array building features (NxD)
        Returns:
            list: list of predictions for each feature as dictionary with target building classification and its confidence score
        """
        n_clusters =  3
        self.building_cluster = AgglomerativeClustering(n_clusters=n_clusters)
        clustering = self.building_cluster.fit(features)

        # Find biggest clusters
        _, counts = np.unique(clustering.labels_, return_counts=True)
        biggest_clusters = np.argwhere(counts == np.max(counts)).flatten()

        confidence_scores = self._compute_cluster_confidence(clustering.labels_, features, n_clusters)
        res = []

        for label, score in zip(clustering.labels_, confidence_scores):
            res.append({
                "is_target": label in biggest_clusters,
                "confidence": score
            })
        
        return res


    def get_image_building_distance(self, image: np.array, focal_length: float, buildings_info: dict):
        h, w = image.shape[:2]
        cx, cy = int(w/2), int(h/2)

        target_buildings_bboxes = []
        neighbor_buildings_bboxes = []

        # get target building and its nearest neighbor
        for building in buildings_info:
            if building['is_target']:
                target_buildings_bboxes.append(building['bbox'])
            else:
                neighbor_buildings_bboxes.append(building['bbox'])
        
        if len(target_buildings_bboxes) == 0 or len(neighbor_buildings_bboxes) == 0:
            return -1
        
        min_building = np.argmin([compute_bbox_area(bbox) for bbox in target_buildings_bboxes])
        target_bbox = target_buildings_bboxes[min_building]

        nearest_bbox = find_nearest_bbox(neighbor_buildings_bboxes, target_bbox)

        # segment buildings
        masks = self.segmentaiton_model.predict(
            image=image,
            bboxes=np.array([target_bbox, nearest_bbox]),
            erode=True
        )

        # estimate depth map
        depth_map = self.depth_estimation_model.predict(image, focal_length=focal_length)

        # calculate 3d distance between 2 buildings
        fx = fy = focal_length
        min_distance = compute_3d_distance(depth_map, masks[0], masks[1], fx, fy, cx, cy)

        return min_distance


    def predict_batch(self, batch: list, image_ids: list, focal_length: float=None) -> dict:
        """
        Identify buildings in batch of images
        
        Args:
            batch: list of numpy arrays of shape (height, width, channels) batch of images.
            image_ids: list of ids for each image in batch
        Returns:
            list: list of predictions as dictionary with image id, building bounding box, building classification, and confidence score
        """

        buildings = []
        image_building_map = []

        for i, image in enumerate(batch):
            target_bounding_boxes = self.object_detect_model.predict(image, ["single building"], self.bbox_threshold, self.text_threshold)
            building_bboxes = target_bounding_boxes["single building"]

            for bounding_box in building_bboxes:
                xmin, ymin, xmax, ymax = bounding_box
                building = image[int(ymin):int(ymax), int(xmin):int(xmax)]
                buildings.append(building)
                image_building_map.append({
                    'image_id': i,
                    'bbox': bounding_box,
                })

        # create building features
        descriptors = []
        transform = transforms.ToTensor()

        for building in buildings:
            feat = self.superpoint.extract(transform(building).to(self.device))
            descriptors.append(feat["descriptors"].cpu().numpy()[0])
        
        building_features = self.get_bovw_features(descriptors)

        # classify buildings
        classify_results = self.classify_buildings(building_features)

        response = []
        for i, image in enumerate(batch):
            target_building_exist = False
            neighbor_exist = False 
            buildings_info = []
            for j, result in enumerate(classify_results):
                if image_building_map[j]['image_id'] == i:
                    buildings_info.append({
                        "bbox": image_building_map[j]['bbox'],
                        "is_target": result['is_target']
                    })

                    target_building_exist |= result['is_target']
                    neighbor_exist |= not result['is_target']
        
            distance = -1
            if target_building_exist and neighbor_exist:
                distance = self.get_image_building_distance(image, focal_length, buildings_info)

            response.append({
                "id": image_ids[i],
                "target": target_building_exist,
                "neighbor_presence": neighbor_exist,
                "min_distance": distance
            })


        return response


# %%
if __name__ == "__main__":
    import time
    from pathlib import Path


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    pipeline = BuildingIdentificationPipeline(
        groundingdino_config_path="../configs/GroundingDINO_SwinB.cfg.py",
        groundingdino_model_path="../models/groundingdino_swinb_cogcoor.pth",
        superpoint_model_path="../models/superpoint_v1.pth",
        sam_model_path="../models/sam2.1_hiera_base_plus.pt",
        depth_model_path="../models/depth_pro.pt",
        depth_model_type="dinov2l16_384",
        bbox_threshold=0.25,
        text_threshold=0.25,
        keypoint_feature_dim=2048,
        building_feature_dim=64,
        device=device
    )

    path = "../test_data/file_store_2024_02_10/"
    dataset_dir = Path(path)
    output_dir = "../analysis/debug_output"
    # dataset_dir = Path("../file_store/")
    output_json = {}

    address = "1_front_st_san_francisco_ca_94111"
    images = []
    for filename in dataset_dir.glob(f"{address}/imagery/street_level/*.jpg"):
        image = cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)
        images.append(image)

    tick = time.time()

    test_batch_size = 4
    # Get predictions from the pipeline
    results = pipeline.predict_batch(
        images[:test_batch_size],
        [f"{i}" for i in range(test_batch_size)],
        focal_length=800
    )
    print(results)
    tock = time.time()
    print(f"Time taken: {tock - tick} seconds")
# %%
