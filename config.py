# model config
groundingdino_config_path="./configs/GroundingDINO_SwinB.cfg.py"
groundingdino_model_path="./models/groundingdino_swinb_cogcoor.pth"
sam_model_path="./models/sam2.1_hiera_large.pt"
sam_model_type="vit_h"
brisque_model_path="./configs/brisque_model_live.yml"
brisque_range_path="./configs/brisque_range_live.yml"
bbox_threshold=0.4
text_threshold=0.25
building_top_max_gap=10  # pixels

superpoint_model_path="./models/superpoint_v1.pth"
depth_model_path="./models/depth_pro.pt"
depth_model_type="dinov2l16_384"
keypoint_feature_dim=2048
building_feature_dim=64
n_clusters = 2
building_bbox_threshold=0.4
building_text_threshold=0.3


# server.py config
local_num_devices = 1
local_workers_per_device = 1
server_num_devices = 1
server_workers_per_device = 4