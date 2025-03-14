mkdir -p models 
cd models
aws s3 cp s3://resiquant-ai-model-artifacts/weights/groundingdino_swinb_cogcoor.pth .
aws s3 cp s3://resiquant-ai-model-artifacts/weights/sam_vit_h_4b8939.pth .
#aws s3 cp s3://resiquant-ai-model-artifacts/weights/superpoint_v1.pth .
aws s3 cp s3://resiquant-ai-model-artifacts/weights/sam2.1_hiera_base_plus.pt .
aws s3 cp s3://resiquant-ai-model-artifacts/weights/sam2.1_hiera_large.pt .
aws s3 cp s3://resiquant-ai-model-artifacts/weights/depth_pro.pt .