mkdir -p models 
cd models
aws s3 cp s3://resiquant-ai-model-artifacts/weights/efficientnet.pt . 
aws s3 cp s3://resiquant-ai-model-artifacts/weights/sam_vit_h_4b8939.pth .
