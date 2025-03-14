from pydantic import BaseModel, Field
from typing import Optional


class ImageEvaluateRequest(BaseModel):
    file_uri: Optional[str] = Field(default=None, description="Cloud storage URI of the image file")
    image: Optional[str] = Field(
        default=None, 
        description="Base64 encoded image string. Can be either a JPEG or PNG image encoded in base64"
    )
    focal_length: Optional[float] = Field(
        default=None,
        description="Image focal length"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "file_uri": "s3://bucket/path/to/image.jpg",
                "image": "/9j/4AAQSkZJRg...base64_encoded_image_data...",  # Added example base64 string
                "focal_length": 400.0
            }
        }

    @property
    def has_input(self):
        return bool(self.file_uri or self.image)

    def model_post_init(self, *args, **kwargs):
        if not self.has_input:
            raise ValueError("Either file_uri or image must be provided")


class ImageEvaluateResponse(BaseModel):
    building_coverage: float
    sky_coverage: float
    building_visibility: float
    building_vertical_completeness: float
    building_horizontal_completeness: float
    image_quality: float
    building_top: float
    pounding_distance: float