from pydantic import BaseModel, Field
from typing import Optional

class FootprinterRequest(BaseModel):
    file_uri: Optional[str] = Field(default=None, description="S3 URI of the image file")
    image: Optional[str] = Field(default=None, description="Base64 encoded image string")

    class Config:
        json_schema_extra = {
            "example": {
                "file_uri": "s3://bucket/path/to/image.jpg",
                "image": "base64_encoded_string"
            }
        }

    @property
    def has_input(self):
        return bool(self.file_uri or self.image)

    def model_post_init(self, *args, **kwargs):
        if not self.has_input:
            raise ValueError("Either file_uri or image must be provided")

class FootprinterResponse(BaseModel):
    polygons: list[list[list[tuple[int, int]]]]