from pydantic import BaseModel, Field


class BuildExpertCommand(BaseModel):
    """Model for expert description"""

    name: str = Field(
        description="Name or identifier of the expert",
        # examples=["Dr. Alex Martinez", "Prof. Sarah Johnson"],
    )
    description: str = Field(
        description="Brief description of the expert's specialties and background",
        # examples=[
        #     "Expert in computer vision with focus on image segmentation techniques",
        #     "Specialist in reinforcement learning applications for robotic control",
        # ],
    )
    query: str = Field(
        description="Query to search for surveys about the expert's topic",
        # examples=[
        #     "image segmentation deep learning survey recent advances",
        #     "reinforcement learning robotics survey state-of-the-art",
        # ],
    )
