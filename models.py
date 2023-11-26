from enum import Enum
from typing import Optional, List

from beanie import Document
# from pydantic import BaseModel


# class TagColors(str, Enum):
#     RED = "RED"
#     BLUE = "BLUE"
#     GREEN = "GREEN"


# class Tag(BaseModel):
#     name: str
#     color: TagColors = TagColors.BLUE


class Note(Document):  # This is the document structure
    title: str
    text: Optional[str]
    # tag_list: List[Tag] = []



# from typing import Optional
# from beanie import PydanticObjectId
# from pydantic import BaseModel, Field

# from src.models import Profession, Pet


# class Pet(BaseModel):
#     name: Optional[str] = None


# class BasePerson(BaseModel):
#     name: Optional[str] = None
#     email: str
#     age: int
#     profession: Optional[Profession] = None
#     pet: Optional[Pet] = None
#     country: Optional[str] = None


# class PersonCreate(BasePerson):
#     pass


# class PersonRead(BasePerson):
#     id: PydanticObjectId = Field()