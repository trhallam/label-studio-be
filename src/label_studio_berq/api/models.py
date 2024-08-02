from typing import Dict, Any, List
from pydantic import BaseModel, Field, HttpUrl, AliasChoices


class SetupModel(BaseModel):
    project: str
    ls_schema: str = Field(alias=AliasChoices("schema", "ls_schema"))
    hostname: HttpUrl
    access_token: str
    extra_params: Dict[str, Any] | Any = Field(default=None)


class PredictModel(BaseModel):
    tasks: List[Dict]
    project: str  # f'{project.id}.{int(project.created_at.timestamp())}'
    label_config: str
    params: Dict
