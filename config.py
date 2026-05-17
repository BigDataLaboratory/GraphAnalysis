from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class LogConfig(BaseModel):
    filepath: str
    level: str = "INFO"


# Graph generation — input

class SourceConf(BaseModel):
    uri: Optional[str] = None
    db_name: Optional[str] = None
    collection: Optional[str] = None
    chunk_start_date: str
    chunk_end_date: str
    method: str
    username: Optional[str] = None
    password: Optional[str] = None
    authName: Optional[str] = None
    authMechanism: Optional[str] = None

    @field_validator("chunk_start_date", "chunk_end_date", mode="before")
    @classmethod
    def parse_date(cls, v: str) -> str:
        # Validate format eagerly; keep as string so datetime is built later
        datetime.strptime(v, "%d/%m/%Y")
        return v

    @property
    def start_date(self) -> datetime:
        return datetime.strptime(self.chunk_start_date, "%d/%m/%Y")

    @property
    def end_date(self) -> datetime:
        return datetime.strptime(self.chunk_end_date, "%d/%m/%Y")

class SourceInput(BaseModel):
    type: str
    conf: SourceConf

class GraphTypes(BaseModel):
    tweet_retweet: bool = False
    user_hashtag: bool = False
    hashtag_cooccurrences: bool = False
    retweet: bool = False
    response: bool = False
    mention: bool = False
    user_user: bool = False

class GraphFileNames(BaseModel):
    retweet: str = "retweet"
    mention: str = "mention"
    retweet_user: str = "retweet_user"
    response: str = "response"
    response_user: str = "response_user"
    hashtag: str = "hashtag"
    hashtag_cooccurrences: str = "hashtag_cooccurrences"
    multigraph: str = "multigraph"

class GraphOutput(BaseModel):
    path: str
    graph_file_name: GraphFileNames
    map_file_name_prefix: str = "map"
    # Support legacy single `file_format` or new split formats
    intermediate_file_format: str = "parquet"
    final_file_format: str = "parquet"

    @model_validator(mode="before")
    @classmethod
    def resolve_file_formats(cls, values: dict) -> dict:
        legacy = values.get("file_format")
        if legacy:
            values.setdefault("intermediate_file_format", legacy)
            values.setdefault("final_file_format", legacy)
        return values

class LoadSnapshot(BaseModel):
    status: bool = False
    tmp_path: str = ""

class GraphParameters(BaseModel):
    checkpoint_every: int = 5000
    n_workers: int = 4
    fast_rt_threshold: int = 60
    is_community: bool
    community_file: str
    community_strategy: str
    community_batch_size: int
    load_snapshot: Optional[LoadSnapshot] = LoadSnapshot()
    input: SourceInput
    graph_type: GraphTypes
    output: GraphOutput

class GraphGenerationConfig(BaseModel):
    to_execute: bool
    delete_tmp_after_merge: bool = False
    parameters: GraphParameters

    # Inject env-variable overrides for MongoDB credentials
    @model_validator(mode="after")
    def apply_env_overrides(self) -> "GraphGenerationConfig":
        conf = self.parameters.input.conf
        conf.uri           = os.getenv("MONGO_URI",           conf.uri)
        conf.username      = os.getenv("MONGO_USERNAME",      conf.username)
        conf.password      = os.getenv("MONGO_PASSWORD",      conf.password)
        conf.db_name       = os.getenv("MONGO_DATABASE",      conf.db_name)
        conf.collection    = os.getenv("MONGO_COLLECTION",    conf.collection)
        conf.authName      = os.getenv("MONGO_AUTH_SOURCE",   conf.authName)
        conf.authMechanism = os.getenv("MONGO_AUTH_MECHANISM", conf.authMechanism)
        return self


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------

class CommunitySharedParameters(BaseModel):
    read_from_edge_list: bool = False
    read_from_file: bool = False
    temporal: bool = False
    graph_file_path: List[str] = []
    pickle_graph_file_path: str = ""

class LeidenParameters(BaseModel):
    community_output_file_path: str = ""

class LeidenConfig(BaseModel):
    to_execute: bool = False
    parameters: LeidenParameters = LeidenParameters()

class ComboParameters(BaseModel):
    community_output_file_path: str = ""

class ComboConfig(BaseModel):
    to_execute: bool = False
    parameters: ComboParameters = ComboParameters()

class HierarchicalParameters(BaseModel):
    first_level_communities_file: str = ""
    community_col_name: str = ""
    community_output_file_path: str = ""
    top_k: int = 500
    collapse_nodes: bool = True
    min_size: int = 30
    pickle_output_path: str = ""   # replaces the hard-coded path in GraphAnalysis

class HierarchicalConfig(BaseModel):
    to_execute: bool = False
    parameters: HierarchicalParameters = HierarchicalParameters()

class CommunityDetectionConfig(BaseModel):
    parameters: CommunitySharedParameters
    leiden: LeidenConfig
    combo: ComboConfig
    hierarchical: HierarchicalConfig


# ---------------------------------------------------------------------------
# Get users text
# ---------------------------------------------------------------------------

class CommunitiesConfig(BaseModel):
    indexes: List[int] = []
    community_col_name: str = ""
    read_communities_from_file: bool = False
    community_file_path: str = ""

class MapFileConfig(BaseModel):
    read_maps_from_file: bool = False
    user_map: str = ""
    hashtag_map: str = ""
    retweet_user_map: str = ""

class TextDataConf(BaseModel):
    uri: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    authName: Optional[str] = None
    authMechanism: Optional[str] = None
    db_name: Optional[str] = None
    collection: Optional[str] = None

class TextDataInput(BaseModel):
    type: str
    conf: TextDataConf

class GetUsersTextParameters(BaseModel):
    communities: CommunitiesConfig
    map_file_path: MapFileConfig
    input: TextDataInput

class GetUsersTextConfig(BaseModel):
    to_execute: bool = False
    parameters: GetUsersTextParameters


# ---------------------------------------------------------------------------
# Topic builder
# ---------------------------------------------------------------------------

class TopicModel(BaseModel):
    model_path: str
    serialization: str = "pickle"

class TopicBuilderParameters(BaseModel):
    topics_file_path: str
    docs_file_path: str
    model: TopicModel

class TopicBuilderConfig(BaseModel):
    to_execute: bool = False
    parameters: TopicBuilderParameters


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class AppConfig(BaseModel):
    log: LogConfig
    graph_generation: GraphGenerationConfig
    community_detection: CommunityDetectionConfig
    get_users_text: GetUsersTextConfig
    topic_builder: TopicBuilderConfig

    @classmethod
    def from_yaml(cls, path: str | Path = "properties/prop.yaml") -> "AppConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)