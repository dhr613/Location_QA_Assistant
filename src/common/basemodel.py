"""Shared Pydantic base classes for structured agent outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentBaseModel(BaseModel):
    """Base model for structured outputs returned by LangGraph agents.

    This class centralizes common configuration for schemas that are exposed to
    LLM tool responses, ensuring consistent serialization and validation rules.
    Downstream models should inherit from this base rather than directly from
    ``pydantic.BaseModel``.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid", strict=True)


######################################### ROUTER#######################
class AgentOutput(BaseModel):
    """每个子agent的输出格式"""

    source: str
    result: str


class Classification(BaseModel):
    """一个单一的路由决策：用什么query来呼叫哪个agent。"""

    source: Literal["around_search_agent", "path_planning_agent"]
    query: str


class ClassificationResult(BaseModel):
    """将用户查询分类为特定于代理的子问题的结果。"""

    classifications: list[Classification] = Field(
        description="要调用的代理及其目标子问题列表"
    )


__all__ = ["AgentBaseModel"]
