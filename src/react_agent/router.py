import os
import uuid
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain_dev_utils.chat_models import load_chat_model, register_model_provider
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import Annotated, NotRequired

from common.basemodel import ClassificationResult
from common.prompts import (
    AROUND_SEARCH_ROUTER_SYSTEM_PROMPT,
    CLASSIFY_QUERY_PROMPT,
    PATH_PLANNING_ROUTER_SYSTEM_PROMPT,
)
from common.tools import around_search, district_search, driving_route
from react_agent.state import Gaodemap_State_Router

load_dotenv()

register_model_provider(
    provider_name="dhrzhipu",
    chat_model="openai-compatible",
    base_url=os.getenv("DHRZHIPU_BASE_URL"),
    compatibility_options={
        "support_tool_choice": ["auto"],
        "support_response_format": [
            "json_schema",
        ],
    },
    # model_profiles=_PROFILES
)

model = load_chat_model(
    model="dhrzhipu:glm-4.7",
    # model = "dhrark:doubao-1-5-pro-32k-250115",
    extra_body={
        "thinking": {"type": "disabled"},
    },
)


around_search_agent = create_agent(
    model=model,
    tools=[district_search, around_search],
    system_prompt=AROUND_SEARCH_ROUTER_SYSTEM_PROMPT,
)


path_planning_agent = create_agent(
    model=model,
    tools=[district_search, driving_route],
    system_prompt=PATH_PLANNING_ROUTER_SYSTEM_PROMPT,
)


# @tool("around_search_agent")
async def call_around_search_agent(state: Gaodemap_State_Router):
    """
    这里利用对某一区域进行周边区域地点检索的工具。你输入的参数需要明确是哪个地点或区域，明确省，市的单位，例如：成都武侯区的洗浴中心
    """
    print("当前处于节点around_search_agent")
    query = state["query"]
    result = await around_search_agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return {
        "results": [
            {"source": "around_search_agent", "result": result["messages"][-1].content}
        ]
    }


# @tool("path_planning_agent")
async def call_path_planning_agent(state: Gaodemap_State_Router):
    """
    这里利用对起点和终点进行路径规划的工具。你输入的参数需要明确是哪个地点或区域，明确省，市的单位，例如：成都天府三街到东郊记忆
    """
    print("当前处于节点path_planning_agent")
    query = state["query"]
    result = await path_planning_agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return {
        "results": [
            {"source": "path_planning_agent", "result": result["messages"][-1].content}
        ]
    }


async def classify_query(state: Gaodemap_State_Router) -> dict:
    """Classify query and determine which agents to invoke."""
    structured_llm = model.with_structured_output(ClassificationResult)

    result = structured_llm.invoke(
        [
            {"role": "system", "content": CLASSIFY_QUERY_PROMPT},
            {"role": "user", "content": state["query"]},
        ]
    )

    return {"classifications": result.classifications}


def route_to_agents(state: Gaodemap_State_Router) -> list[Send]:
    """根据分类将任务分派给各代理。"""

    print(state["classifications"])
    return [Send(c.source, {"query": c.query}) for c in state["classifications"]]


def synthesize_results(state: Gaodemap_State_Router) -> dict:
    """将所有代理的结果组合成一个连贯的答案。"""
    if not state["results"]:
        return {"final_answer": "没有找到任何知识来源的结果。"}

    formatted = [
        f"**From {r['source'].title()}:**\n{r['result']}" for r in state["results"]
    ]

    synthesis_response = model.invoke(
        [
            {
                "role": "system",
                "content": f"""综合这些搜索结果来回答最初的问题: "{state["query"]}"

- 整合多个来源的信息，避免冗余
- 突出最相关且可操作的信息
- 记录不同来源之间的差异之处
- 确保回复简洁明了、条理清晰""",
            },
            {"role": "user", "content": "\n\n".join(formatted)},
        ]
    )

    return {"final_answer": synthesis_response.content}


workflow = (
    StateGraph(Gaodemap_State_Router)
    .add_node("classify", classify_query)
    .add_node("around_search_agent", call_around_search_agent)
    .add_node("path_planning_agent", call_path_planning_agent)
    .add_node("synthesize", synthesize_results)
    .add_edge(START, "classify")
    .add_conditional_edges(
        "classify", route_to_agents, ["around_search_agent", "path_planning_agent"]
    )
    .add_edge("around_search_agent", "synthesize")
    .add_edge("path_planning_agent", "synthesize")
    .add_edge("synthesize", END)
    .compile(checkpointer=InMemorySaver())
)

if __name__ == "__main__":
    pass
