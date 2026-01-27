import asyncio
import os

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import AIMessage

from langchain_dev_utils.chat_models import register_model_provider, load_chat_model

from dotenv import load_dotenv

from common.tools import (
    district_search,
    around_search,
    driving_route,
    jump_to_around_search_agent_multi,
    jump_to_path_planning_agent_multi,
)
from common.prompts import (
    AROUND_SEARCH_HANDOFF_MULTI_SYSTEM_PROMPT,
    PATH_PLANNING_HANDOFF_MULTI_SYSTEM_PROMPT,
)
from react_agent.state import Gaodemap_State_Handoff_Multi
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal

load_dotenv()

register_model_provider(
    provider_name="dhrzhipu",
    chat_model="openai-compatible",
    base_url=os.getenv("DHRZHIPU_BASE_URL"),
    compatibility_options={
        "support_tool_choice": ["auto"],
        "support_response_format": ["json_schema"],
    },
    # model_profiles=_PROFILES
)

model = load_chat_model(
    model="dhrzhipu:glm-4.5",
    # model = "dhrark:doubao-1-5-pro-32k-250115",
    extra_body={
        "thinking": {"type": "disabled"},
    }
)

around_search_agent = create_agent(
    model=model,
    tools=[district_search, around_search],
    system_prompt=AROUND_SEARCH_HANDOFF_MULTI_SYSTEM_PROMPT
)

path_planning_agent = create_agent(
    model=model,
    tools=[district_search, driving_route],
    system_prompt=PATH_PLANNING_HANDOFF_MULTI_SYSTEM_PROMPT
)


@tool("around_search_agent")
async def call_around_search_agent(query: str):
    """
    这里利用对某一区域进行周边区域地点检索的工具。你输入的参数需要明确是哪个地点或区域，明确省，市的单位，例如：成都武侯区的洗浴中心
    """
    result = await around_search_agent.ainvoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


@tool("path_planning_agent")
async def call_path_planning_agent(query: str):
    """
    这里利用对起点和终点进行路径规划的工具。你输入的参数需要明确是哪个地点或区域，明确省，市的单位，例如：成都天府三街到东郊记忆
    """
    result = await path_planning_agent.ainvoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


around_node = create_agent(
    model=model,
    tools=[call_around_search_agent, jump_to_path_planning_agent_multi],
    system_prompt=AROUND_SEARCH_HANDOFF_MULTI_SYSTEM_PROMPT
)

path_node = create_agent(
    model=model,
    tools=[call_path_planning_agent, jump_to_around_search_agent_multi],
    system_prompt=PATH_PLANNING_HANDOFF_MULTI_SYSTEM_PROMPT
)


async def call_around_node(state: Gaodemap_State_Handoff_Multi):
    response = await around_node.ainvoke(state)
    return response


async def call_path_node(state: Gaodemap_State_Handoff_Multi):
    response = await path_node.ainvoke(state)
    return response


def route_after_agent(
    state: Gaodemap_State_Handoff_Multi,
) -> Literal["call_around_node", "call_path_node", "__end__"]:
    """根据当前状态判断下一步的执行"""
    messages = state.get("messages", [])
    print("当前处于路由节点：", state.get("current_step"))
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
            return "__end__"

    active = state.get("current_step", "call_around_node")
    return active if active else "call_around_node"


def route_initial(
    state: Gaodemap_State_Handoff_Multi,
) -> Literal["call_around_node", "call_path_node"]:
    """根据当前状态判断下一步的执行"""
    print("当前处于初始化路由节点：", state.get("current_step"))
    return state.get("current_step") or "call_around_node"


builder = StateGraph(Gaodemap_State_Handoff_Multi)
builder.add_node("call_around_node", call_around_node)
builder.add_node("call_path_node", call_path_node)

builder.add_conditional_edges(START, route_initial, ["call_around_node", "call_path_node"])

builder.add_conditional_edges(
    "call_around_node", route_after_agent, ["call_around_node", "call_path_node", "__end__"]
)
builder.add_conditional_edges(
    "call_path_node", route_after_agent, ["call_around_node", "call_path_node", "__end__"]
)

graph = builder.compile()

if __name__ == "__main__":
    query = "从成都保利星座到东郊记忆怎么走"
    config = {"configurable": {"thread_id": "test_3"}}

    async def main():
        async for step in graph.astream(
            {"messages": [{"role": "user", "content": query}]},
                # config=config
            ):
                for update in step.values():
                    for message in update.get("messages", []):
                        print(message.content)
                        print("-"*100)

    asyncio.run(main())
