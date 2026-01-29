import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_dev_utils.chat_models import load_chat_model, register_model_provider
from langgraph.checkpoint.memory import InMemorySaver

from common.prompts import (
    AROUND_SEARCH_HANDOFF_SINGLE_V2_SYSTEM_PROMPT,
    MAIN_HANDOFF_SINGLE_V2_SYSTEM_PROMPT,
    DRIVING_ROUTE_INNER_SYSTEM_PROMPT,
    AROUND_SEARCH_INNER_SYSTEM_PROMPT,
    DRIVING_ROUTE_HANDOFF_SINGLE_V2_SYSTEM_PROMPT,
)
from common.tools import (
    around_search,
    district_search,
    driving_route_handoff_single_v2,
    jump_to_around_search_agent,
    jump_to_path_planning_agent,
)
from react_agent.state import Gaodemap_State_Handoff_Single_V2

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
    system_prompt=AROUND_SEARCH_INNER_SYSTEM_PROMPT,
)

path_planning_agent = create_agent(
    model=model,
    tools=[district_search, driving_route_handoff_single_v2],
    system_prompt=DRIVING_ROUTE_INNER_SYSTEM_PROMPT,
)


@tool("around_search_agent")
async def call_around_search_agent(query: str):
    """
    这里利用对某一区域进行周边区域地点检索的工具。你输入的参数需要明确是哪个地点或区域，明确省，市的单位，例如：成都武侯区的洗浴中心
    """
    result = await around_search_agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result["messages"][-1].content


@tool("path_planning_agent")
async def call_path_planning_agent(query: str):
    """
    这里利用对起点和终点进行路径规划的工具。你输入的参数需要明确是哪个地点或区域，明确省，市的单位，例如：成都天府三街到东郊记忆
    """
    result = await path_planning_agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result["messages"][-1].content


tools = [
    call_around_search_agent,
    call_path_planning_agent,
    jump_to_around_search_agent,
    jump_to_path_planning_agent,
]


STEP_CONFIG = {
    "main_step": {
        "prompt": MAIN_HANDOFF_SINGLE_V2_SYSTEM_PROMPT,
        "tools": [jump_to_around_search_agent, jump_to_path_planning_agent],
    },
    "around_search_agent_step": {
        "prompt": AROUND_SEARCH_HANDOFF_SINGLE_V2_SYSTEM_PROMPT,
        "tools": [call_around_search_agent, jump_to_path_planning_agent],
    },
    "path_planning_agent_step": {
        "prompt": DRIVING_ROUTE_HANDOFF_SINGLE_V2_SYSTEM_PROMPT,
        "tools": [call_path_planning_agent, jump_to_around_search_agent],
    },
}

from typing import Callable

from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call


@wrap_model_call
async def apply_step_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    current_step = request.state.get("current_step", "main_step")
    print(f"当前阶段为：{current_step}")

    stage_config = STEP_CONFIG[current_step]

    # for key in stage_config["requires"]:
    #     if request.state.get(key) is None:
    #         raise ValueError(f"{key} must be set before reaching {current_step}")

    system_prompt = stage_config["prompt"].format(**request.state)
    print(f"系统提示词为：{system_prompt[0:100]}")
    print("当前可调用工具为：", [tool.name for tool in stage_config["tools"]])
    request = request.override(
        system_prompt=system_prompt,
        tools=stage_config["tools"],
    )

    return await handler(request)


all_tools = [
    call_around_search_agent,
    call_path_planning_agent,
    jump_to_around_search_agent,
    jump_to_path_planning_agent,
]

agent = create_agent(
    model,
    tools=all_tools,
    state_schema=Gaodemap_State_Handoff_Single_V2,
    middleware=[apply_step_config],
    # checkpointer=InMemorySaver(),
)

if __name__ == "__main__":
    query = "从成都保利星座到东郊记忆怎么走，那里有什么好玩的吗"
    config = {"configurable": {"thread_id": "test_4"}}

    async def main():
        async for step in agent.astream(
            {"messages": [{"role": "user", "content": query}]},
            # config=config
        ):
            for update in step.values():
                for message in update.get("messages", []):
                    print(message.content)
                    print("-" * 100)

    asyncio.run(main())
