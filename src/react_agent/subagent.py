import os
from typing import Callable

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain.tools.tool_node import ToolCallRequest
from langchain_dev_utils.chat_models import load_chat_model, register_model_provider
from langgraph.types import Command

from common.prompts import (
    AROUND_SEARCH_SUBAGENT_SYSTEM_PROMPT,
    DISTRICT_SEARCH_SUBAGENT_SYSTEM_PROMPT,
    MAIN_SYSTEM_PROMPT,
    PATH_PLANNING_SUBAGENT_SYSTEM_PROMPT,
    TRAVEL_GUIDE_SUBAGENT_SYSTEM_PROMPT,
)
from common.tools import (
    around_search,
    calculate_distance,
    district_search,
    driving_route,
    geocode,
    walking_route,
)

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
    model="dhrzhipu:glm-4.5",
    # model = "dhrark:doubao-1-5-pro-32k-250115",
    extra_body={
        "thinking": {"type": "disabled"},
    },
)

district_search_agent = create_agent(
    model=model,
    tools=[district_search],
    system_prompt=DISTRICT_SEARCH_SUBAGENT_SYSTEM_PROMPT,
)


@tool("district_search_agent")
async def call_district_search_agent(query: str):
    """
    这里的关键字是对用户的目的进行大类（如果有的话），中类（如果有的话），小类（如果有的话）的描述。
    例如：大类包括生活服务，金融服务，政府机构等（大类没有固定的名称）。中类则是稍微细化的分类，例如银行，酒店，体育馆。小类则是最精细的分类，例如中国银行，宜家，五棵松等。三种分类至少有一个。
    你的关键字的方向需要顺着上面的要求来改变。

    district_search_agent工具只能查询行政区划下的地点，所以任何抽象关键字都是无效的，比如：路线，安排，攻略等。一定要避免这类抽象的关键字出现在district_search_agent工具的输入参数中。
    """
    result = await district_search_agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result["messages"][-1].content


around_search_agent = create_agent(
    model=model,
    tools=[geocode, around_search],
    system_prompt=AROUND_SEARCH_SUBAGENT_SYSTEM_PROMPT,
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


path_planning_agent = create_agent(
    model=model,
    tools=[geocode, calculate_distance, walking_route, driving_route],
    system_prompt=PATH_PLANNING_SUBAGENT_SYSTEM_PROMPT,
)


@tool("path_planning_agent")
async def call_path_planning_agent(query: str):
    """
    这里利用对起点和终点进行路径规划的工具。你输入的参数需要明确是哪个地点或区域，明确省，市的单位，例如：成都天府三街到东郊记忆
    """
    result = await path_planning_agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result["messages"][-1].content


@wrap_tool_call
async def retry_geocode(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    if request.tool_call["name"] == "geocode":
        try:
            return await handler(request)
        except Exception as e:
            return ToolMessage(
                content=f"工具调用失败，错误信息是：{e}。请你一定要保证传入的地点名称是全名，不要使用简称。",
                tool_call_id=request.tool_call["id"],
            )
    return await handler(request)


travel_guide_agent = create_agent(
    model=model,
    tools=[district_search, around_search],
    system_prompt=TRAVEL_GUIDE_SUBAGENT_SYSTEM_PROMPT,
    # middleware=[retry_geocode]
)


@tool("travel_guide_agent")
async def call_travel_guide_agent(query: str):
    """
    这里利用对用户的需求不断进行检索与地点定位的然后确定旅游线路的工具。
    """
    result = await travel_guide_agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result["messages"][-1].content


main_agent = create_agent(
    system_prompt=MAIN_SYSTEM_PROMPT,
    model=model,
    tools=[  # call_district_search_agent,
        call_around_search_agent,
        call_path_planning_agent,
        call_travel_guide_agent,
    ],
)


if __name__ == "__main__":
    import asyncio

    query = "我想在喀什进行自驾游"

    async def main():
        async for step in main_agent.astream(
            {"messages": [{"role": "user", "content": query}]},
            config={"recursion_limit": 100},
        ):
            for update in step.values():
                for message in update.get("messages", []):
                    print(message.content)
                    print("-" * 100)

    asyncio.run(main())
