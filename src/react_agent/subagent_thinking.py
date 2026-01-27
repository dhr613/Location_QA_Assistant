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
    AROUND_SEARCH_SUBAGENT_THINKING_SYSTEM_PROMPT,
    DISTRICT_SEARCH_SUBAGENT_THINKING_SYSTEM_PROMPT,
    MAIN_SUBAGENT_THINKING_SYSTEM_PROMPT,
    PATH_PLANNING_SUBAGENT_THINKING_SYSTEM_PROMPT,
    TRAVEL_GUIDE_SUBAGENT_THINKING_SYSTEM_PROMPT,
)
from common.tools import (
    around_search,
    calculate_distance,
    district_deepthink,
    district_search,
    driving_route,
    geocode,
    guiding_deepthink,
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
    tools=[district_search, district_deepthink],
    system_prompt=DISTRICT_SEARCH_SUBAGENT_THINKING_SYSTEM_PROMPT,
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
    system_prompt=AROUND_SEARCH_SUBAGENT_THINKING_SYSTEM_PROMPT,
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
    system_prompt=PATH_PLANNING_SUBAGENT_THINKING_SYSTEM_PROMPT,
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
    tools=[guiding_deepthink, district_search, around_search],
    system_prompt=TRAVEL_GUIDE_SUBAGENT_THINKING_SYSTEM_PROMPT,
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


@tool
async def main_think(thinking: str):
    """
    你的任务是对当前的旅游信息进行深度思考，以合理地实现下一步的规划。
    你能使用的工具只有3个，分别是：
    call_travel_guide_agent：负责对根据用户的需求生成旅游大纲，在整个规划过程只会在最开始使用一次，在后面的过程中你不需要再调用。
    call_around_search_agent：负责对旅游大纲中的地点进行周边区域地点检索，例如，成都东郊记忆附近的宾馆，就会返回给你对应范围内的一定数量的宾馆信息
    call_path_planning_agent：负责对旅游大纲中的地点进行路径规划，例如，从成都东郊记忆到二仙桥，就会返回给你从东郊记忆到二仙桥的路线规划信息

    你的思考方向是：
    1：大纲中出现的地区附近是否还有其他值得游玩的地点，如果有，请使用call_around_search_agent工具进行检索。如果call_around_search_agent工具返回的地点不满足用户的需求，你有以下几个需要修改的方向：
        1: 你指定的地点不精确，例如“海底捞”但是没有明确是哪个城市，哪个区。从而导致检索得到的结果不符合需求
        2：你检索的关键词是抽象的，例如路线，安排，攻略等，这会导致检索得到的结果不符合需求
    2：call_path_planning_agent工具的输入参数是起点和终点，你需要合理地选择起点和终点，以确保路径规划的合理性。你的输入地点必须足够的精确，不能使用抽象的关键字，例如路线，安排，攻略等。不能使用地点的缩写和俗称。

    如果你认为目前的信息已充足，可以总结路线反馈给用户以后，整合所有信息，返回给用户一个具体的旅行计划。
    """
    return thinking


# Main agent with subagent as a tool
main_agent = create_agent(
    system_prompt=MAIN_SUBAGENT_THINKING_SYSTEM_PROMPT,
    model=model,
    tools=[  # call_district_search_agent,
        call_around_search_agent,
        call_path_planning_agent,
        call_travel_guide_agent,
        main_think,
    ],
)

if __name__ == "__main__":
    import asyncio

    query = "我想在成都进行自驾游"

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
