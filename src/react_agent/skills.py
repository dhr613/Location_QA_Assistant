import aiohttp
import os
from typing import Optional, Dict, Any,List, TypedDict
import asyncio

from langchain.tools import tool
from langchain.agents import AgentState
from typing_extensions import NotRequired, Annotated
from typing import Literal

from langchain.tools import ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage

from langchain_dev_utils.chat_models import register_model_provider
# from _profiles import _PROFILES

import os
from IPython.display import Markdown, display
from langchain_dev_utils.chat_models import load_chat_model
from langchain.messages import SystemMessage, HumanMessage,AIMessage
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()


register_model_provider(
    provider_name="dhrzhipu",
    chat_model="openai-compatible",
    base_url=os.getenv("DHRZHIPU_BASE_URL"),
    compatibility_options = {
        "support_tool_choice":["auto"],
        "support_response_format":['json_schema',]
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


from langchain.agents.middleware import ModelRequest, ModelResponse, AgentMiddleware
from typing import Callable

from common.tools import (load_skill, around_search, district_search, driving_route)
from common.prompts import SKILLS

class SkillMiddleware(AgentMiddleware):

    tools = [load_skill]

    def __init__(self):
        
        # 加载所有skill的名称和描述
        skills_list = []
        for skill in SKILLS:
            skills_list.append(
                f"- **{skill['name']}**: {skill['description']}"
            )
        self.skills_prompt = "\n".join(skills_list)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        skills_addendum = (
            f"\n\n## 可使用的 Skills\n\n{self.skills_prompt}\n\n"
            "当你需要详细了解如何处理特定类型的请求时，请使用load_skill工具。"
        )

        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]
        new_system_message = SystemMessage(content=new_content)

        if not request.state.get("skill_name"):
            tools = [load_skill]

        elif request.state.get("skill_name")=="around_search":
            tools = [around_search,district_search]

        elif request.state.get("skill_name")=="path_planning":
            tools = [driving_route,district_search]

        print("当前state保存的skill_name：",request.state.get("skill_name"))
        print("修改后的工具列表：", [t.name for t in tools])
        modified_request = request.override(
            system_message=new_system_message,
            tools=tools
        )
        return await handler(modified_request)


from langgraph.checkpoint.memory import InMemorySaver
from react_agent.state import Gaodemap_State_Skills

agent = create_agent(
    model,
    state_schema=Gaodemap_State_Skills,
    system_prompt=(
        "你是一名路径规划与地点周边检索的专家。"
        "你能够根据用户的需求，进行路径规划与地点周边检索。"
        "你需要根据需求来加载特定的skill和工具来帮你完成任务。"
        # "在你使用load_skill工具获取到详细的工具信息以后"
    ),
    middleware=[SkillMiddleware()],
    tools=[load_skill,around_search,district_search,driving_route],
    # checkpointer=InMemorySaver(),
)

if __name__ == "__main__":
    query = "郑州管城升龙广场有什么好玩的吗？从长盛广场怎么到这里呢？"
    config = {"configurable": {"thread_id": "test_1"}}

    async def main():
        async for step in agent.astream(
            {"messages": [{"role": "user", "content": query}]},
            # config=config
        ):
            for update in step.values():
                for message in update.get("messages", []):
                    message.pretty_print()
    asyncio.run(main())