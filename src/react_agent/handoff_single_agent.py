import os
import asyncio
from dotenv import load_dotenv
from typing import Callable

from langchain_dev_utils.chat_models import register_model_provider,load_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

from langgraph.checkpoint.memory import InMemorySaver

from react_agent.state import Gaodemap_State_Handoff_Single
from common.tools import geocode_handoff_single, driving_route_handoff_single, around_search, back_to_geocode, back_to_around_search, back_to_driving_route
from common.prompts import GEOCODE_HANDOFF_SINGLE_SYSTEM_PROMPT, DRIVING_ROUTE_HANDOFF_SINGLE_SYSTEM_PROMPT, AROUND_SEARCH_HANDOFF_SINGLE_SYSTEM_PROMPT

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
    model="dhrzhipu:glm-4.7",
    # model = "dhrark:doubao-1-5-pro-32k-250115",
    extra_body={
        "thinking": {"type": "disabled"},
    }
)

STEP_CONFIG = {
    "geocode_step": {
        "prompt": GEOCODE_HANDOFF_SINGLE_SYSTEM_PROMPT,
        "tools": [geocode_handoff_single],
        "requires": [],
    },
    "driving_route_step": {
        "prompt": DRIVING_ROUTE_HANDOFF_SINGLE_SYSTEM_PROMPT,
        "tools": [driving_route_handoff_single,back_to_geocode,back_to_around_search],
        "requires": ["geocode"],
    },
    "around_search_step": {
        "prompt": AROUND_SEARCH_HANDOFF_SINGLE_SYSTEM_PROMPT,
        "tools": [around_search,back_to_geocode,back_to_driving_route],
        "requires": ["geocode"],
    },
}


@wrap_model_call
async def apply_step_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:

    current_step = request.state.get("current_step", "geocode_step")
    print(f"当前阶段为：{current_step}")

    stage_config = STEP_CONFIG[current_step]  

    # for key in stage_config["requires"]:
    #     if request.state.get(key) is None:
    #         raise ValueError(f"{key} must be set before reaching {current_step}")

    system_prompt = stage_config["prompt"].format(**request.state)
    print(f"系统提示词为：{system_prompt[0:100]}")
    print("当前可调用工具为：",[tool.name for tool in stage_config["tools"]])
    request = request.override(  
        system_prompt=system_prompt,  
        tools=stage_config["tools"],  
    )

    return await handler(request)


all_tools = [
    geocode_handoff_single,
    driving_route_handoff_single,
    around_search,
    back_to_geocode,
    back_to_around_search,
    back_to_driving_route,
]

agent = create_agent(
    model,
    tools=all_tools,
    state_schema=Gaodemap_State_Handoff_Single,  
    middleware=[apply_step_config],  
    # checkpointer=InMemorySaver(),  
)


if __name__ == "__main__":
    query = "成都二仙桥附近有什么好吃的吗？"
    config = {"configurable": {"thread_id": "test_4"}}

    async def main():
        async for step in agent.astream(
            {"messages": [{"role": "user", "content": query}]},
                # config=config
            ):
                for update in step.values():
                    for message in update.get("messages", []):
                        print(message.content)
                        print("-"*100)
    asyncio.run(main())