"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence,Literal,NotRequired,Callable
from typing_extensions import Annotated
import operator
from pydantic import BaseModel,Field

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langchain.agents import AgentState
from langgraph.managed import IsLastStep
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

from common.basemodel import Classification,AgentOutput




@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    Messages tracking the primary execution state of the agent.

    Typically accumulates a pattern of:
    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    4. AIMessage without .tool_calls - agent responding in unstructured format to the user
    5. HumanMessage - user responds with the next conversational turn

    Steps 2-5 may repeat as needed.

    The `add_messages` annotation ensures that new messages are merged with existing ones,
    updating by ID to maintain an "append-only" state unless a message with the same ID is provided.
    """


@dataclass
class State(InputState):
    """Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    """

    is_last_step: IsLastStep = field(default=False)
    """
    Indicates whether the current step is the last one before the graph raises an error.

    This is a 'managed' variable, controlled by the state machine rather than user code.
    It is set to 'True' when the step count reaches recursion_limit - 1.
    """

    # Additional attributes can be added here as needed.
    # Common examples include:
    # retrieved_documents: List[Document] = field(default_factory=list)
    # extracted_entities: Dict[str, Any] = field(default_factory=dict)
    # api_connections: Dict[str, Any] = field(default_factory=dict)


###################### ROUTER ######################################

class Gaodemap_State_Router(AgentState):
    query: str
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], operator.add]
    final_answer: str

###################### HANDOFF SINGLE AGENT ########################

class Gaodemap_State_Handoff_Single(AgentState):
    current_step: Literal["around_search_step", "geocode_step", "driving_route_step"]
    current_position: NotRequired[str]


##################### HANDOFF SINGLE AGENT V2 ######################

class Gaodemap_State_Handoff_Single_V2(AgentState):
    current_step: Literal["around_search_agent_step", "main_step", "path_planning_agent_step"]
    current_position: NotRequired[str]


##################### HANDOFF MULTI AGENTS ######################

class Gaodemap_State_Handoff_Multi(AgentState):
    current_step: Literal["call_around_node", "call_path_node"]
    current_position: NotRequired[str]





