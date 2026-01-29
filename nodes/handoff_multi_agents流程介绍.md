## 示意图

![image-20260127234515154](./../words/imgs/image-20260127234515154.png)

这是利用langgraph将子智能体作为节点来构建多智能体



## 任务目的

实现路径规划和周边搜索



## 基座模型

全程使用GLM-4.7



## 实现思想

两个节点之间可以实现相互跳转，关键还是路由函数。路由函数依然由状态里面的变量（current_step）控制。current_step依然由绑定的agent调用工具来改变：

```python
@tool
async def jump_to_path_planning_agent(
    runtime: ToolRuntime[Gaodemap_State]
) -> Command:
    """
    将当前阶段跳转到path_planning_agent
    """
    last_ai_message = next(  
        msg for msg in reversed(runtime.state["messages"]) if isinstance(msg, AIMessage)  
    )
    transfer_message = ToolMessage(  
        content="将当前阶段跳转到path_planning_agent_step",  
        tool_call_id=runtime.tool_call_id,  
    )  
    return Command(
        update={
            "goto": "call_path_node",
            "messages": [
                last_ai_message,
                transfer_message
            ],
            "current_step": "call_path_node"
        },
        graph=Command.PARENT
    )
```

这里必须使用Command来传递状态，因为此时的节点是一个子agent，如果不使用Command里面的graph参数指定改变的是父图的变量，那么父图的current_step是无法被改变的，也就无法实现跳转。





