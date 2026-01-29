## 示意图

![image-20260127224021919](./../words/imgs/image-20260127224021919.png)

## 任务目的

以另一种思路实现handoff，同样针对的都是路径规划和周边搜索这两类任务



## 基座模型

全程使用GLM-4.5

说明：智普官方对于普通用户在4.6和4.7模型上的并发只有1，所以这里能使用的较快效果最好的模型就是4.5，并发是10。该代码是基于asyncio的异步架构，所以对并发有一定的要求。



## 改进思路

在v1中，三个实操工具和三个跳转工具是放在一起绑定的，现在可以利用subagents的思路，将实操工具全部写成子智能体里面，这样就可以将两类工具分开，逻辑性更强的同时，稳定性也会提升。



## 实现思想

将district_search, driving_route包装起来作为路径规划的子智能体

将district_search, around_search包装起来作为周边搜索的子智能体

搭配两个交接任务的工具jump_to_around_search_agent和jump_to_path_planning_agent来实现跳转。



### 架构简介

#### call_around_search_agent

由于不需要子智能体完成交接，所以绑定的工具只需要完成最直接的IO就可以了

`around_search`

- input：
  - location：str，require，"116.473168,39.993015"（只支持一个点的经纬度检索）
  - keywords：str，norequire，酒吧|KTV|洗浴中心（支持多个关键字）
  - city：str，norequire，成都
- output：
  - 一个经过解析的字符串，这部分输出会被自动转化为字符串



`district_search` 它可以返回指定行政区内指定类别下的场所（该API会返回场所的POI信息）

- input：
  - keywords：str，require，串串香|洗浴中心
  - city：str，norequire，成都武侯区
- output：
  - 一个经过解析的字典



#### call_path_planning_agent

`district_search`  同上

`driving_route`  

- input：

  - origin：str，require，"116.397428,39.90923"

  - destination：str，require，格式同上

  - origin_id：str，norequire，"xxxxxx"

  - destination_id，str，norequire，"xxxxxx"

- output：
  - 解析过滤后的一个字典



#### 两个jump

```python
    return Command(
        update={
            "messages": [
                ToolMessage(content=f"将当前阶段跳转到path_planning_agent_step", tool_call_id=runtime.tool_call_id)
            ],
            "current_step": "path_planning_agent_step"
        }
    )
```

只负责任务交接



## 注意事项

这里面一共使用了5个提示词

- 刚开始的main_system_prompt
- 用来在wrap_model_call替换的around_search_prompt和path_planning_prompt
- 在子agent内部还各自有一个around_search和path_planning的提示词