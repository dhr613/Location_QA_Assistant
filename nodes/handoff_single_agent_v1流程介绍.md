## 示意图

![image-20260127163437058](./../words/imgs/image-20260127163437058.png)

## 任务目的

这是一个很直接的关于地点的问答助手。主要有两个功能，路径规划和周边搜索。这两个功能可以同时问，也可以分开问，没有前后的关系。（例如“从成都天府三街到东郊记忆怎么走，那里有什么好玩的吗”）



## 基座模型

全程使用GLM-4.5

说明：智普官方对于普通用户在4.6和4.7模型上的并发只有1，所以这里能使用的较快效果最好的模型就是4.5，并发是10。该代码是基于asyncio的异步架构，所以对并发有一定的要求。



## 实现思想

该结构是基于单agent开发的。利用handoff的思想，使得工具的使用按照指定的方式进行交接。



:pig: handoff的代码实现核心是什么呢？

:panda_face: 在state部分会有一个专门记录当前执行工具/智能体的激活的变量，通过middleware的方式，根据该变量的不同，​在每次经过网络之前，利用wrap_model_call替换掉当前的system_prompt和绑定的tools。从而实现当该变量改变的时候（即任务进行交接以后），system_prompt和tools随变量一起改变。



:pig: 这个变量是如何改变的呢？

:panda_face: 假设当前变量=A，此时匹配对应的系统提示词和工具，这里面的工具，一定有一个的作用更新state里面的该变量为B或其他。就看模型判断什么时候使用这个工具了。



:pig: 这种方式的优缺点分别是什么呢？

:panda_face: handoff的核心就是控制权的交接。在单agent中就是工具的交接。它最大的特点有两点：:one: ​是能够保持当前的交接状态，在遇到相同类型任务的时候，不进行任务的交接直接执行该任务。:two: ​它利用了更换提示词和工具的方式，使一个agent发挥出了多agent的特点，在代码层面上肯定是更简洁的，但是逻辑上就不一定了（通常会更复杂一点）它的缺点是并不能完成实现上下文的隔离。根据上一个问题，我们只会改变系统提示词和绑定的工具，中间的状态（尤其是messages）是共享的。



### 架构简介

该项目一共会绑定6个tools，一个wrap_model_call。由于本项目只有路径规划和周边检索两个子agent，所以只会在这两个任务进行切换。



`geocode`

- input：

  - address：str，require，地点1|地点2|...|地点N

  - city：str，norequire，成都

  - batch：bool，norequire，是否进行批量转化（多个地点的时候会用到）

  - runtime：截断模型调用，用来获取模型参数

- output：

  - ```python
    return Command(
        update={
            "messages": [
                ToolMessage(content=f"{res}", 									tool_call_id=runtime.tool_call_id)
            ],
            "current_step": query_type,
        }
    )
    ```
  
  - query_type就是那个变量，根据state来决定。表示此时的geocode是为路径规划还是周边搜索执行的（因为这两个任务都需要geocode来获取经纬度）
  
  - res是请求后解析的字典，表示请求地点的经纬度。由于该工具的输出是用Command传递的，所以必须规定tool_id



`around_search`

- input：

  - location：str，require，"116.473168,39.993015"（只支持一个点的经纬度检索）

  - keywords：str，norequire，酒吧|KTV|洗浴中心（支持多个关键字）

  - city：str，norequire，成都

- output：
  - 一个经过解析的字符串，这部分输出会被自动转化为字符串



`driving_route`

- input：

  - origin：str，require，"116.397428,39.90923"

  - destination：str，require，格式同上

  - origin_id：str，norequire，"xxxxxx"

  - destination_id，str，norequire，"xxxxxx"
  - runtime

- output：

  - ```python
    return Command(
        update={
            "messages": [
                ToolMessage(content=f"{path}", 									tool_call_id=runtime.tool_call_id)
            ],
            "current_step": "around_search_step",
        }
    )
    ```

  - 这里current_step表示运行这个工具以后下一个任务指向周边检索任务。该参数这里是可以不传递的，我这里的设计思路是，默认路径规划下会查询目的地的周边检索。（单纯是演示不利用额外的工具进行跳转）



`back_to_geocode` ，`back_to_around_search` ，`back_to_driving_route`

```python
return Command(update={"current_step": "driving_route_step"})
```

三个back函数，都只负责修改current_step变量的值，转化时机由提示词和模型来控制。



### 改进方向

:one: ​只有提示词和绑定的工具进行了隔离，中间的状态都是共享的，尤其是多余的AM和TM肯定会污染上下文。

:two: 该思路对系统提示词的要求较大，需要保证每个子agent的提示词都足够明确清晰，才能自由完成任务的交接。​

