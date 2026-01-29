## 示意图

![image-20260126142837238](./../words/imgs/image-20260126142837238.png)

## 基本思想

router实现的思想很直接。首先有一个分类器classify，它负责将用户的提问进行拆解成多个子query，然后不同的query根据分类前往不同的子agent。最后利用synthesize将每个子agent返回的问题进行总结后最后反馈给用户。



## 代码层面的注意事项

router的思想在langchain，langgraph 0.x的时候就已经很成熟了。所以官方给的代码是基于langgraph来实现的。



每一个节点就是一个函数，至于你函数里面怎么实现，本身和框架无关：

:one: ​classify：内部是一个结构化输出的实现。可以采用with_structure_output，也可以用create_agent的response_format的形式实现。最后将分类的结果保存到state中的classification属性中。

:two: around_search_agent：内部是一个create_agent的实例，它能够对用户的周边需求进行解析，并返回周边的信息

:three: path_planning_agent：内部是一个create_agent的实例，它能够对用户传入的始终点返回路径规划

:four: synthesize：内部是就一个IO，从state中读取前面的信心进行总结并保存回state中​

---

## 细节/改进

### classify本身拆解的子query并不固定为每一个类别只能有一个

classify是一个负责结构化输出的路由，它会将用户的问题拆解成N（N为子agent的个数）个子query，然后每个query会对应一个子agent

:panda_face: 严格来说可以实现N+个子query。比如用户提问：我想知道成都天府三街到东郊记忆怎么走？这两地有什么好吃的吗？那么子query既可以拆解成：

- 成都天府三街到东郊记忆怎么走
- 成都天府三街有什么好吃的
- 成都东郊记忆有什么好吃的

第1个子query经过path_planning_agent，第2,3个经过around_search_agent。

router的好处就是所有的子agent都是并行操作的，就算你只有一个子agent，也能够实现并行。这是因为 [Send](https://reference.langchain.com/python/langgraph/types/?h=send#langgraph.types.Send) 关键字本身就可以支持并行。



### 上下文隔离的思考

本项目是基于官方的代码改写的。本身官方的代码就有一个问题：

state是如下定义的：

```python
class Gaodemap_State_Router(AgentState):
    query: str
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], operator.add]
    final_answer: str
```

在不进行记忆储存的时候当然不用担心上下文隔离的问题。但是如果添加了长期记忆，那么新的query会覆盖旧的query，但是results却可以保留旧的结果，这是很矛盾的。我只是在这里提出来，具体的隔离可以根据项目自己发挥。