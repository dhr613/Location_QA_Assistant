## 示意图

![image-20260126170816359](./../words/imgs/image-20260126170816359.png)

subagents_v2外部架构依然是一个标准的react结构



## 主要改进点

在v1中本人使用了深度思考的工具，其实这部分工具是可以替换到系统提示词里面的。v2主要的改进点就是这里。这样会显著提高最后文档生成的速度。

但是内容的长度会有明显的减少。

通过我预留的测试文档，可以很容易看到生成的过程。在v1中，会出现非常有逻辑的思考过程，并且读取工具的次数也明显更多（思考工具本身也并不特别耗时，主要的耗时还是增加了工具的调用）



所以V1和V2我个人认为都有很好的利用价值



:bangbang: 本文档与v1文档存在较大篇幅的重复内容，如之前已读取过v1文档，可只看**改进点总结**部分​



## 任务目的

该多agent架构的目的是能通过用户简单的提示词，写一篇非常详细的旅游攻略，包括具体的住宿，游玩项目，具体路线等，一应俱全。文本内容非常丰富。



## 基座模型

全程使用GLM-4.5

说明：智普官方对于普通用户在4.6和4.7模型上的并发只有1，所以这里能使用的较快效果最好的模型就是4.5，并发是10。该代码是基于asyncio的异步架构，所以对并发有一定的要求。

整个文档生成耗时普遍在3min。文本内容在1000字左右。



## 实现思想

subagents的思想就是将子agent包装成工具，从而实现上下文的隔离。子agent中的state在不进行人为操作的情况下是不进行保存的。也就是对于主agent而言，它只传递输入到子agent，然后接受子agent的输出，中间过程是不参与的。



### 架构简介

在该版本下有3个子agent

- call_around_search_agent：负责根据一个固定地点，搜索周边符合需求的其他地点（例如”成都天府三街附近的火锅店”，成都天府三街是一个地点，火锅店是关键字。关键字也可以是描述性关键字，例如好吃的，好玩的等等）
- call_path_planning_agent：负责对两个地点进行路径规划，如果这两个地点离得近，那么会推荐使用步行规划，如果离得远则优先使用驾车规划。
- call_travel_guide_agent：负责对整个旅游项目撰写一个大纲，最后的旅游规划就是基于这个大纲进行扩展的。



#### call_around_search_agent

该子智能体绑定了两个工具，geocode和around_search。前者负责将自然语言的地点转化为具体的经纬度坐标（支持多个地点同时转化），后者利用经纬度坐标去检索该地点周边的场所。

`geocode`

- input：

  - address：str，require，地点1|地点2|...|地点N

  - city：str，norequire，成都

  - batch：bool，norequire，是否进行批量转化（多个地点的时候会用到）

- output：

  - 一个经过解析过滤的字典，这部分输出会被自动转化为字符串



`around_search`

- input：
  - location：str，require，"116.473168,39.993015"（只支持一个点的经纬度检索）
  - keywords：str，norequire，酒吧|KTV|洗浴中心（支持多个关键字）
  - city：str，norequire，成都
- output：
  - 一个经过解析的字符串，这部分输出会被自动转化为字符串



#### call_path_planning_agent

该子智能体绑定了四个工具

`geocode`：同上

`calculate`：输入起终点的经纬度坐标，返回两地的距离，单位：米

- input：
  - origins：List[List[float]] ，require，[[116.481028, 39.989643], [114.481028, 39.989643]]
  - destination：List[List[float]] ，require，与origins不同的是，它只能输入一组经纬度，因为它的内部并不是矩阵乘法的逻辑。
- output：
  - "10000"

`walking_route`：输入起终点的经纬度坐标，以及各自的POI（如果有的话），返回一个解析的路径字典（POI信息是高德为每一个地点定制的唯一的编号，geocode并不能获取到这一信息，也没有专门获取POI的接口，但是该信息可能会出现在其他接口的返回值里面，所以这里选择让模型动态填入会更合适）

- input：

  - origin：str，require，"116.397428,39.90923"

  - destination：str，require，格式同上

  - origin_id：str，norequire，"xxxxxx"

  - destination_id，str，norequire，"xxxxxx"

- output：
  - 解析过滤后的一个字典

`driving_route` 同上



#### call_travel_guide_agent

该子智能体会绘制一个旅游攻略的大纲，绑定了3个工具

`district_search` 它可以返回指定行政区内指定类别下的场所（该API会返回场所的POI信息）

- input：
  - keywords：str，require，串串香|洗浴中心
  - city：str，norequire，成都武侯区
- output：
  - 一个经过解析的字典

`around_search` 与call_around_search_agent相同的工具



### 改进点总结

:one: 删除了v1中所有的思考工具

:two: 将思考的部分全部总结到了对应的系统提示词中

:three: 丰富了district_search的工具描述

