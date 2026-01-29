"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Literal, Optional, cast

import aiohttp
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_tavily import TavilySearch
from langgraph.runtime import get_runtime
from langgraph.types import Command

from common.context import Context
from common.mcp import get_deepwiki_tools
from common.utils import (
    gaode_parse_geocode,
    gaode_parse_key_words_and_around_search,
    gaode_parse_poi_search,
    gaode_parse_polygon_search,
)
from common.prompts import SKILLS

from react_agent.state import (
    Gaodemap_State_Handoff_Multi,
    Gaodemap_State_Handoff_Single,
    Gaodemap_State_Handoff_Single_V2,
)


logger = logging.getLogger(__name__)


async def web_search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    runtime = get_runtime(Context)
    wrapped = TavilySearch(max_results=runtime.context.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


# async def get_tools() -> List[Callable[..., Any]]:
#     """Get all available tools based on configuration."""
#     tools = [web_search]

#     runtime = get_runtime(Context)

#     if runtime.context.enable_deepwiki:
#         deepwiki_tools = await get_deepwiki_tools()
#         tools.extend(deepwiki_tools)
#         logger.info(f"Loaded {len(deepwiki_tools)} deepwiki tools")

#     return tools


@tool
async def district_search(keywords: str, city: Optional[str] = None) -> Dict[str, Any]:
    """
    根据固定行政区域进行检索的工具
    参数:
        keywords: 查询关键字，多个关键字用"|"分割，文本总长度不可超过80字符（必需）
        city: 查询城市，可选值：城市中文、中文全拼、citycode、adcode
              如：北京/beijing/010/110000（可选）

    返回:
        API响应的JSON数据字典
    """
    print("开始执行district_search工具，参数：", keywords, city)
    # 从环境变量获取API Key
    key = os.getenv("GAODE_MAP_KEY")
    if not key:
        raise EnvironmentError("未配置高德地图API Key，请设置环境变量GAODE_MAP_KEY")

    types = ""  # 查询POI类型，空字符串表示不限制，多个类型用"|"分割
    city_limit = True  # 默认严格限制在指定区域内
    page_size = 10  # 默认每页记录数据10（范围1-25）
    page_num = 1  # 默认当前页数1（范围1-100）
    show_fields = (
        "base,business,children,indoor,navi,photos"  # 返回字段控制，多个字段用","分割
    )
    output = "json"  # 返回格式，默认json（目前只支持json）

    params: Dict[str, Any] = {
        "keywords": keywords,
        "key": key,
        "page_size": page_size,  # 每一页的数量
        "page_num": page_num,  # 一共多少页
        "city_limit": str(city_limit).lower(),  # 转换为字符串true/false
        "show_fields": show_fields,
        "output": output,
    }

    # 可选参数：只在有值时才添加
    if types:
        params["types"] = types

    if city is not None:
        params["region"] = city

    url = "https://restapi.amap.com/v5/place/text"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            res = await response.json()
            # return res
            return await gaode_parse_key_words_and_around_search(res)


@tool
async def around_search(
    location: str, keywords: Optional[str] = None, city: Optional[str] = None
) -> Dict[str, Any]:
    """
    针对固定地区周边地区进行检索的工具

    参数:
        location: 中心点坐标，格式：经度,纬度（必需）
                  例如："116.473168,39.993015"
                  圆形区域检索中心点，不支持多个点，经纬度小数点后不得超过6位
        keywords: 查询关键字，多个关键字用"|"分割，文本总长度不可超过80字符（可选）
        city: 查询城市，可选值：城市中文、中文全拼、citycode、adcode
              如：北京/beijing/010/110000（可选）

    返回:
        API响应的JSON数据字典，包含距离信息
    """
    print("开始执行around_search工具，参数：", location, keywords, city)
    # 从环境变量获取API Key
    key = os.getenv("GAODE_MAP_KEY")
    if not key:
        raise EnvironmentError("未配置高德地图API Key，请设置环境变量GAODE_MAP_KEY")

    # 函数内部指定的默认参数（2.0版本）
    types = ""  # 查询POI类型，空字符串表示不限制，多个类型用"|"分割
    radius = 3000  # 查询半径，单位：米，默认3000米（范围0-50000米）
    sortrule = "distance"  # 排序规则，distance按距离排序，weight综合排序
    city_limit = True  # 默认严格限制在指定区域内
    page_size = 10  # 默认每页记录数据10（范围1-25）
    page_num = 1  # 默认当前页数1（范围1-100）
    show_fields = (
        "base,business,children,indoor,navi,photos"  # 返回字段控制，多个字段用","分割
    )
    output = "json"  # 返回格式，默认json（目前只支持json）

    # 构建参数字典（使用2.0版本参数名）
    params: Dict[str, Any] = {
        "location": location,
        "key": key,
        "radius": min(radius, 50000),  # 限制不超过50000米
        "sortrule": sortrule,
        "page_size": min(page_size, 2),  # 限制不超过25
        "page_num": min(page_num, 1),  # 限制不超过100
        "city_limit": str(city_limit).lower(),  # 转换为字符串true/false
        "show_fields": show_fields,
        "output": output,
    }

    # 可选参数：只在有值时才添加
    if keywords is not None:
        params["keywords"] = keywords

    if types:
        params["types"] = types

    if city is not None:
        params["region"] = city  # 2.0版本使用region替代city

    url = "https://restapi.amap.com/v5/place/around"  # 2.0版本使用v5

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            res = await response.json()
            return await gaode_parse_key_words_and_around_search(res)


@tool
async def polygon_search(
    polygon: str, keywords: Optional[str] = None, city: Optional[str] = None
) -> Dict[str, Any]:
    """
    根据多边形检索的工具

    参数:
        polygon: 多边形边界坐标，格式：经度1,纬度1|经度2,纬度2|...|经度n,纬度n（必需）
                  要求：1. 至少3个顶点，首尾坐标建议闭合（不闭合API会自动补全）
                        2. 经纬度小数点后不得超过6位
                        3. 顶点数量不超过100个，多边形面积不超过全国范围
        keywords: 查询关键字，多个关键字用"|"分割，文本总长度不可超过80字符（可选）
        city: 查询城市，可选值：城市中文、中文全拼、citycode、adcode
              如：北京/beijing/010/110000（可选）
    """
    print("开始执行polygon_search工具，参数：", polygon, keywords, city)
    # 从环境变量获取API Key
    key = os.getenv("GAODE_MAP_KEY")
    if not key:
        raise EnvironmentError("未配置高德地图API Key，请设置环境变量GAODE_MAP_KEY")

    # 函数内部指定的默认参数（2.0版本）
    types = ""  # 查询POI类型，空字符串表示不限制，多个类型用"|"分割
    city_limit = True  # 默认严格限制在指定区域内
    page_size = 2  # 默认每页记录数据10（范围1-25）
    page_num = 1  # 默认当前页数1（范围1-100）
    show_fields = (
        "base,business,children,indoor,navi,photos"  # 返回字段控制，多个字段用","分割
    )
    output = "json"  # 返回格式，默认json（目前只支持json）

    # 构建参数字典（使用2.0版本参数名，遵循原有代码格式）
    params: Dict[str, Any] = {
        "polygon": polygon,
        "key": key,
        "page_size": min(page_size, 25),  # 限制不超过API上限25条/页
        "page_num": min(page_num, 100),  # 限制不超过API上限100页
        "city_limit": str(
            city_limit
        ).lower(),  # 转换为小写字符串true/false，符合API要求
        "show_fields": show_fields,
        "output": output,
    }

    # 可选参数：只在有值时才添加（保持与原有两个函数一致的逻辑）
    if keywords is not None:
        params["keywords"] = keywords

    if types:
        params["types"] = types

    if city is not None:
        params["region"] = city  # 2.0版本统一使用region替代city，保持接口一致性

    # 多面检索API端点（2.0版本v5）
    url = "https://restapi.amap.com/v5/place/polygon"

    # 发送异步GET请求，遵循原有代码的会话管理和错误处理逻辑
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()  # 抛出HTTP状态码错误（4xx/5xx）
            res = await response.json()
            # return res
            return await gaode_parse_polygon_search(res)  # 复用统一的解析函数


@tool
async def id_search(id: str, show_fields: Optional[str] = None) -> Dict[str, Any]:
    """
    作用：针对具体的POI进行精细化检索的函数
    参数:
        id: POI唯一标识ID（必需）
        show_fields: 返回字段控制，多个字段用","分割，可选值：base,business,children,indoor,navi,photos
                     若不指定，使用默认完整字段集（可选）
    """
    print("开始执行id_search工具对具体地点进行精细化检索，参数：", id, show_fields)
    if not id or not id.strip():
        raise ValueError("POI ID不能为空，请传入有效的POI唯一标识ID")

    key = os.getenv("GAODE_MAP_KEY")
    if not key:
        raise EnvironmentError("未配置高德地图API Key，请设置环境变量GAODE_MAP_KEY")

    default_show_fields = (
        "base,business,children,indoor,navi,photos"  # 默认返回完整字段集
    )
    output = "json"  # 返回格式，默认json（目前只支持json）

    params: Dict[str, Any] = {
        "id": id.strip(),
        "key": key,
        "output": output,
        "show_fields": show_fields if show_fields is not None else default_show_fields,
    }

    url = "https://restapi.amap.com/v5/place/detail"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()  # 抛出HTTP状态码错误（4xx/5xx）
            res = await response.json()  # 解析JSON响应
            # return res
            return await gaode_parse_poi_search(res)  # 复用统一的解析函数


@tool
async def geocode(
    address: str, city: Optional[str] = None, batch: Optional[bool] = False
) -> Dict[str, Any]:
    """
    将结构化地址转换为高德经纬度坐标的工具（地理编码）

    参数:
        address: 结构化地址信息（必需）
                 规则遵循：国家、省份、城市、区县、城镇、乡村、街道、门牌号码、屋邨、店铺名称、大厦
                 例如：北京市朝阳区阜通东大街6号
                 如果需要解析多个地址，请用"|"进行间隔，并且将batch参数设置为true，最多支持10个地址
        city: 指定查询的城市（可选）
              可选输入内容包括：指定城市的中文（如北京）、指定城市的中文全拼（beijing）、
              citycode（010）、adcode（110000），不支持县级市
        batch: 是否批量解析地址，默认为false。当address包含多个地址（用"|"分隔）时，应设置为true
    """
    print(
        "开始执行geocode工具，将具体的地点转化为经纬度坐标，参数：",
        address,
        city,
        batch,
    )
    # 从环境变量获取API Key
    key = os.getenv("GAODE_MAP_KEY")
    if not key:
        raise EnvironmentError("未配置高德地图API Key，请设置环境变量GAODE_MAP_KEY")

    # 函数内部指定的默认参数
    output = "json"  # 返回数据格式类型，默认json

    # 构建参数字典
    params: Dict[str, Any] = {
        "address": address,
        "key": key,
        "output": output,
        "batch": "true" if batch else "false",
    }

    # 可选参数：只在有值时才添加
    if city is not None:
        params["city"] = city

    url = "https://restapi.amap.com/v3/geocode/geo"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()  # 抛出HTTP状态码错误（4xx/5xx）
            res = await response.json()  # 解析JSON响应
            # return res
            print(res)
            return await gaode_parse_geocode(res)


@tool
async def geocode_handoff_single(
    address: str,
    query_type: Literal["driving_route_step", "around_search_step"],
    runtime: ToolRuntime[Gaodemap_State_Handoff_Single],
    city: Optional[str] = None,
    batch: Optional[bool] = False,
) -> Dict[str, Any]:
    """
    将单个或多个地址转换为高德经纬度坐标的工具（地理编码）

    参数:
        address: 结构化地址信息（必需）
                 规则遵循：国家、省份、城市、区县、城镇、乡村、街道、门牌号码、屋邨、店铺名称、大厦。你必须使用地点的全名，不要使用简称
                 例如：北京市朝阳区阜通东大街6号
                 如果需要解析多个地址，请用"|"进行间隔，并且将batch参数设置为true，最多支持10个地址
        city: 指定查询的城市（可选）
              可选输入内容包括：指定城市的中文（如北京）、指定城市的中文全拼（beijing）、
              citycode（010）、adcode（110000），不支持县级市
        batch: 是否批量解析地址，默认为false。当address包含多个地址（用"|"分隔）时，应设置为true
    """
    print(
        "开始执行geocode工具，将具体的地点转化为经纬度坐标，参数：",
        address,
        city,
        batch,
    )
    # 从环境变量获取API Key
    key = os.getenv("GAODE_MAP_KEY")
    if not key:
        raise EnvironmentError("未配置高德地图API Key，请设置环境变量GAODE_MAP_KEY")

    # 函数内部指定的默认参数
    output = "json"  # 返回数据格式类型，默认json

    # 构建参数字典
    params: Dict[str, Any] = {
        "address": address,
        "key": key,
        "output": output,
        "batch": "true" if batch else "false",
    }

    # 可选参数：只在有值时才添加
    if city is not None:
        params["city"] = city

    url = "https://restapi.amap.com/v3/geocode/geo"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()  # 抛出HTTP状态码错误（4xx/5xx）
            res = await response.json()  # 解析JSON响应

            res = await gaode_parse_geocode(res)
            return Command(
                update={
                    "messages": [
                        ToolMessage(content=f"{res}", tool_call_id=runtime.tool_call_id)
                    ],
                    "current_step": query_type,
                }
            )


@tool
async def regeocode(location: str, poitype: Optional[str] = None) -> Dict[str, Any]:
    """
    将经纬度坐标转换为详细结构化地址的工具（逆地理编码）

    参数:
        location: 经纬度坐标（必需）
                 格式：经度,纬度（经度在前，纬度在后，经纬度小数点后不超过6位）
                 例如：116.481488,39.990464
                 如果需要解析多个坐标，请用"|"进行间隔，最多支持20个坐标点
        poitype: POI类型过滤（可选）
                 支持传入POI TYPECODE及名称，多个POI类型用"|"分隔
                 当设置此参数时，会自动返回附近POI信息
    """
    print(
        "开始执行regeocode工具，将经纬度坐标转化为结构化地址，参数：", location, poitype
    )
    # 从环境变量获取API Key
    key = os.getenv("GAODE_MAP_KEY")
    if not key:
        raise EnvironmentError("未配置高德地图API Key，请设置环境变量GAODE_MAP_KEY")

    # 函数内部指定的默认参数
    output = "json"  # 返回数据格式类型，默认json
    radius = 1000  # 搜索半径，单位：米，取值范围0~3000，默认1000
    extensions = (
        "all" if poitype is not None else "base"
    )  # 如果有poitype，则返回all，否则返回base
    roadlevel = 1  # 道路等级，0显示所有道路，1仅输出主干道路
    homeorcorp = 0  # POI排序优化，0不干扰，1居家优先，2公司优先

    # 自动检测是否为批量查询（location中包含"|"分隔符）
    batch = "|" in location

    # 构建参数字典
    params: Dict[str, Any] = {
        "location": location,
        "key": key,
        "output": output,
        "radius": radius,
        "extensions": extensions,
        "batch": "true" if batch else "false",
        "roadlevel": roadlevel,
        "homeorcorp": homeorcorp,
    }

    # 可选参数：只在有值时才添加
    if poitype is not None:
        params["poitype"] = poitype

    url = "https://restapi.amap.com/v3/geocode/regeo"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()  # 抛出HTTP状态码错误（4xx/5xx）
            res = await response.json()  # 解析JSON响应
            return res


@tool
async def weather_query(city: str, extensions: Optional[str] = None) -> Dict[str, Any]:
    """
    查询指定城市的天气信息

    参数:
        city: 城市编码，输入城市的adcode，adcode信息可参考城市编码表（必需）
              例如："110101"（北京东城区）
        extensions: 气象类型（可选）
                   可选值：base（返回实况天气）、all（返回预报天气）
                   默认值：base
    """
    print("开始执行weather_query工具，查询指定城市的天气信息，参数：", city, extensions)
    # 从环境变量获取API Key
    key = os.getenv("GAODE_MAP_KEY")
    if not key:
        raise EnvironmentError("未配置高德地图API Key，请设置环境变量GAODE_MAP_KEY")

    # 函数内部指定的默认参数
    output = "json"  # 返回格式，默认json

    # 构建参数字典
    params: Dict[str, Any] = {
        "city": city,
        "key": key,
        "output": output,
    }

    # 可选参数：只在有值时才添加
    if extensions is not None:
        params["extensions"] = extensions
    else:
        params["extensions"] = "base"  # 默认返回实况天气

    url = "https://restapi.amap.com/v3/weather/weatherInfo"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            res = await response.json()
            return res


@tool
async def driving_route(
    origin: str,
    destination: str,
    origin_id: Optional[int] = None,
    destination_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    驾车路线规划工具

    参数:
        origin: 起点经纬度，经度在前，纬度在后，经度和纬度用","分割（必需）
                经纬度小数点后不得超过6位
                例如："116.397428,39.90923"
        destination: 终点经纬度，经度在前，纬度在后，经度和纬度用","分割（必需）
                     经纬度小数点后不得超过6位
        origin_id: 起点POI ID，起点为POI时，建议填充此值，可提升路线规划准确性（可选）
        destination_id: 目的地POI ID，目的地为POI时，建议填充此值，可提升路线规划准确性（可选）
    """
    print(
        "开始执行driving_route工具，规划驾车路线，参数：",
        origin,
        destination,
        origin_id,
        destination_id,
    )
    # 从环境变量获取API Key
    key = os.getenv("GAODE_MAP_KEY")
    if not key:
        raise EnvironmentError("未配置高德地图API Key，请设置环境变量GAODE_MAP_KEY")

    # 函数内部指定的默认参数
    output = "json"  # 返回结果格式类型，默认json
    strategy = 32  # 默认，高德推荐
    # 其他参数使用默认值（不传）

    # 构建参数字典
    params: Dict[str, Any] = {
        "origin": origin,
        "destination": destination,
        "key": key,
        "output": output,
        "strategy": strategy,
    }

    # 可选参数：只在有值时才添加
    if origin_id is not None:
        params["origin_id"] = origin_id

    if destination_id is not None:
        params["destination_id"] = destination_id

    url = "https://restapi.amap.com/v5/direction/driving"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            res = await response.json()
            return res


@tool
async def driving_route_handoff_single(
    origin: str,
    destination: str,
    runtime: ToolRuntime[Gaodemap_State_Handoff_Single],
    origin_id: Optional[int] = None,
    destination_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    驾车路线规划工具

    参数:
        origin: 起点经纬度，经度在前，纬度在后，经度和纬度用","分割（必需）
                经纬度小数点后不得超过6位
                例如："116.397428,39.90923"
        destination: 终点经纬度，经度在前，纬度在后，经度和纬度用","分割（必需）
                     经纬度小数点后不得超过6位
        origin_id: 如果存在该起点为POI信息时，建议填充此值，可提升路线规划准确性（可选）
        destination_id: 如果存在该目的地为POI信息时，建议填充此值，可提升路线规划准确性（可选）
    """
    print(
        "开始执行driving_route工具，规划驾车路线，参数：",
        origin,
        destination,
        origin_id,
        destination_id,
    )
    # 从环境变量获取API Key
    key = os.getenv("GAODE_MAP_KEY")
    if not key:
        raise EnvironmentError("未配置高德地图API Key，请设置环境变量GAODE_MAP_KEY")

    # 函数内部指定的默认参数
    output = "json"  # 返回结果格式类型，默认json
    strategy = 32  # 默认，高德推荐
    # 其他参数使用默认值（不传）

    # 构建参数字典
    params: Dict[str, Any] = {
        "origin": origin,
        "destination": destination,
        "key": key,
        "output": output,
        "strategy": strategy,
    }

    # 可选参数：只在有值时才添加
    if origin_id is not None:
        params["origin_id"] = origin_id

    if destination_id is not None:
        params["destination_id"] = destination_id

    url = "https://restapi.amap.com/v5/direction/driving"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            res = await response.json()

            route = res.get("route", {})
            path = route.get("paths", [])[0]
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"{path}", tool_call_id=runtime.tool_call_id
                        )
                    ],
                    "current_step": "around_search_step",
                }
            )


@tool
async def walking_route(
    origin: str,
    destination: str,
    origin_id: Optional[str] = None,
    destination_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    步行路线规划工具

    参数:
        origin: 起点经纬度，经度在前，纬度在后，经度和纬度用","分割（必需）
                经纬度小数点后不得超过6位
                例如："116.397428,39.90923"
        destination: 终点经纬度，经度在前，纬度在后，经度和纬度用","分割（必需）
                      经纬度小数点后不得超过6位
        origin_id: 起点POI ID，起点为POI时，建议填充此值，可提升路线规划准确性（可选）
        destination_id: 目的地POI ID，目的地为POI时，建议填充此值，可提升路线规划准确性（可选）
    """
    print(
        "开始执行walking_route工具，规划步行路线，参数：",
        origin,
        destination,
        origin_id,
        destination_id,
    )
    # 从环境变量获取API Key
    key = os.getenv("GAODE_MAP_KEY")
    if not key:
        raise EnvironmentError("未配置高德地图API Key，请设置环境变量GAODE_MAP_KEY")

    # 函数内部指定的默认参数
    output = "json"  # 返回结果格式类型，默认json
    # alternative_route 不传则默认返回一条路线方案
    # isindoor 不传则默认不需要室内算路
    # show_fields 不传则只返回基础信息类内字段

    # 构建参数字典
    params: Dict[str, Any] = {
        "origin": origin,
        "destination": destination,
        "key": key,
        "output": output,
    }

    # 可选参数：只在有值时才添加
    if origin_id is not None:
        params["origin_id"] = origin_id

    if destination_id is not None:
        params["destination_id"] = destination_id

    url = "https://restapi.amap.com/v5/direction/walking"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            res = await response.json()
            return res


@tool
async def transit_route(
    origin: str,
    destination: str,
    origin_id: Optional[str] = None,
    destination_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    公交路线规划工具

    参数:
        origin: 起点经纬度，经度在前，纬度在后，经度和纬度用","分割（必需）
                经纬度小数点后不得超过6位
                例如："116.397428,39.90923"
        destination: 目的地经纬度，经度在前，纬度在后，经度和纬度用","分割（必需）
                     经纬度小数点后不得超过6位
        origin_id: 起点POI ID（可选）
                   起点POI ID与起点经纬度均填写时，服务使用起点POI ID
                   该字段必须和目的地POI ID成组使用
        destination_id: 目的地POI ID（可选）
                        目的地POI ID与目的地经纬度均填写时，服务使用目的地POI ID
                        该字段必须和起点POI ID成组使用
    """
    print(
        "开始执行transit_route工具，规划公交路线，参数：",
        origin,
        destination,
        origin_id,
        destination_id,
    )
    # 从环境变量获取API Key
    key = os.getenv("GAODE_MAP_KEY")
    if not key:
        raise EnvironmentError("未配置高德地图API Key，请设置环境变量GAODE_MAP_KEY")

    # 函数内部指定的默认参数
    output = "json"  # 返回结果格式类型，默认json
    strategy = "0"  # 推荐模式，综合权重，同高德APP默认
    # 其他参数使用默认值（不传）

    # 构建参数字典
    params: Dict[str, Any] = {
        "origin": origin,
        "destination": destination,
        "key": key,
        "output": output,
        "strategy": strategy,
    }

    # 可选参数：只在有值时才添加
    # 注意：API使用originpoi和destinationpoi，但函数参数使用origin_id和destination_id以保持一致性
    if origin_id is not None:
        params["originpoi"] = origin_id

    if destination_id is not None:
        params["destinationpoi"] = destination_id

    url = "https://restapi.amap.com/v5/direction/transit/integrated"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            res = await response.json()
            return res


async def calculate_distance(
    origins: List[List[float]], destination: List[List[float]]
) -> Dict[str, Any]:
    """
    批量计算多个起点到一个终点的距离

    参数:
        origins: 起点经纬度列表，每个元素为[经度, 纬度]，长度不超过100（必需）
                 例如：[[116.481028, 39.989643], [114.481028, 39.989643]]
        destination: 终点经纬度列表，包含一个元素[经度, 纬度]，长度必须为1（必需）
                     例如：[[114.465302, 40.004717]]

    """
    print("开始执行calculate_distance工具，参数：", origins, destination)

    # 参数验证
    if not origins or len(origins) == 0:
        raise ValueError("起点列表不能为空")
    if len(origins) > 100:
        raise ValueError("起点列表长度不能超过100")
    if len(destination) != 1:
        raise ValueError("终点列表长度必须为1")
    if len(destination[0]) != 2:
        raise ValueError("终点必须是包含经度和纬度的列表，格式为[经度, 纬度]")

    # 从环境变量获取API Key
    key = os.getenv("GAODE_MAP_KEY")
    if not key:
        raise EnvironmentError("未配置高德地图API Key，请设置环境变量GAODE_MAP_KEY")

    # 构建起点参数字符串，格式：经度,纬度|经度,纬度|...
    origins_str = "|".join([f"{lon},{lat}" for lon, lat in origins])

    # 构建终点参数字符串，格式：经度,纬度
    destination_str = f"{destination[0][0]},{destination[0][1]}"

    # 构建参数字典
    params: Dict[str, Any] = {
        "origins": origins_str,
        "destination": destination_str,
        "type": 1,  # 1：直线距离，0：驾车距离
        "key": key,
        "output": "JSON",
    }

    url = "https://restapi.amap.com/v3/distance"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            res = await response.json()

            res = res.get("results", [])

            for result in res:
                if "duration" in result:
                    del result["duration"]
            return res


####################################THINKING TOOL####################################
@tool
async def guiding_deepthink(thinking: str):
    """
    你的任务是对当前检索到的内容进行深度思考。包括对照目前的已经获取到的信息，深度思考自己下一步的规划，以改变下一次查询的关键字来可能获取更多信息，尝试完善目前的规划：

    什么是关键字？
    district_search使用的关键字是对用户的目的进行大类（如果有的话），中类（如果有的话），小类（如果有的话）的描述。
    例如：大类包括生活服务，金融服务，政府机构等（大类没有固定的名称）。中类则是稍微细化的分类，例如银行，酒店，体育馆。小类则是最精细的分类，例如中国银行，宜家，五棵松等。三种分类至少有一个。
    你的关键字的方向需要顺着上面的要求来改变。

    优化district_search的方向:
        1:检索回来的结果为空，或者结果较少，基本是由于以下原因：
            1.1:行政区的名称错误，包括使用了缩写，比如东北，大西北扥。使用了俗称，比如：成渝，京津，珠三角，长三角等。
            1.2:关键字中存在抽象的概念。district_search中的关键字是针对具体地点的类别，比如美食，医疗，住宿，景点等。而不是路线，安排，攻略抽象的概念。
            1.3:检索的关键字与用户的目的不符。

        2:检索回来的结果不符合目前的需求，基本是由于以下原因造成的：
            2.1:检索的地点太广，比如：成都，西安等，这会导致检索到的地方离用户的目的地较远，从而导致你使用相同的参数渴望检测到不同的结果
            2.2:反复使用相同的参数来进行检索，相同的参数返回的结果一定是相同的。你需要调整参数或使用around_search工具来精细化检索指定周边的场所。

    优化around_search的方向：
        1:around_search的目的是输入一个地点的经纬度坐标，然后检索该地点周边的场所。它适合在确定某一个位置以后，检索该位置周边的相关的场所。当你想要知道周边的场所时，这非常有用而不是采用district_search工具。
        2:与district_search工具不同，around_search工具的输入参数是地点的经纬度坐标，是针对某一个地点的周边区域的精细化检索。
        3:反复使用相同的参数来进行检索，相同的参数返回的结果一定是相同的。你需要调整参数或使用around_search工具来精细化检索指定周边的场所。

    当你认为目前的信息已充足，可以总结路线反馈给用户以后，整合所有信息，返回给用户一个总结。
    """
    return thinking


@tool
async def district_deepthink(thinking: str):
    """
    你的任务是对当前检索到的内容进行深度思考。包括对照目前的已经获取到的信息，深度思考自己下一步的规划，改变下一次查询的关键字来可能获取更多信息还是终止district_search工具的调用来总结目前的内容。

    如果当前的信息缺乏或偏离了用户的需求，同时是因为检索的关键字出现了问题。包括但不限于：
    1：行政区的名称错误，包括使用了缩写，比如东北，大西北扥。使用了俗称，比如：成渝，京津，珠三角，长三角等。
    2：关键字中存在抽象的概念。district_search中的关键字是针对具体地点的类别，比如美食，宾馆，景点等。而不是路线，安排，攻略抽象的概念。

    如果当前的信息没有偏离，你需要适当丰富信息，从检索的地点中，进一步利用新的关键词扩展出新的地点。

    如果用户的需求是路线，安排，攻略等抽象的概念，你需要转化为具象的概念，例如景点，宾馆，美食等进行检索。禁止拒答。

    当前检索到的信息已经足够，可以终止所有的工具调用，整合所有信息，返回给用户检索到的内容。
    """
    return thinking


################################# HANDOFF SINGLE AGENT TOOLS ##############################################
@tool
async def back_to_geocode() -> Dict[str, Any]:
    """
    将当前阶段返回到geocode_step
    """
    return Command(update={"current_step": "geocode_step"})


@tool
async def back_to_around_search() -> Dict[str, Any]:
    """
    将当前阶段返回到around_search_step
    """
    return Command(update={"current_step": "around_search_step"})


@tool
async def back_to_driving_route() -> Dict[str, Any]:
    """
    将当前阶段返回到driving_route_step
    """
    return Command(update={"current_step": "driving_route_step"})


@tool
async def jump_to_other(
    step: Literal["geocode_step", "around_search_step", "driving_route_step"],
):
    """
    将当前阶段跳转到指定的阶段
    """
    return Command(update={"current_step": step})


################################# HANDOFF SINGLE AGENT V2 TOOLS ##############################################
@tool
def jump_to_around_search_agent(
    runtime: ToolRuntime[Gaodemap_State_Handoff_Single_V2],
) -> Command:
    """
    将当前阶段跳转到around_search_agent_step
    """
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"将当前阶段跳转到around_search_agent_step",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "current_step": "around_search_agent_step",
        }
    )


@tool
def jump_to_path_planning_agent(
    runtime: ToolRuntime[Gaodemap_State_Handoff_Single_V2],
) -> Command:
    """
    将当前阶段跳转到path_planning_agent
    """
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"将当前阶段跳转到path_planning_agent_step",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "current_step": "path_planning_agent_step",
        }
    )


@tool
async def driving_route_handoff_single_v2(
    origin: str,
    destination: str,
    runtime: ToolRuntime[Gaodemap_State_Handoff_Single_V2],
    origin_id: Optional[int] = None,
    destination_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    驾车路线规划工具

    参数:
        origin: 起点经纬度，经度在前，纬度在后，经度和纬度用","分割（必需）
                经纬度小数点后不得超过6位
                例如："116.397428,39.90923"
        destination: 终点经纬度，经度在前，纬度在后，经度和纬度用","分割（必需）
                     经纬度小数点后不得超过6位
        origin_id: 如果存在该起点为POI信息时，建议填充此值，可提升路线规划准确性（可选）
        destination_id: 如果存在该目的地为POI信息时，建议填充此值，可提升路线规划准确性（可选）
    """
    print(
        "开始执行driving_route工具，规划驾车路线，参数：",
        origin,
        destination,
        origin_id,
        destination_id,
    )
    # 从环境变量获取API Key
    key = os.getenv("GAODE_MAP_KEY")
    if not key:
        raise EnvironmentError("未配置高德地图API Key，请设置环境变量GAODE_MAP_KEY")

    # 函数内部指定的默认参数
    output = "json"  # 返回结果格式类型，默认json
    strategy = 32  # 默认，高德推荐
    # 其他参数使用默认值（不传）

    # 构建参数字典
    params: Dict[str, Any] = {
        "origin": origin,
        "destination": destination,
        "key": key,
        "output": output,
        "strategy": strategy,
    }

    # 可选参数：只在有值时才添加
    if origin_id is not None:
        params["origin_id"] = origin_id

    if destination_id is not None:
        params["destination_id"] = destination_id

    url = "https://restapi.amap.com/v5/direction/driving"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            res = await response.json()

            route = res.get("route", {})
            path = route.get("paths", [])[0]
            # return Command(
            #     update={
            #         "messages": [
            #             ToolMessage(
            #                 content=f"{path}", tool_call_id=runtime.tool_call_id
            #             )
            #         ],
            #         "current_step": "around_search_step",
            #     }
            # )
            return path

################################# HANDOFF MULTI AGENTS TOOLS ##############################################
@tool
async def jump_to_around_search_agent_multi(
    runtime: ToolRuntime[Gaodemap_State_Handoff_Multi],
) -> Command:
    """
    将当前阶段跳转到call_around_node
    """
    from langchain.messages import AIMessage

    last_ai_message = next(
        msg for msg in reversed(runtime.state["messages"]) if isinstance(msg, AIMessage)
    )
    transfer_message = ToolMessage(
        content="将当前阶段跳转到call_around_node",
        tool_call_id=runtime.tool_call_id,
    )
    return Command(
        update={
            "goto": "call_around_node",
            "messages": [last_ai_message, transfer_message],
            "current_step": "call_around_node",
        },
        graph=Command.PARENT,
    )


@tool
async def jump_to_path_planning_agent_multi(
    runtime: ToolRuntime[Gaodemap_State_Handoff_Multi],
) -> Command:
    """
    将当前阶段跳转到call_path_node
    """
    from langchain.messages import AIMessage

    last_ai_message = next(
        msg for msg in reversed(runtime.state["messages"]) if isinstance(msg, AIMessage)
    )
    transfer_message = ToolMessage(
        content="将当前阶段跳转到call_path_node",
        tool_call_id=runtime.tool_call_id,
    )
    return Command(
        update={
            "goto": "call_path_node",
            "messages": [last_ai_message, transfer_message],
            "current_step": "call_path_node",
        },
        graph=Command.PARENT,
    )

################################# SKILLS TOOLS ##############################################
@tool
async def load_skill(skill_name: str,runtime:ToolRuntime) -> Command:
    """将skill的主要内容加载到智能体的上下文中。

    当你需要详细了解如何处理特定类型的请求时，请使用此工具。
    这将为你提供关于技能领域的全面指导、政策和指南。

    你目前有两个skill可以加载：
    - around_search: 用于检索周边区域地点
    - path_planning: 用于规划路线

    使用skill的规则如下：
    1. 如果用户只是询问周边区域地点检索，并没有路线规划的需求。只需要加载around_search
    2. 如果用户只询问路线规划，并没有对特定区域进行检索的需求。只需要加载path_planning
    3. 如果用户既有周边区域地点检索的需求，又有路线规划的需求。请先加载around_search。在得到周边检索结果以后，再加载path_planning来获取路线规划结果
    

    Args:
        skill_name: 要加载的技能的名称（例如“around_search”，“path_planning”）
    """
    for skill in SKILLS:
        if skill["name"] == skill_name:
            return Command(
                update={
                    "messages":[
                        ToolMessage(content=f"已加载Skill: {skill_name}\n\n{skill['content']}",tool_call_id=runtime.tool_call_id)
                    ],
                    "skill_name":skill_name
                }
            )
            
    available = ", ".join(s["name"] for s in SKILLS)
    return Command(
        update={
            "messages":[
                ToolMessage(content=f"Skill '{skill_name}' 未找到。可用的技能: {available}",tool_call_id=runtime.tool_call_id)
            ]
        }
    )


if __name__ == "__main__":
    import asyncio

    # 测试关键字搜索
    print("=== 测试关键字搜索 ===")
    res = asyncio.run(district_search("二仙桥", "成都"))
    print(res)

    # # 测试周边搜索
    # print("\n=== 测试周边搜索 ===")
    # res = asyncio.run(around_search("116.473168,39.993015", "美食"))
    # print(res)
    # test_polygon = "116.39748,39.90872|116.40748,39.90872|116.40748,39.91872|116.39748,39.90872"
    # res = asyncio.run(polygon_search(test_polygon, "美食", "北京"))
    # print(res)

    # test_poi_id = "B0JDPRN8A5"
    # res = asyncio.run(id_search(test_poi_id))
    # print(res)

    # print("=== 测试地理编码 ===")
    # res = asyncio.run(geocode("华西医院","成都",batch=False))
    # print(res)

    # print("=== 测试逆地理编码 ===")
    # res = asyncio.run(regeocode("113.620685,34.749012","购物"))
    # print(res)
    # print("=== 查询天气信息 ===")
    # res = asyncio.run(weather_query("411422","all"))
    # print(res)
    # print("=== 驾车路线规划 ===")
    # res = asyncio.run(driving_route("成都天府三街", "成都东郊记忆",None,None))
    # print(res)
    # print("=== 步行路线规划 ===")
    # res = asyncio.run(walking_route("113.620685,34.749012", "113.620645,34.74348","B0JK2CU2RR","B01730IHYA"))
    # print(res)
    # print("=== 公交路线规划 ===")
    # res = asyncio.run(transit_route("116.397428,39.90923", "116.473168,39.993015","B0JK2CU2RR","B01730IHYA"))
    # print(res)

    # print("=== 计算距离 ===")
    # res = asyncio.run(calculate_distance([[113.620685,34.749012],[113.620645,34.74348]],[[113.620645,34.74348]]))
    # print(res)
