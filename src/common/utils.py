"""Utility & helper functions."""

from typing import Optional, Union

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_qwq import ChatQwen, ChatQwQ


def normalize_region(region: str) -> Optional[str]:
    """Normalize region aliases to standard values.

    Args:
        region: Region string to normalize

    Returns:
        Normalized region ('prc' or 'international') or None if invalid
    """
    if not region:
        return None

    region_lower = region.lower()
    if region_lower in ("prc", "cn"):
        return "prc"
    elif region_lower in ("international", "en"):
        return "international"
    return None


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(
    fully_specified_name: str,
) -> Union[BaseChatModel, ChatQwQ, ChatQwen]:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider:model'.
    """
    provider, model = fully_specified_name.split(":", maxsplit=1)
    provider_lower = provider.lower()

    # Handle Qwen models specially with dashscope integration
    if provider_lower == "qwen":
        from .models import create_qwen_model

        return create_qwen_model(model)

    # Handle SiliconFlow models
    if provider_lower == "siliconflow":
        from .models import create_siliconflow_model

        return create_siliconflow_model(model)

    # Use standard langchain initialization for other providers
    return init_chat_model(model, model_provider=provider)

def _remove_none_values(data: dict) -> dict:
    """
    递归删除字典中值为 None 或空列表的键值对
    
    Args:
        data: 要清理的字典
    
    Returns:
        清理后的字典（不包含值为 None 或空列表的键值对）
    """
    if not isinstance(data, dict):
        return data
    
    result = {}
    for key, value in data.items():
        # 跳过 None 值
        if value is None:
            continue
        # 跳过空列表
        elif isinstance(value, list) and len(value) == 0:
            continue
        elif isinstance(value, dict):
            # 递归处理嵌套字典
            cleaned = _remove_none_values(value)
            # 如果清理后的字典不为空，则保留
            if cleaned:
                result[key] = cleaned
        else:
            result[key] = value
    
    return result


async def parse_poi_detail(data: dict) -> list:

    if data.get('status') != 0:
        raise ValueError(f"API 请求失败: {data.get('message', '未知错误')}")

    result = data['result']
    
    # 情况1: result 是列表（搜索结果）
    if isinstance(result, list):
        pois = []
        for item in result:
            pois.append(await _extract_basic_poi(item))
        return pois

    # 情况2: result 是字典（详情接口）
    elif isinstance(result, dict):
        # 构造一个只含该 POI 的列表
        return [await _extract_detailed_poi(result)]

    else:
        raise TypeError("不支持的 result 类型")


async def _extract_basic_poi(item: dict) -> dict:
    """从搜索结果中的单个 POI 提取基础信息"""
    poi_data = {
        'name': item.get('name'),
        'uid': item.get('uid'),
        'address': item.get('address'),
        'province': item.get('province'),
        'city': item.get('city'),
        'area': item.get('district'),  # 注意：这里是 'district' 而非 'area'
        'town': item.get('town'),
        'telephone': None,  # 搜索结果中通常没有电话
        'location': {
            'lat': item['location']['lat'],
            'lng': item['location']['lng']
        },
        'navi_location': None,
        'brand': None,
        'price': None,
        'overall_rating': None,
        'taste_rating': None,
        'service_rating': None,
        'environment_rating': None,
        'shop_hours': None,
        'atmosphere': [],
        'featured_service': [],
        'tags': {
            'classified_poi_tag': item.get('classified_poi_tag'),
            'tag': item.get('tag')
        },
        'detail_url': None
    }
    return _remove_none_values(poi_data)


async def _extract_detailed_poi(result: dict) -> dict:
    """从详情接口的 result 中提取完整信息"""
    detail = result.get('detail_info', {})
    
    poi_data = {
        'name': result.get('name'),
        'uid': result.get('uid'),
        'address': result.get('address'),
        'province': result.get('province'),
        'city': result.get('city'),
        'area': result.get('area'),  # 详情接口用 'area'
        'town': result.get('town'),
        'telephone': result.get('telephone'),
        'location': {
            'lat': result['location']['lat'],
            'lng': result['location']['lng']
        },
        'navi_location': {
            'lat': detail.get('navi_location', {}).get('lat'),
            'lng': detail.get('navi_location', {}).get('lng')
        } if detail.get('navi_location') else None,
        'brand': detail.get('brand'),
        'price': float(detail['price']) if detail.get('price') and detail['price'].replace('.', '', 1).isdigit() else None,
        'overall_rating': float(detail['overall_rating']) if detail.get('overall_rating') and detail['overall_rating'].replace('.', '', 1).isdigit() else None,
        'taste_rating': float(detail['taste_rating']) if detail.get('taste_rating') and detail['taste_rating'].replace('.', '', 1).isdigit() else None,
        'service_rating': float(detail['service_rating']) if detail.get('service_rating') and detail['service_rating'].replace('.', '', 1).isdigit() else None,
        'environment_rating': float(detail['environment_rating']) if detail.get('environment_rating') and detail['environment_rating'].replace('.', '', 1).isdigit() else None,
        'shop_hours': detail.get('shop_hours'),
        'atmosphere': detail.get('atmosphere', []),
        'featured_service': detail.get('featured_service', []),
        'tags': {
            'classified_poi_tag': detail.get('classified_poi_tag'),
            'tag': detail.get('tag')
        },
        'detail_url': detail.get('detail_url')
    }
    return _remove_none_values(poi_data)

async def parse_poi_data(data):
    """
    通用解析百度地图 POI 数据，适用于多种餐饮类结果。
    保留 uid，排除 navi_location 和 detail_url。
    
    Args:
        data (dict): 百度地图 Place API 返回的原始 JSON 数据
    
    Returns:
        List[dict]: 结构化后的餐厅信息列表
    """
    results = []
    
    for item in data.get("results", []):
        # 基础字段（uid 必须）
        uid = item.get("uid")
        name = item.get("name")
        address = item.get("address")
        telephone = item.get("telephone")  # 可能为 None 或多号码
        province = item.get("province")
        city = item.get("city")
        area = item.get("area")
        location = item.get("location")  # dict or None
        
        # detail_info 字段（可能部分缺失）
        detail_info = item.get("detail_info", {})
        brand = detail_info.get("brand")  # 注意：有些店没有 brand
        price = detail_info.get("price")
        overall_rating = detail_info.get("overall_rating")
        shop_hours = detail_info.get("shop_hours")  # 可能为空字符串
        label = detail_info.get("label")  # 菜系/品类，如 "鱼火锅"、"烧鸡"
        
        # 安全转换数值字段
        try:
            price_per_person = float(price) if price else None
        except (ValueError, TypeError):
            price_per_person = None

        try:
            rating = float(overall_rating) if overall_rating else None
        except (ValueError, TypeError):
            rating = None


        # 构建结构化记录
        record = {
            "uid": uid,
            "name": name,
            "brand": brand,  # 可能为 None
            "address": address,
            "province": province,
            "city": city,
            "area": area,
            "telephone": telephone,
            "latitude": location.get("lat") if location else None,
            "longitude": location.get("lng") if location else None,
            "price_per_person": price_per_person,
            "rating": rating,
            "opening_hours": shop_hours or None,  # 空字符串转为 None（可选）
            "cuisine_type": label
        }
        
        results.append(_remove_none_values(record))
    
    return results


async def gaode_parse_key_words_and_around_search(data):
    """
    解析高德地图关键字搜索返回数据（兼容1.0和2.0版本）
    并移除值为 None 或空列表的字段
    """
    if data.get('status') != '1' or data.get('infocode') != '10000':
        raise ValueError("API 返回状态异常")

    # 处理pois可能是数组或对象的情况
    pois_raw = data.get('pois', [])
    
    if isinstance(pois_raw, dict):
        pois = pois_raw.get('poi', []) if 'poi' in pois_raw else []
        if not pois and pois_raw:
            pois = [pois_raw]
    elif isinstance(pois_raw, list):
        pois = pois_raw
    else:
        pois = []
    
    parsed_list = []

    for poi in pois:
        name = poi.get('name', '').strip()
        address = poi.get('address', '').strip()
        city = poi.get('cityname', '')
        district = poi.get('adname', '')
        location = poi.get('location', '')
        
        try:
            lng, lat = map(float, location.split(',')) if location else (None, None)
        except (ValueError, AttributeError):
            lng, lat = None, None

        business = poi.get('business', {})
        biz_ext = poi.get('biz_ext', {})
        
        if business:
            tel = business.get('tel', '').strip()
            cost = business.get('cost')
            rating = business.get('rating')
            tag = business.get('tag', '')
            business_area = business.get('business_area', '')
        else:
            tel = poi.get('tel', '').strip() or biz_ext.get('tel', '').strip()
            cost = biz_ext.get('cost')
            rating = biz_ext.get('rating')
            tag = poi.get('tag', '') or poi.get('atag', '') or poi.get('keytag', '')
            business_area = poi.get('business_area', '')

        tags = (
            [t.strip() for t in tag.split(',') if t.strip()]
            or [t.strip() for t in poi.get('tag', '').split(',') if t.strip()]
            or [t.strip() for t in poi.get('atag', '').split(',') if t.strip()]
            or ([poi.get('keytag')] if poi.get('keytag') else [])
        )

        photos_obj = poi.get('photos', {})
        photos_list = poi.get('photos', [])
        
        if isinstance(photos_obj, dict) and 'url' in photos_obj:
            photo_url = photos_obj.get('url')
        elif isinstance(photos_list, list) and photos_list:
            photo_url = photos_list[0].get('url') if isinstance(photos_list[0], dict) and 'url' in photos_list[0] else None
        else:
            photo_url = None

        distance = poi.get('distance')
        distance_m = int(distance) if isinstance(distance, str) and distance.isdigit() else None

        simplified = {
            'name': name,
            'address': address,
            'tel': tel,
            'city': city,
            'district': district,
            'business_area': business_area,
            'distance_meters': distance_m,
            'location': {'longitude': lng, 'latitude': lat} if lng is not None and lat is not None else None,
            'cost_per_person': float(cost) if cost and str(cost).replace('.', '', 1).isdigit() else None,
            'rating': float(rating) if rating and str(rating).replace('.', '', 1).isdigit() else None,
            'tags': tags,
            'photo_url': photo_url,
            'poi_id': poi.get('id', ''),
        }

        # 过滤掉值为 None 或 [] 的项
        filtered = {
            k: v for k, v in simplified.items()
            if v is not None and v != []
        }
        parsed_list.append(filtered)

    return parsed_list


async def gaode_parse_polygon_search(data):
    if data.get('status') != '1' or data.get('infocode') != '10000':
        raise ValueError("API 返回状态异常，请检查数据")

    pois = data.get('pois', [])
    results = []

    for poi in pois:
        name = poi.get('name', '').strip()
        address = poi.get('address', '').strip()
        city = poi.get('cityname', '')
        district = poi.get('adname', '')
        location_str = poi.get('location', '')
        
        try:
            lng, lat = map(float, location_str.split(',')) if location_str else (None, None)
        except (ValueError, AttributeError):
            lng, lat = None, None

        business = poi.get('business', {})
        cost = business.get('cost')
        rating = business.get('rating')
        tel = business.get('tel', '')
        business_area = business.get('business_area', '')
        keytag = business.get('keytag', '')
        rectag = business.get('rectag', '')
        tag_str = business.get('tag', '')

        tags = []
        if tag_str:
            tags = [t.strip() for t in tag_str.split(',') if t.strip()]
        elif rectag:
            tags = [rectag]
        elif keytag:
            tags = [keytag]

        photos = poi.get('photos', [])
        photo_url = None
        if photos and isinstance(photos[0], dict) and 'url' in photos[0]:
            photo_url = photos[0]['url']

        distance_raw = poi.get('distance', '')
        distance_m = int(distance_raw) if distance_raw.isdigit() else None

        item = {
            'name': name,
            'address': address,
            'tel': tel,
            'city': city,
            'district': district,
            'business_area': business_area,
            'distance_meters': distance_m,
            'location': {'longitude': lng, 'latitude': lat} if lng is not None and lat is not None else None,
            'cost_per_person': float(cost) if cost and cost.replace('.', '', 1).isdigit() else None,
            'rating': float(rating) if rating and rating.replace('.', '', 1).isdigit() else None,
            'tags': tags,
            'photo_url': photo_url,
            'poi_id': poi.get('id', ''),
        }

        # 过滤掉值为 None 或 [] 的项
        filtered = {
            k: v for k, v in item.items()
            if v is not None and v != []
        }
        results.append(filtered)

    return results

async def gaode_parse_poi_search(poi_dict: dict) -> dict:
    
    pois_list = poi_dict.get('pois', [])
    if pois_list and isinstance(pois_list[0], dict):
        main_poi = pois_list[0]  # 取列表中的第一条核心POI数据
        business_info = main_poi.get('business', {})  # 提取商户经营信息
        photos_list = main_poi.get('photos', [])  # 提取图片信息
        
        poi_detail = {
            'name': main_poi.get('name'),
            'address': main_poi.get('address'),
            'cityname': main_poi.get('cityname'),
            'adname': main_poi.get('adname'),
            'type': main_poi.get('type'),
            'tel': business_info.get('tel'),
            'keytag': business_info.get('keytag'),
            'rating': business_info.get('rating'),
            'cost': business_info.get('cost'),
            'opentime_today': business_info.get('opentime_today'),
            'opentime_week': business_info.get('opentime_week'),
            'business_area': business_info.get('business_area'),
            'location': main_poi.get('location'),
            'id': main_poi.get('id'),
            'photo_count': len(photos_list) if photos_list else None,
            'main_photo_url': photos_list[0].get('url') if photos_list else None
        }
        
        # 过滤核心POI信息中值为None或空列表的键值对
        filtered_poi_detail = {k: v for k, v in poi_detail.items() if v is not None and v != []}
    
    return filtered_poi_detail

async def gaode_parse_geocode(data):
    if data.get('status') != '1' or data.get('infocode') != '10000':
        raise ValueError("API 返回状态异常，请检查数据")
        
    position_list = data.get('geocodes', [])

    location_list = []
    for position in position_list:
        # 处理 location 字段：可能是字符串或列表
        location = position.get('location', '')
        if isinstance(location, list):
            # 如果是列表（通常是空列表），跳过该项
            continue
        if not location or not isinstance(location, str):
            # 如果不是字符串或为空，跳过该项
            continue
        
        # 解析经纬度
        try:
            lng, lat = location.split(',')
        except (ValueError, AttributeError):
            # 如果 split 失败，跳过该项
            continue
        
        # 处理 formatted_address 字段：可能是字符串或列表
        formatted_address = position.get('formatted_address', '')
        if isinstance(formatted_address, list):
            # 如果是列表，尝试取第一个元素，否则使用空字符串
            formatted_address = formatted_address[0] if formatted_address else ''
        if not isinstance(formatted_address, str):
            formatted_address = str(formatted_address) if formatted_address else ''

        location_list.append({
            "name": formatted_address,
            "longitude": lng,
            "latitude": lat
        })

    return location_list

test = {
  'status': '1',
   'info': 'OK',
   'infocode': '10000',
   'count': '1',
   'route': {
    'origin': '113.620685,34.749012',
     'destination': '113.620645,34.74348',
     'paths': [
      {
        'distance': '604',
         'cost': {
          'duration': '483'
        },
         'steps': [
          {
            'instruction': '沿工人路向南步行604米到达目的地',
             'orientation': '南',
             'road_name': '工人路',
             'step_distance': '604'
          }
        ]
      }
    ]
  }
}
def parse_path_planning(data):

    route = data.get('route', {})
    path = route.get('paths', [])[0]

    return path