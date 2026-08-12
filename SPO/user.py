from typing import List, TypedDict


class DeskSalespersonUserPersona(TypedDict):
    # 用户喜好
    preference: str
    # 用户的憎恶点
    hate: str
    # 用户不喜欢的商品id列表
    dislike_goods_id: List[int]
    # 用户不喜欢的品牌列表
    dislike_brand: List[str]
    # 用户不喜欢的风格
    dislike_style: str
    # 用户喜欢的商品id列表
    like_goods_id: List[int]
    # 用户喜欢的品牌列表
    like_brand: List[str]
    # 用户喜欢的风格
    like_style: str
