import json
import re
import json_repair
from typing import List, Dict, Set, Any
from collections import Counter
import logging

def compute_score(rank1: int, rank2: int, total: int, w1: float = 1.0, w2: float = 1.0) -> float:
    """
    计算两个排名的加权分数
    
    Args:
        rank1: 第一个排名
        rank2: 第二个排名
        total: 总数
        w1: 第一个权重
        w2: 第二个权重
        
    Returns:
        加权分数
    """
    score = w1 * (total - rank1 + 1) + w2 * (total - rank2 + 1)
    return score


def validate_json(data: Dict) -> bool:
    """
    验证JSON数据是否符合预期格式
    
    Args:
        data: 要验证的JSON数据
        
    Returns:
        是否符合预期格式
    """
    # 检查是否有且仅有两个键: "mashup" 和 "related_apis"
    if set(data.keys()) != {"mashup", "related_apis"}:
        return False
    
    # 检查 "mashup" 是否为字符串
    if not isinstance(data["mashup"], str):
        return False
    
    # 检查 "related_apis" 是否为列表，并且其中所有元素都是字符串
    if not isinstance(data["related_apis"], list) or not all(isinstance(api, str) for api in data["related_apis"]):
        return False
    
    return True


def process_json_pattern(text: str) -> Dict:
    """
    从文本中提取JSON代码块并解析
    
    Args:
        text: 包含JSON代码块的文本
        
    Returns:
        解析后的JSON数据，如果解析失败则返回空的related_apis列表
    """
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            json_data = json.loads(match.strip())  # 转换为 JSON 对象
            # 检查是否存在 "related_apis" 键
            if "related_apis" not in json_data:
                logging.error("解析的 JSON 缺少 'related_apis' 键")
                return {"related_apis": []}
            return json_data
        except json.JSONDecodeError as e:
            logging.error("解析失败: %s", e)
            return {"related_apis":[]}
        except Exception as e:
            logging.error("没有找到JSON的开始或结束符: %s", e)
            return {"related_apis":[]}
    return {"related_apis": []}


def process_json(text: str) -> Dict:
    """
    从文本中提取并解析JSON数据
    
    Args:
        text: 包含JSON数据的文本
        
    Returns:
        解析后的JSON数据，如果解析失败则返回空的related_apis列表
    """
    try:
        json_start = text.index("{")
        json_end = text.index("}")
        json_str = text[json_start : json_end + 1]
        
        json_str = json_repair.repair_json(json_str)
        json_data = json.loads(json_str)
        if not validate_json(json_data):
            return process_json_pattern(text)
        return json_data

    except Exception as e:
        logging.error("没有找到JSON的开始或结束符: %s", e)

        return {"related_apis":[]}


def count_wrong_answer(api_list: List[str], apis: List[Dict]) -> int:
    """
    计算预测API中的错误数量
    
    Args:
        api_list: 预测的API列表
        apis: 真实的API列表
        
    Returns:
        错误的API数量
    """
    if len(api_list) == 0:
        return 10
    wrong_answer = 0
    apis = [a['title'] for a in apis]
    for api in api_list:
        if api not in apis:
            wrong_answer += 1
    return wrong_answer


def deduplicate_api_list(api_list: List[Any]) -> List[str]:
    """
    对API列表进行去重处理
    
    Args:
        api_list: 原始API列表，可能包含字典或字符串
        
    Returns:
        去重后的API标题列表
    """
    unique_api_doc_set = []
    seen_titles = set()
    for obj in api_list:
        if isinstance(obj, dict):
            if 'title' in obj:
                title = obj['title']
                if title not in seen_titles:
                    seen_titles.add(title)
                    unique_api_doc_set.append(title)
        elif isinstance(obj, str):
            if obj not in seen_titles:
                seen_titles.add(obj)
                unique_api_doc_set.append(obj)
    return unique_api_doc_set


def get_truth_api_set(test_item: Dict) -> Set[str]:
    """
    从测试项中获取真实API集合
    
    Args:
        test_item: 测试数据项
        
    Returns:
        真实API标题的集合
    """
    if "related_apis" not in test_item:
        return set()
    
    return {
        question["title"]
        for question in test_item["related_apis"]
        if question is not None
    }


def prepare_answer_list(test_set: List[Dict]) -> List[Dict]:
    """
    从测试集准备答案列表
    
    Args:
        test_set: 测试数据集
        
    Returns:
        答案列表
    """
    answer_list = []
    for q in test_set:
        answer = {}
        apis = []
        answer["title"] = q["title"]
        if "related_apis" in q and q["related_apis"]:
            related_apis = q["related_apis"]
            for api in related_apis:
                if isinstance(api, dict) and api is not None:
                    apis.append(api["title"])
        answer["answers"] = apis
        answer_list.append(answer)
    return answer_list