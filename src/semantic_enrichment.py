import json
from config import *
from openai import OpenAI
import json_repair
import re
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Union
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rewrite.log"),  # 输出到文件
        logging.StreamHandler()              # 输出到控制台（可选）
    ]
)


class SemanticEnrichment:
    def __init__(self, mode: str = 'local'):
        """初始化语义增强类
        
        Args:
            mode: 运行模式，'local':本地大模型
        """
        self.mode = mode
        self.client = None
        if mode == 'local':
            self.client = OpenAI(
                base_url=OPENAI_BASE_URL,
                api_key=OPENAI_API_KEY,
            )
        else:
            raise ValueError(f"不支持的模式: {mode}")

    def _process_rewrite_json(self, text: str, type_name: str) -> Dict[str, Any]:
        """处理LLM返回的JSON文本
        
        Args:
            text: LLM返回的文本
            type_name: 处理类型，'api'或'mashup'
            
        Returns:
            处理后的JSON数据
        """
        try:
            pattern = r"```json(.*?)```"
            matches = re.findall(pattern, text, re.DOTALL)
            json_objects = [json.loads(match) for match in matches]
            
            for match in json_objects:
                try:
                    description = match.get("enhanced_description", "")
                    if not description:
                        continue
                        
                    if type_name == "api":
                        tags = match.get("tags", [])
                        if not tags:
                            continue
                        return {"description": description, "tags": tags}
                        
                    elif type_name == "mashup":
                        categories = match.get("categories", [])
                        if not categories:
                            continue
                        return {"description": description, "categories": categories}
                        
                except Exception as e:
                    logging.error(f"处理JSON对象失败: {e}")
                    
        except Exception as e:
            logging.error(f"提取JSON失败: {e}")
            
        return {}
        
    def _call_llm(self, messages: List[Dict[str, str]], retry_limit: int = 8) -> Dict[str, Any]:
        """调用LLM并处理响应
        
        Args:
            messages: 发送给LLM的消息列表
            retry_limit: 重试次数限制
            
        Returns:
            处理后的JSON数据
        """
        if not self.client:
            logging.error("LLM客户端未初始化")
            return {}
            
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL_PATH,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"调用LLM失败: {e}")
            return ""
        

    def rewrite_api(self, api: Dict[str, Any]) -> Dict[str, Any]:
        """改写API的描述和标签
        
        Args:
            api: API数据
            
        Returns:
            改写后的API数据
        """
        # 准备API数据
        api_str = json.dumps({
            "title": api.get("title", ""),
            "description": api.get("description", ""),
            "tags": api.get("tags", [])
        }, ensure_ascii=False)
        
        # 读取提示词
        try:
            with open(PROMPT_REWRITE_APIS_PATH, "r", encoding="utf-8") as file:
                api_prompt = file.read()
        except Exception as e:
            logging.error(f"读取API提示词文件失败: {e}")
            return {}
        
        # 构建完整提示词
        api_prompt += '''
<Now,My Input Is Follow>
{}
'''.format(api_str)
        
        # 构建消息
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": api_prompt}
        ]
        
        # 调用LLM并处理响应
        retry_count = 0
        rewrite_json = {}
        
        while not rewrite_json and retry_count < 8:
            if retry_count > 0:
                logging.info(f"API改写重试 {retry_count}")
                
            llm_response = self._call_llm(messages)
            if not llm_response:
                retry_count += 1
                continue
                
            rewrite_json = self._process_rewrite_json(llm_response, 'api')
            retry_count += 1
            
        return rewrite_json


    def rewrite_mashup(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """改写Mashup的描述和分类
        
        Args:
            question: Mashup数据
            
        Returns:
            改写后的Mashup数据
        """
        # 准备Mashup数据
        mashup_str = json.dumps({
            "title": question.get("title", ""),
            "description": question.get("description", ""),
            "tags": question.get("tags", [])
        }, ensure_ascii=False)
        
        # 读取提示词
        try:
            with open(PROMPT_REWRITE_MASHUPS_PATH, "r", encoding="utf-8") as file:
                mashup_prompt = file.read()
        except Exception as e:
            logging.error(f"读取Mashup提示词文件失败: {e}")
            return {}
        
        mashup_prompt += '''
<Now,My Input Is Follow>
{}
'''.format(mashup_str)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": mashup_prompt}
        ]
        
        retry_count = 0
        rewrite_json = {}
        
        while not rewrite_json and retry_count < 8:
            if retry_count > 0:
                logging.info(f"Mashup改写重试 {retry_count}")
                
            llm_response = self._call_llm(messages)
            if not llm_response:
                retry_count += 1
                continue
                
            rewrite_json = self._process_rewrite_json(llm_response, 'mashup')
            retry_count += 1
            
        return rewrite_json


    def rewrite_dataset(self, questions_origin: List[Dict[str, Any]], output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """改写整个数据集
        
        Args:
            questions_origin: 原始数据集
            output_path: 输出文件路径，默认为None，不保存文件
            
        Returns:
            改写后的数据集
        """
        pbar = tqdm(total=len(questions_origin), desc="改写问题", colour="blue")
        questions_rewrite = []
        
        for index, question in enumerate(questions_origin):
            pbar.update(1)
            
            # 改写Mashup
            mashup_json = self.rewrite_mashup(question)
            question['description'] = mashup_json.get("description", question['description'])
            question['categories'] = mashup_json.get("categories", question['categories'])
            
            # 改写相关API
            if 'related_apis' in question:
                for api in question['related_apis']:
                    if not api:
                        continue
                    api_json = self.rewrite_api(api)
                    api['description'] = api_json.get("description", api['description'])
                    api['tags'] = api_json.get("tags", api['tags'])
            
            questions_rewrite.append(question)
        
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(questions_rewrite, f, ensure_ascii=False, indent=4)
                logging.info(f"已将改写结果保存至 {output_path}")
            except Exception as e:
                logging.error(f"保存改写结果失败: {e}")
        
        return questions_rewrite       

def main():
    try:
        with open(ORIGIN_TRAIN_MASHUP_PATH, "r", encoding="utf-8") as file:
            train_set = json.load(file)[:10]
        with open(ORIGIN_TEST_MASHUP_PATH, "r", encoding="utf-8") as file:
            test_set = json.load(file)
            
        # 初始化语义增强器
        enricher = SemanticEnrichment(mode='local')
        
        # 改写训练集
        logging.info("开始改写训练集...")
        train_rewrite = enricher.rewrite_dataset(train_set, REWRITE_TRAIN_MASHUP_PATH_0311)
        logging.info(f"训练集改写完成，共 {len(train_rewrite)} 条数据")
        
        # 改写测试集
        # logger.info("开始改写测试集...")
        # test_rewrite = enricher.rewrite_dataset(test_set, REWRITE_TEST_MASHUP_PATH)
        # logger.info(f"测试集改写完成，共 {len(test_rewrite)} 条数据")
        
    except Exception as e:
        logging.error(f"程序执行出错: {e}")


if __name__ == "__main__":
    main()