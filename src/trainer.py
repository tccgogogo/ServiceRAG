import os
import json
import time
from typing import List, Dict, Tuple, Set
from tqdm import tqdm
from openai import OpenAI
import logging
import normalized_dcg
from config import *
from utils import process_json, count_wrong_answer, deduplicate_api_list, get_truth_api_set


class APIRecommendationTrainer:
    """
    API推荐系统训练和评估类
    """
    
    def __init__(self):
        """
        初始化训练器
        """
        self.log_data = {
            "last_index": 0,
            "hallu_answer": 0,
            "llm_truth_num": 0,
            "all_num": 0,
            "total_ndcg": 0.0
        }
    
    def load_datasets(self) -> Tuple[List[Dict], List[Dict]]:
        """
        加载训练集和测试集
        
        Returns:
            训练集和测试集
        """
        with open(REWRITE_TRAIN_MASHUP_PATH, "r", encoding="utf-8") as file:
            train_set = json.load(file)
        with open(REWRITE_TEST_MASHUP_PATH, "r", encoding="utf-8") as file:
            test_set = json.load(file)
        
        return train_set, test_set
    
 
    
    def load_log_data(self):
        """
        加载日志数据
        """
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                self.log_data = json.load(f)
    
    def save_log_data(self):
        """
        保存日志数据
        """
        with open(LOG_FILE, "w") as f:
            json.dump(self.log_data, f, indent=4)
    
    def call_with_messages(self, question: Dict, answer_apis: List[Dict]) -> str:
        """
        调用大模型进行API推荐
        
        Args:
            question: 问题数据
            answer_apis: 候选API列表
            
        Returns:
            大模型的响应
        """
        with open(PROMPT_COT_PATH, "r", encoding="utf-8") as file:
            prompt_origin = file.read()
        
        d = question["description"]
        c = question["categories"]
        
        mashup = json.dumps(
            {"mashup": f"description:{d}, categories:{c}", "related_apis": answer_apis}, 
            ensure_ascii=False
        )

        prompt_origin += """
            <Now,My Input Is Follow>
            {}
            """.format(mashup)
        
        messages = [
             {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {"role": "user", "content": prompt_origin}
        ]

        client = OpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
        )

        completion = client.chat.completions.create(
            model=LLM_MODEL_PATH, 
            messages=messages
        )
        
        response = completion.choices[0].message.content
        return response
    
    def evaluate_api_recommendation(self, answer_apis: List[List[Dict]], test_set: List[Dict]):
        """
        评估API推荐结果
        
        Args:
            answer_apis: 推荐的API列表
            test_set: 测试集
        """
        # 加载日志数据
        self.load_log_data()
        
        # 恢复状态
        start_index = self.log_data["last_index"]
        hallu_answer = self.log_data["hallu_answer"]
        llm_truth_num = self.log_data["llm_truth_num"]
        all_num = self.log_data["all_num"]
        total_ndcg = self.log_data["total_ndcg"]
        
        pbar = tqdm(total=len(answer_apis), desc="Processing LLM", colour="blue")
        
        # 跳过已处理的部分
        for index, api_set in enumerate(answer_apis[start_index:], start=start_index):
            pbar.update(1)
            
            predict_apis = [api for api in api_set if api is not None]
            predict_apis = predict_apis[:FINAL_API_LIST_MAX]
        
            count = 0
            max_correct_len = 0
            
            while True:
                if count > 0:
                    logging.info("开始重试了: %d", count)

                llm_str = self.call_with_messages(test_set[index], predict_apis)
                llm_res_json = process_json(llm_str)

                # 获取预测的API列表并去重
                llm_predict_apis = llm_res_json.get("related_apis", [])[:PREDICTED_APIS_TOPN]
                llm_predict_apis = deduplicate_api_list(llm_predict_apis)
                
                # 计算幻觉数量
                single_hallu = count_wrong_answer(llm_predict_apis, predict_apis)
                
                # 计算正确预测的API数量
                if "related_apis" in test_set[index]:
                    truth_api = get_truth_api_set(test_set[index])
                    
                    # 计算与真实API的交集
                    common_llm_apis = set(llm_predict_apis).intersection(truth_api)
                    if len(common_llm_apis) > max_correct_len:
                        max_correct_len = len(common_llm_apis)
                
                count += 1
                if count >= MAX_RETRY_COUNT or single_hallu < 1:
                    break
            
            hallu_answer += single_hallu
            
            # 计算评估指标
            if "related_apis" in test_set[index]:
                truth_api = get_truth_api_set(test_set[index])
                
                # 计算与真实API的交集
                common_llm_apis = set(llm_predict_apis).intersection(truth_api)

                # 更新正确预测的API数量
                if max_correct_len > 0:
                    llm_truth_num += max_correct_len
                else:
                    llm_truth_num += len(common_llm_apis)
                
                # 更新真实API总数
                all_num += len(truth_api)
                
                # 计算NDCG
                ndcg_calculator = normalized_dcg.NormalizedDCG(top_k=PREDICTED_APIS_TOPN)
                ndcg = ndcg_calculator.calculate_ndcg(llm_predict_apis, list(truth_api))
                total_ndcg += ndcg
            
            # 更新日志
            self.log_data["last_index"] = index + 1
            self.log_data["hallu_answer"] = hallu_answer
            self.log_data["llm_truth_num"] = llm_truth_num
            self.log_data["all_num"] = all_num
            self.log_data["total_ndcg"] = total_ndcg
            
            # 保存日志
            self.save_log_data()
        
        # 计算最终评估指标
        questions_count = len(test_set)
        predicted_total = questions_count * PREDICTED_APIS_TOPN
        
        llm_recall = llm_truth_num / all_num if all_num > 0 else 0
        llm_precision = llm_truth_num / predicted_total if predicted_total > 0 else 0
        llm_hallu = hallu_answer / predicted_total if predicted_total > 0 else 0
        llm_ndcg = total_ndcg / len(answer_apis)
        
        if llm_recall + llm_precision > 0:
            llm_f1_score = 2 * (llm_recall * llm_precision) / (llm_precision + llm_recall)
        else:
            llm_f1_score = 0
        
        return {
            "recall": llm_recall,
            "precision": llm_precision,
            "f1_score": llm_f1_score,
            "ndcg": llm_ndcg,
            "hallucination_count": hallu_answer,
            "hallucination_rate": llm_hallu
        }
    
    def run_training_pipeline(self):
        """
        运行完整的训练和评估流程
        
        Returns:
            评估结果
        """
        
        train_set, test_set = self.load_datasets()
        questions = [m["description"] for m in test_set]
        
        start_time = time.time()
        
        from rag import RAGSystem
        rag_system = RAGSystem()
        answer_apis = rag_system.rag_baseline(train_set, questions, question_origin=test_set)
        
        # 评估结果
        evaluation_results = self.evaluate_api_recommendation(answer_apis, test_set)
        
        # 计算总耗时
        end_time = time.time()
        elapsed_time = end_time - start_time
        evaluation_results['execution_time'] = elapsed_time
        return evaluation_results