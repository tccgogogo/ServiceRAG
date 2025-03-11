# evaluator.py
from typing import List, Dict, Set
from servicegraphreco.graph import normalized_dcg
from config import Config

class Evaluator:
    """
    负责评估系统性能的类
    """
    def __init__(self):
        self.config = Config()
        self.ndcg_calculator = normalized_dcg.NormalizedDCG(top_k=self.config.NDCG_TOP_K)
    
    def count_wrong_answer(self, api_list: List[str], apis: List[Dict]) -> int:
        """
        计算错误答案的数量
        """
        if len(api_list) == 0:
            return 10
        wrong_answer = 0
        api_titles = [a['title'] for a in apis]
        for api in api_list:
            if api not in api_titles:
                wrong_answer += 1
        return wrong_answer
    
    def calculate_ndcg(self, predicted_apis: List[str], truth_apis: List[str]) -> float:
        """
        计算NDCG指标
        """
        return self.ndcg_calculator.calculate_ndcg(predicted_apis, truth_apis)
    
    def calculate_metrics(self, llm_truth_num: int, all_num: int, predicted_total: int, total_ndcg: float, hallu_answer: int, answer_apis_len: int) -> Dict:
        """
        计算所有评估指标
        """
        # 计算召回率
        llm_recall = llm_truth_num / all_num if all_num > 0 else 0
        
        # 计算准确率
        llm_precision = llm_truth_num / predicted_total if predicted_total > 0 else 0
        
        # 计算幻觉率
        llm_hallu = hallu_answer / predicted_total if predicted_total > 0 else 0
        
        # 计算平均NDCG
        llm_ndcg = total_ndcg / answer_apis_len if answer_apis_len > 0 else 0
        
        # 计算F1分数
        if llm_recall + llm_precision > 0:
            llm_f1_score = 2 * (llm_recall * llm_precision) / (llm_precision + llm_recall)
        else:
            llm_f1_score = 0
        
        # 返回所有指标
        return {
            "recall": llm_recall,
            "precision": llm_precision,
            "f1_score": llm_f1_score,
            "ndcg": llm_ndcg,
            "hallucination_rate": llm_hallu,
            "hallucination_count": hallu_answer
        }
    
    def print_metrics(self, metrics: Dict) -> None:
        """
        打印评估指标
        """
        print("llm 召回率", metrics["recall"])
        print("llm 准确率：", metrics["precision"])
        print("llm f1的值", metrics["f1_score"])
        print("llm NDCG的值", metrics["ndcg"])
        print(f"llm出现幻觉的次数{metrics['hallucination_count']}, 概率{metrics['hallucination_rate']}")