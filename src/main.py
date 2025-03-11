import sys
sys.path.append("/home/tiancongcong/ServiceGraphReco/src")

from trainer import APIRecommendationTrainer
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("metrics.log"),  # 输出到文件
        logging.StreamHandler()              # 输出到控制台（可选）
    ]
)


if __name__ == "__main__":
    
    # 创建训练器实例并运行训练流程
    trainer = APIRecommendationTrainer()
    evaluation_results = trainer.run_training_pipeline()
    
    # 打印评估结果
    logging.info("\n评估结果汇总:")

    logging.info("llm 召回率: %.4f", evaluation_results["recall"])
    logging.info("llm 准确率: %.4f", evaluation_results['precision'])
    logging.info("llm F1值: %.4f", evaluation_results['f1_score'])
    logging.info("llm NDCG值: %.4f", evaluation_results['ndcg'])
    logging.info("llm出现幻觉次数: %.2f%%, 概率: %.4f",  evaluation_results["hallucination_count"], evaluation_results["hallucination_rate"])
    if "execution_time" in evaluation_results:
        logging.info("函数执行耗时: %.6f 秒", evaluation_results["execution_time"])