# 配置文件，存储所有常量

# 环境配置
CUDA_VISIBLE_DEVICES = "1"

# 文件路径
REWRITE_TRAIN_MASHUP_PATH = "data/rewrite/seed42/train_rewrite_data_42.json"
REWRITE_TEST_MASHUP_PATH = "data/rewrite/seed42/test_rewrite_data_42.json"

REWRITE_TRAIN_MASHUP_PATH_0311 = "data/rewrite/seed42/train_rewrite_data_42_0311.json"

ORIGIN_TRAIN_MASHUP_PATH = "data/origin/train_data.json"
ORIGIN_TEST_MASHUP_PATH = "data/origin/test_data.json"

LOG_FILE = "./qwen-mashup90-top10-0203.json"
PROMPT_COT_PATH = "src/prompt/prompt_cot.txt"
PROMPT_REWRITE_APIS_PATH = "src/prompt/rewrite_apis.txt"
PROMPT_REWRITE_MASHUPS_PATH = "src/prompt/rewrite_mashups.txt"


# 模型路径
EMBED_MODEL_PATH = "model/all-MiniLM-L6-v2"
RERANK_MODEL_PATH = "model/bge-reranker-v2-m3"
LLM_MODEL_PATH = "model/Qwen2.5-14B-Instruct"

# 向量数据库配置
VECTORDB_DIR = "./servicegraphreco/graph/FAISS_DB"

# 检索配置
RETRIEVER_K = 45
BM25_K1 = 1.5
BM25_B = 0.75
RETRIEVER_WEIGHTS = [0.5, 0.5]  # BM25和向量检索的权重
TOP_N_RERANK = 90

# API处理配置
ORDERED_APIS_COUNT = 40
TOP_APIS_BY_SIMILARITY = 10
FINAL_API_LIST_MAX = 50

# OpenAI API配置
OPENAI_BASE_URL = "http://localhost:8002/v1"
OPENAI_API_KEY = "xxxx"

# 评估配置
MAX_RETRY_COUNT = 20
PREDICTED_APIS_TOPN = 10