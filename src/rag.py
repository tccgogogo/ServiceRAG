import os
import json
from typing import List, Dict, Tuple, Any
import time
from tqdm import tqdm
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import jieba

from config import *


class RAGSystem:
    """
    RAG系统类，用于实现基于检索增强生成的API推荐系统
    """
    
    def __init__(self):
        """
        初始化RAG系统
        """
        # 设置环境变量
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
        
        # 初始化模型和检索器
        self.api_embed_model = None
        self.vector_retriever = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.compression_retriever = None
    
    def prepare_documents(self, mashups: List[Dict]) -> List[Document]:
        """
        准备文档集合
        
        Args:
            mashups: Mashup数据列表
            
        Returns:
            文档列表
        """
        docs = []
        for mashup in mashups:
            d = mashup["description"]
            content = d
            doc = Document(page_content=content, metadata={"title": mashup["title"]})
            docs.append(doc)
        return docs
    
    def setup_retrievers(self, docs: List[Document]):
        """
        设置检索器
        
        Args:
            docs: 文档列表
        """
        # 设置向量检索器
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_PATH,
            show_progress=True,
            model_kwargs={
                "trust_remote_code": True,
            },
        )
        db = FAISS.from_documents(documents=docs, embedding=embeddings)
        db.save_local(VECTORDB_DIR)
        self.vector_retriever = db.as_retriever(search_kwargs={"k": RETRIEVER_K})
        
        # 设置BM25检索器
        self.bm25_retriever = BM25Retriever.from_documents(
            docs,
            k=RETRIEVER_K,
            bm25_params={"k1": BM25_K1, "b": BM25_B},
            preprocess_func=jieba.lcut,
        )
        
        # 设置集成检索器
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever], 
            weights=RETRIEVER_WEIGHTS
        )
        
        # 设置重排序检索器
        rerank_model = HuggingFaceCrossEncoder(model_name=RERANK_MODEL_PATH)
        compressor = CrossEncoderReranker(model=rerank_model, top_n=TOP_N_RERANK)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=self.ensemble_retriever
        )
        
        # 初始化嵌入模型
        self.api_embed_model = SentenceTransformer(EMBED_MODEL_PATH)
    
    def get_topn_mashup_api(self, rerank_answers: List[List[str]], mashups: List[Dict], top_n: int = FINAL_API_LIST_MAX) -> List[List[Dict]]:
        """
        获取每个Mashup的Top-N API
        
        Args:
            rerank_answers: 重排序后的Mashup答案列表
            mashups: 原始Mashup数据
            top_n: 返回的最大API数量
            
        Returns:
            每个Mashup的Top-N API列表
        """
        api_docs = []
        for index, mashup_set in enumerate(rerank_answers):
            api_doc_set = []
            for mashup in mashup_set:
                for origin_mashup in mashups:
                    if origin_mashup["title"] == mashup:
                        if (
                            "related_apis" in origin_mashup
                            and origin_mashup["related_apis"]
                        ):
                            related_apis = origin_mashup["related_apis"]

                            for api in related_apis:
                                if (
                                    api is not None
                                    and isinstance(api, dict)
                                    and "title" in api
                                    and "tags" in api
                                ):
                                    api_json = {
                                        "title": api["title"],
                                        "tags": api["tags"],
                                    }
                                    api_doc_set.append(api_json)
            # Step 1: 统计每个 title 出现的次数
            title_counts = Counter(obj["title"] for obj in api_doc_set)
            # Step 2: 根据 title 去重，保留第一个出现的对象
            unique_objects = {}
            for obj in api_doc_set:
                if obj["title"] not in unique_objects:
                    unique_objects[obj["title"]] = obj
            # Step 3: 根据重复次数从大到小排序
            sorted_objects = sorted(
                unique_objects.values(),
                key=lambda x: title_counts[x["title"]],
                reverse=True,
            )
            api_docs.append(sorted_objects)
        return api_docs
    
    def rerank_apis_by_similarity(self, top_n_mashup_apis: List[List[Dict]], question_origin: List[Dict]) -> List[List[Dict]]:
        """
        根据相似度对API进行重排序
        
        Args:
            top_n_mashup_apis: 每个Mashup的Top-N API列表
            question_origin: 原始问题数据
            
        Returns:
            重排序后的API列表
        """
        rerank_answers = []
        pbar = tqdm(total=len(top_n_mashup_apis), desc="Processing LLM", colour="blue")
        
        for index, api_doc_set in enumerate(top_n_mashup_apis):
            pbar.update(1)
            if len(api_doc_set) > 0:
                rerank_apis = []  # 存储推荐的API

                # 获取当前Mashup的标签
                cat_list = question_origin[index]["categories"]
                cat_list = ", ".join(cat_list)
                tag_list = question_origin[index]["tags"]
                cat_list = cat_list + ", ".join(tag_list)
                # 对Mashup标签生成嵌入
                mashup_embedding = self.api_embed_model.encode(cat_list)

                # 对每个API文档集进行嵌入生成
                for api_doc in api_doc_set:
                    api_tags = api_doc["tags"]
                    api_tags = ", ".join(api_tags)

                    api_embedding = self.api_embed_model.encode(api_tags)

                    similarity_score = cosine_similarity(
                        [mashup_embedding], [api_embedding]
                    )[0][0]
                    rerank_apis.append((api_doc, similarity_score))

                ordered_apis = api_doc_set[:ORDERED_APIS_COUNT]
                sorted_apis = sorted(rerank_apis, key=lambda x: x[1], reverse=True)
                top_10_apis_by_similarity = [
                    api[0] for api in sorted_apis if api[0] not in ordered_apis
                ]
                top_10_apis_by_similarity = top_10_apis_by_similarity[:TOP_APIS_BY_SIMILARITY]
                final_api_list = ordered_apis + top_10_apis_by_similarity
                final_api_list = final_api_list[:FINAL_API_LIST_MAX]
                rerank_answers.append(final_api_list)
            else:
                rerank_apis = []
                rerank_answers.append(rerank_apis)
        return rerank_answers
    
    def rag_baseline(self, mashups: List[Dict], questions: List[str], question_origin: List[Dict]) -> Tuple[List[List[Dict]], int]:
        """
        RAG基线方法
        
        Args:
            mashups: Mashup数据列表
            questions: 问题列表
            question_origin: 原始问题数据
            
        Returns:
            推荐的API列表和总字符数
        """
        docs = self.prepare_documents(mashups)
        
        # 设置检索器
        self.setup_retrievers(docs)

        # 对每个问题进行检索
        rerank_answers = []
        for question in tqdm(question_origin):
            d = question["description"]
            content = d
            relevant_docs = self.compression_retriever.invoke(content)

            rerank_mashups = []
            for rd in relevant_docs:
                rerank_mashups.append(rd.metadata["title"])
            rerank_answers.append(rerank_mashups)
        
        top_n_mashup_apis = self.get_topn_mashup_api(rerank_answers, mashups)
        rerank_answers = self.rerank_apis_by_similarity(top_n_mashup_apis, question_origin)
        
        return rerank_answers