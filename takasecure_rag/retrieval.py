from dataclasses import dataclass

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from .config import Settings


@dataclass
class RetrieverBundle:
    direct: ContextualCompressionRetriever
    multi_query: ContextualCompressionRetriever


def build_retrievers(settings: Settings, llm) -> RetrieverBundle:
    pages = PyPDFLoader(str(settings.policy_pdf), mode="page").load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=180,
        add_start_index=True,
    )
    documents = splitter.split_documents(pages)

    dense = HuggingFaceEmbeddings(
        model_name=settings.dense_embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
    sparse = FastEmbedSparse(model_name=settings.sparse_embedding_model)
    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=dense,
        sparse_embedding=sparse,
        location=settings.qdrant_location,
        collection_name=settings.qdrant_collection,
        retrieval_mode=RetrievalMode.HYBRID,
    )
    hybrid = vector_store.as_retriever(
        search_kwargs={"k": settings.retrieval_k},
    )

    reranker_model = HuggingFaceCrossEncoder(model_name=settings.reranker_model)
    reranker = CrossEncoderReranker(
        model=reranker_model,
        top_n=settings.rerank_top_n,
    )
    direct = ContextualCompressionRetriever(
        base_retriever=hybrid,
        base_compressor=reranker,
    )
    model_rewriting = MultiQueryRetriever.from_llm(retriever=hybrid, llm=llm)
    multi_query = ContextualCompressionRetriever(
        base_retriever=model_rewriting,
        base_compressor=reranker,
    )
    return RetrieverBundle(direct=direct, multi_query=multi_query)
