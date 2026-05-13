from typing import List, Literal
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

# Markdown 标题层级
HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
MIN_CHUNK_LENGTH = 50

def split_documents(
    documents: List[Document],
    strategy: Literal["base", "section-aware", "hierarchical"] = "hierarchical",
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> List[Document]:

    # ==========================
    # 1 BASE
    # ==========================

    if strategy == "base":

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        return splitter.split_documents(documents)

    # ==========================
    # 2 SECTION ONLY
    # ==========================

    elif strategy == "section-aware":

        all_chunks = []

        for doc in documents:

            splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=HEADERS_TO_SPLIT_ON,
                strip_headers=False,
            )

            chunks = splitter.split_text(doc.page_content)

            for chunk in chunks:
                chunk.metadata.update(doc.metadata)

            all_chunks.extend(chunks)
        final_chunks = [chunk for chunk in all_chunks if len(chunk.page_content.strip()) >= MIN_CHUNK_LENGTH]

        return all_chunks

    # ==========================
    # 3 HIERARCHICAL (推荐)
    # ==========================

    elif strategy == "hierarchical":

        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )

        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        final_chunks = []

        for doc in documents:

            # Step1: section split
            sections = header_splitter.split_text(doc.page_content)

            for sec in sections:

                sec.metadata.update(doc.metadata)

                # Step2: 如果 section 太大 → recursive
                if len(sec.page_content) > chunk_size:

                    sub_chunks = recursive_splitter.split_documents([sec])

                    final_chunks.extend(sub_chunks)

                else:

                    final_chunks.append(sec)
        final_chunks = [chunk for chunk in final_chunks if len(chunk.page_content.strip()) >= MIN_CHUNK_LENGTH]

        return final_chunks

    else:

        raise ValueError(
            f"未知策略: {strategy}，可选 'base' | 'section-aware' | 'hierarchical'"
        )