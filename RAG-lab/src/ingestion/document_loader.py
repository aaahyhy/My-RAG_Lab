import os
import re
import gc
from typing import List
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

# --- 1. 全局实例化一个配置好的转换器，避免循环初始化 ---
def get_optimized_converter():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False  # 禁用 OCR
    pipeline_options.do_table_structure = False  # 如果不需要精准表格，建议禁用
    # pipeline_options.do_layout = False # 建议保留，否则双栏解析会乱


    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
# 初始化全局变量
GLOBAL_CONVERTER = get_optimized_converter()

def remove_references_section(markdown_text: str) -> str:
    ref_pattern = r"(?i)^#{0,6}\s*(references|bibliography|works cited|literature cited)\s*$"
    match = re.search(ref_pattern, markdown_text, re.IGNORECASE | re.MULTILINE)
    if match:
        return markdown_text[:match.start()]
    return markdown_text

def load_single_pdf(file_path: str) -> Document:
    # --- 2. 使用全局转换器，不要在这里重新生成 ---
    result = GLOBAL_CONVERTER.convert(file_path)
    markdown_text = result.document.export_to_markdown()
    cleaned_text = remove_references_section(markdown_text)

    return Document(
        page_content=cleaned_text,
        metadata={
            "source": os.path.basename(file_path),
            "type": "academic_paper",
        }
    )

def load_documents(data_dir: str) -> List[Document]:
    all_docs = []
    # 确保路径存在
    if not os.path.exists(data_dir):
        print(f"Error: 路径 {data_dir} 不存在")
        return []

    for filename in os.listdir(data_dir):
        if not filename.lower().endswith(".pdf"):
            continue
        file_path = os.path.join(data_dir, filename)
        try:
            doc = load_single_pdf(file_path)
            all_docs.append(doc)
            print(f"✅ 成功解析: {filename}")
            # 手动触发垃圾回收，清理内存
            gc.collect()
        except Exception as e:
            print(f"❌ 解析失败 {filename}: {e}")

    return all_docs
