import csv
import re
import codecs
from typing import List

from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# 自定义文档类，假设有page_content属性和metadata属性
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def load_and_split(path: str, pdf_name, max_length=100):
    # 加载pdf文档
    loader = PyPDFLoader(path)
    docs = loader.load()

    # 将文档（进一步）拆分成小段文本texts
    texts = []
    for doc in docs:
        texts += split_paragraph(doc.page_content, pdf_name, max_length)

    return texts


def split_paragraph(text, pdf_name, max_length=300):
    text = text.replace('\n', '')
    text = text.replace('\n\n', '')
    text = re.sub(r'\s+', ' ', text)

    sentences = re.split('(。|！|\!|？|\?)', text)
    new_sents = []
    for i in range(int(len(sentences) / 2)):
        sent = sentences[2 * i] + sentences[2 * i + 1]
        new_sents.append(sent)
    if len(sentences) % 2 == 1:
        new_sents.append(sentences[len(sentences) - 1])

    paragraphs = []
    current_length = 0
    current_paragraph = ""
    for sentence in new_sents:
        sentence_length = len(sentence)
        if current_length + sentence_length <= max_length:
            current_paragraph += sentence
            current_length += sentence_length
        else:
            paragraphs.append(current_paragraph.strip())
            current_paragraph = sentence
            current_length = sentence_length
    paragraphs.append(current_paragraph.strip())

    documents = []
    metadata = {"source": pdf_name}
    for paragraph in paragraphs:
        new_doc = Document(page_content=paragraph, metadata=metadata)
        documents.append(new_doc)
    return documents


def save_to_csv(texts, filename):
    with codecs.open(filename, mode='w', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        for text in texts:
            writer.writerow([text.page_content])  # 使用page_content属性

pdf_path = "data/data.pdf" # PDF文件路径
pdf_name = "data.pdf" # 可以是您希望用作PDF名称的任何字符串

texts = load_and_split(pdf_path, pdf_name) # 替换成你的文件
save_to_csv(texts, "data/data.csv") # 将文本保存到CSV文件中

def save_to_csv(texts, filename):
    with codecs.open(filename, mode='w', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        for text in texts:
            writer.writerow([text.page_content])


