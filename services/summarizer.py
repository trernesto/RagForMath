from transformers import pipeline
from services.file_processor import vector_store

# Глобальная модель для суммаризации
_summarizer = None

def generate_summary(doc_id, max_length=150):
    """Генерация краткого содержания"""
    global _summarizer
    
    # Инициализация модели
    if _summarizer is None:
        _summarizer = pipeline("summarization", model="Falconsai/text_summarization")
    
    # Получение текста документа
    if vector_store is None:
        raise Exception("Документ не загружен")
    
    # Поиск всех чанков документа
    docs = vector_store.search("", search_type="similarity", filter={"doc_id": doc_id})
    full_text = " ".join(doc.page_content for doc in docs)
    
    # Суммаризация
    summary = _summarizer(
        full_text, 
        max_length=max_length, 
        min_length=30,
        do_sample=False
    )[0]['summary_text']
    
    return summary