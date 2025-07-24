from transformers import pipeline
from services.file_processor import vector_store

# Глобальная модель для QA
_qa_model = None

def ask_question(question, doc_id):
    """Ответ на вопрос по документу"""
    global _qa_model
    
    # Инициализация модели
    if _qa_model is None:
        _qa_model = pipeline("question-answering")
    
    # Поиск релевантных чанков
    if vector_store is None:
        raise Exception("Документ не загружен")
    
    # Поиск с фильтрацией по документу
    docs = vector_store.similarity_search(
        query=question, 
        k=3,
        filter={"doc_id": doc_id}
    )
    
    # Объединение контекста
    context = " ".join(doc.page_content for doc in docs)
    
    # Генерация ответа
    result = _qa_model(question=question, context=context)
    return result['answer']