from transformers import pipeline
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM

class RAG():
    def __init__(self):
        RAG_MODEL_NAME = "RefalMachine/RuadaptQwen3-4B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(
            RAG_MODEL_NAME,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(RAG_MODEL_NAME)
        
        prompt_in_chat_format = [
            {
                "role": "system",
                "content": """Используя информацию, содержащуюся в контексте,
                        дайте развернутый ответ на вопрос.
                        Отвечайте только на заданный вопрос, ответ должен быть кратким и соответствовать теме.
                        Укажите номер исходного документа, если это необходимо.
                        Если ответ не может быть выведен из контекста, не давайте ответ.""",
            },
            {
                "role": "user",
                "content": """Context:
        {context}
        ---
        Вот вопрос на который ты должен дать ответ.

        Вопрос: {question}""",
            },
        ]

        
        self.RAG_PROMPT_TEMPLATE = self.tokenizer.apply_chat_template(
            prompt_in_chat_format, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def ask_question(self, question, vector_store, doc_ids = None):
        print('Started working with question')
        if vector_store is None:
            raise Exception("Документ не загружен")
        
        if doc_ids is not None:
            docs = vector_store.similarity_search(
                query=question, 
                k=3,
                filter={"doc_id": doc_ids}
            )
        else:
            docs = vector_store.similarity_search(
                query=question, 
                k=5
            )
            
        context = " ".join(doc.page_content for doc in docs)
        final_prompt = self.RAG_PROMPT_TEMPLATE.format(question=question, context=context)
        model_inputs = self.tokenizer([final_prompt], return_tensors="pt")

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=2048
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
        answer = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        print('Answer:', answer)
        return answer