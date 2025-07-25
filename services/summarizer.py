from transformers import pipeline
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, T5ForConditionalGeneration


class Summarizer():
    def __init__(self):
        READER_MODEL_NAME = "IlyaGusev/rut5_base_sum_gazeta"

        self.model = T5ForConditionalGeneration.from_pretrained(READER_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
        

    def generate_summary(self, doc_ids: list, vector_store: FAISS, max_length=150):
        if vector_store is None:
            raise Exception("Документ не загружен")
        
        docs = vector_store.get_by_ids(doc_ids.split(','))
        context = " ".join(doc.page_content for doc in docs)

        input_ids = self.tokenizer(
                [context],
                max_length=4_800,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )["input_ids"]
        
        output_ids = self.model.generate(
            input_ids=input_ids,
            no_repeat_ngram_size=4
        )[0]
        summary = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        print(summary)
        
        return summary