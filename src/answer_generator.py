from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "google/flan-t5-small"


class AnswerGenerator:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME
        )

    def generate_answer(self, question, evidence):

        if not evidence:
            return (
                "I could not find relevant evidence "
                "in the uploaded paper."
            )

        evidence_text = "\n\n".join(
            item["chunk"]
            for item in evidence
        )

        prompt = f"""
Answer the question using ONLY the evidence below.

If the evidence does not contain the answer, say:
The paper does not provide enough information.

Question:
{question}

Evidence:
{evidence_text}

Answer:


        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        return answer.strip()