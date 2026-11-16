from __future__ import annotations

from typing import Any, Dict, List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "google/flan-t5-small"


class AnswerGenerator:
    """Ground answers in retrieved evidence and abstain when the paper lacks support."""

    def __init__(self, model_name: str = MODEL_NAME):
        print("Loading local answer-generation model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("Answer-generation model loaded.")

    def _summarize_evidence(self, evidence: List[Dict[str, Any]]) -> str:
        chunks: List[str] = []
        for item in evidence:
            chunk = str(item.get("chunk", "")).strip()
            if chunk:
                section = item.get("section", "Unknown section")
                chunks.append(f"[{section}] {chunk}")
        return "\n\n".join(chunks)

    def _fallback_answer(self, question: str, evidence: List[Dict[str, Any]]) -> str:
        summary = self._summarize_evidence(evidence)
        if not summary:
            return "The paper does not provide enough information to answer this question."

        lines = [line.strip() for line in summary.splitlines() if line.strip()]
        evidence_excerpt = "\n\n".join(lines[:3])
        return (
            "Based on the evidence in this paper, the answer is limited to the provided passages. "
            "The most relevant supporting context is: \n\n"
            f"{evidence_excerpt}"
        )

    def generate_answer(self, question: str, evidence: List[Dict[str, Any]]) -> str:
        cleaned_question = (question or "").strip()
        if not cleaned_question:
            return "Please provide a question to answer."

        if not evidence:
            return "The paper does not provide enough information to answer this question."

        evidence_text = self._summarize_evidence(evidence)
        if not evidence_text.strip():
            return "The paper does not provide enough information to answer this question."

        prompt = f"""
Answer the question using ONLY the evidence provided below.
Do not use outside knowledge. Do not guess.
If the evidence is insufficient, respond exactly:
The paper does not provide enough information.

Question:
{cleaned_question}

Evidence:
{evidence_text}

Answer:
"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.model.generate(**inputs, max_new_tokens=120, do_sample=False)
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception:
            return self._fallback_answer(cleaned_question, evidence)

        if not answer:
            return "The paper does not provide enough information to answer this question."

        normalized = answer.strip()
        if normalized.lower() in {"the paper does not provide enough information.", "the paper does not provide enough information"}:
            return "The paper does not provide enough information to answer this question."

        return normalized