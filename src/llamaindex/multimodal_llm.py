from typing import Any, List, Optional, Sequence
from llama_index.core.multi_modal_llms import MultiModalLLM, MultiModalLLMMetadata
from llama_index.core.base.llms.types import (
    CompletionResponse, CompletionResponseAsyncGen, CompletionResponseGen,
    ChatResponse, ChatResponseGen, ChatResponseAsyncGen, ChatMessage, MessageRole
)
from llama_index.core.schema import ImageDocument
from src.vlm.client import GroqVLMClient, clean_thinking_trace

class GroqLlamaMultiModalLLM(MultiModalLLM):
    """LlamaIndex MultiModalLLM adapter for Groq VLM (qwen/qwen3.6-27b)."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._vlm_client = GroqVLMClient(api_key=api_key, model=model)

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        return MultiModalLLMMetadata(
            model_name=self._vlm_client.vlm_model,
            is_multimodal=True,
            max_input_tokens=4096,
        )

    def complete(
        self, prompt: str, image_documents: Sequence[ImageDocument] = (), **kwargs: Any
    ) -> CompletionResponse:
        """Execute multimodal completion using Groq VLM SDK."""
        if image_documents and hasattr(image_documents[0], "image_url") and image_documents[0].image_url:
            storyboard_b64 = image_documents[0].image_url
            shot_info = image_documents[0].extra_info if hasattr(image_documents[0], "extra_info") else {}
            raw_json = self._vlm_client.analyze_storyboard(storyboard_b64, shot_info)
            facts = []
            for ev in raw_json.get("events", []):
                facts.append(f"[{ev.get('start_time')}-{ev.get('end_time')}] {ev.get('description')}")
            context_facts = "\n".join(facts)
            raw_text = self._vlm_client.generate_text_answer(prompt, context_facts)
        else:
            raw_text = self._vlm_client.generate_text_answer(prompt, prompt)

        cleaned_text = clean_thinking_trace(raw_text)
        return CompletionResponse(text=cleaned_text)

    def stream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument] = (), **kwargs: Any
    ) -> CompletionResponseGen:
        res = self.complete(prompt, image_documents, **kwargs)
        yield res

    async def acomplete(
        self, prompt: str, image_documents: Sequence[ImageDocument] = (), **kwargs: Any
    ) -> CompletionResponse:
        return self.complete(prompt, image_documents, **kwargs)

    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument] = (), **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        res = self.complete(prompt, image_documents, **kwargs)
        yield res

    def chat(
        self, messages: Sequence[ChatMessage], image_documents: Sequence[ImageDocument] = (), **kwargs: Any
    ) -> ChatResponse:
        prompt_text = "\n".join([m.content for m in messages if m.content])
        comp = self.complete(prompt_text, image_documents, **kwargs)
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=comp.text))

    def stream_chat(
        self, messages: Sequence[ChatMessage], image_documents: Sequence[ImageDocument] = (), **kwargs: Any
    ) -> ChatResponseGen:
        resp = self.chat(messages, image_documents, **kwargs)
        yield resp

    async def achat(
        self, messages: Sequence[ChatMessage], image_documents: Sequence[ImageDocument] = (), **kwargs: Any
    ) -> ChatResponse:
        return self.chat(messages, image_documents, **kwargs)

    async def astream_chat(
        self, messages: Sequence[ChatMessage], image_documents: Sequence[ImageDocument] = (), **kwargs: Any
    ) -> ChatResponseAsyncGen:
        resp = self.chat(messages, image_documents, **kwargs)
        yield resp
