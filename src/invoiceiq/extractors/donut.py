"""Donut (Kim et al., ECCV 2022), fine-tuned on CORD: the specialist to beat.

An OCR-free encoder-decoder (Swin + BART) that reads the image and writes CORD's JSON tree
directly. It was trained on the 800 CORD training receipts, so it knows the label conventions
exactly; it needs `pip install -e ".[donut]"` and runs locally on CPU.
"""

from __future__ import annotations

import re
import threading
import time

from invoiceiq.cord import from_cord
from invoiceiq.documents import Document
from invoiceiq.extractors.base import Extraction, ExtractionError

MODEL_ID = "naver-clova-ix/donut-base-finetuned-cord-v2"
TASK_PROMPT = "<s_cord-v2>"


class DonutExtractor:
    name = "donut"

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        try:
            import torch
            from transformers import DonutProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise ExtractionError('Donut needs extra packages: pip install -e ".[donut]"') from exc
        torch.set_grad_enabled(False)
        self._processor = DonutProcessor.from_pretrained(self.model_id)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_id).eval()

    def extract(self, document: Document) -> Extraction:
        with self._lock:
            if self._model is None:
                self._load()
            processor, model = self._processor, self._model
            started = time.perf_counter()
            pixel_values = processor(document.pages[0], return_tensors="pt").pixel_values
            prompt_ids = processor.tokenizer(
                TASK_PROMPT, add_special_tokens=False, return_tensors="pt"
            ).input_ids
            tokenizer = processor.tokenizer
            output = model.generate(
                pixel_values,
                decoder_input_ids=prompt_ids,
                max_length=model.decoder.config.max_position_embeddings,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bad_words_ids=[[tokenizer.unk_token_id]],
                use_cache=True,
                num_beams=1,
            )
            sequence = processor.batch_decode(output)[0]
            seconds = time.perf_counter() - started
        sequence = sequence.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # the task start token
        tree = processor.token2json(sequence)
        if not isinstance(tree, dict):
            tree = {}
        return Extraction(
            receipt=from_cord(tree),
            model=self.model_id.split("/")[-1],
            seconds=round(seconds, 2),
            native=tree,
        )
