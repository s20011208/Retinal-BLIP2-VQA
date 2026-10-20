"""BLIP-2 OPT adaptation with the upstream 4.46.3 forward and answer-only labels.

No custom prefix is inserted on the training path. Both legacy prefix and
processor-expanded image slots are checked explicitly, avoiding double image
insertion. OPT is frozen and in eval mode but is intentionally *not* under
no_grad: its Jacobian is required to train the image-to-language adapter.
"""
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn

from .answer_scoring import SCORING_VERSION, answer_log_scores, decode_generated_answers

MODEL_ID = "Salesforce/blip2-opt-2.7b"
MODEL_REVISION = "59a1ef6c1e5117b3f65523d1c6066825bcf315e3"
PROMPT_VERSION = "question_newline_answer_v1"


def format_prompt(question: str) -> str:
    return "Question: " + str(question).strip() + "\nAnswer:"


class RetinalBLIP2(nn.Module):
    def __init__(self, model: nn.Module, processor: Any, mode: str = "M1"):
        super().__init__()
        if not model.config.use_decoder_only_language_model or model.config.text_config.model_type != "opt":
            raise ValueError("This implementation is validated for decoder-only OPT BLIP-2")
        self.model = model
        self.processor = processor
        self.source_model_revision = getattr(model.config, "_commit_hash", None)
        self.mode = mode
        self.model.language_model.config.use_cache = False
        self.train_mode(mode)

    @classmethod
    def from_pretrained(cls, path: str, mode: str = "M1", dtype: Any = None,
                        device: str = "cpu", revision: Optional[str] = None,
                        local_files_only: bool = True):
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        kwargs = {"local_files_only": local_files_only}
        if revision is not None:
            kwargs["revision"] = revision
        processor = Blip2Processor.from_pretrained(str(path), **kwargs)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        model = Blip2ForConditionalGeneration.from_pretrained(
            str(path), torch_dtype=dtype or torch.float32, low_cpu_mem_usage=True, **kwargs)
        result = cls(model, processor, mode)
        result.source_model_revision = revision or result.source_model_revision
        result.to(device)
        result.promote_trainable_fp32()
        return result

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.vision_model.parameters()).dtype

    @property
    def answer_eos_id(self):
        # The official BLIP-2 OPT checkpoint stops on newline (50118), whereas
        # tokenizer.eos_token_id is </s> (2). Supervision and likelihood must use
        # the same termination rule as its language_model.generate().
        eos = self.model.language_model.generation_config.eos_token_id
        if isinstance(eos, (list, tuple)):
            if len(eos) != 1:
                raise ValueError("multiple generation EOS tokens need an explicit frozen scoring rule")
            eos = eos[0]
        if eos is None:
            raise ValueError("official language model needs an explicit EOS")
        return int(eos)

    def autocast_context(self):
        if self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16):
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    def promote_trainable_fp32(self):
        """Keep small AdamW updates representable with half precision backbones."""
        for parameter in self.model.parameters():
            if parameter.requires_grad and parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
        return self

    def train_mode(self, mode: Optional[str] = None):
        if mode is not None:
            self.mode = mode
        if self.mode not in ("B0", "B1", "M1", "B2"):
            raise ValueError("mode must be B0, B1, M1, or B2")
        for name, parameter in self.model.named_parameters():
            allowed = self.mode in ("B1", "M1", "B2") and name.startswith("language_projection.")
            allowed = allowed or (self.mode in ("M1", "B2") and
                                  (name.startswith("qformer.") or name == "query_tokens"))
            parameter.requires_grad_(allowed)
        self.promote_trainable_fp32()
        return self.train(True)

    def train(self, mode: bool = True):
        super().train(mode)
        # parent .train() must never re-enable dropout in the frozen backbones.
        self.model.vision_model.eval()
        self.model.language_model.eval()
        self.model.qformer.train(mode and self.mode in ("M1", "B2"))
        self.model.language_projection.train(mode and self.mode != "B0")
        return self

    def parameter_report(self) -> Dict[str, Any]:
        return {"mode": self.mode,
                "total_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "trainable_names": [n for n, p in self.model.named_parameters() if p.requires_grad],
                "vision_eval": not self.model.vision_model.training,
                "language_eval": not self.model.language_model.training,
                "prompt_version": PROMPT_VERSION, "scoring_version": SCORING_VERSION,
                "source_model_revision": self.source_model_revision,
                "num_query_tokens": self.model.config.num_query_tokens,
                "answer_eos_id": self.answer_eos_id,
                "tokenizer_eos_id": self.processor.tokenizer.eos_token_id,
                "image_token_index": getattr(self.model.config, "image_token_index", None)}

    def _inputs(self, image: Any, text: str):
        batch = self.processor(images=image, text=text, return_tensors="pt")
        batch = {key: value for key, value in batch.items()
                 if key in ("pixel_values", "input_ids", "attention_mask")}
        image_index = getattr(self.model.config, "image_token_index", None)
        count = int(batch["input_ids"].eq(image_index).sum()) if image_index is not None else 0
        if image_index is not None and count != self.model.config.num_query_tokens:
            raise ValueError("Official processor image slots do not equal num_query_tokens: %s" % count)
        if image_index is None and getattr(self.processor, "num_query_tokens", None) is not None:
            raise ValueError("Processor expands image slots but model has legacy visual-prefix config")
        return batch

    def build_batch(self, images: Sequence[Any], questions: Sequence[str],
                    answers: Optional[Sequence[str]] = None) -> Dict[str, torch.Tensor]:
        if len(images) != len(questions) or not images:
            raise ValueError("images and questions must be nonempty and aligned")
        if answers is not None and len(answers) != len(images):
            raise ValueError("answers must align with images")
        rows = []
        for index, (image, question) in enumerate(zip(images, questions)):
            prompt = format_prompt(question)
            prefix = self._inputs(image, prompt)
            if answers is None:
                rows.append(prefix)
                continue
            answer = str(answers[index]).strip()
            if not answer:
                raise ValueError("empty reference answer")
            row = self._inputs(image, prompt + " " + answer)
            prefix_ids = prefix["input_ids"][0]
            ids = row["input_ids"][0]
            if not torch.equal(ids[:len(prefix_ids)], prefix_ids):
                raise ValueError("Tokenizer merged across answer boundary; choose a validated prompt boundary")
            labels = ids.clone()
            labels[:len(prefix_ids)] = -100
            eos = self.answer_eos_id
            if ids[-1].item() != eos:
                ids = torch.cat((ids, torch.tensor([eos], dtype=ids.dtype)))
                labels = torch.cat((labels, torch.tensor([eos], dtype=labels.dtype)))
            row["input_ids"] = ids.unsqueeze(0)
            row["attention_mask"] = torch.ones_like(row["input_ids"])
            row["labels"] = labels.unsqueeze(0)
            rows.append(row)
        max_len = max(row["input_ids"].shape[-1] for row in rows)
        visual_extra = self.model.config.num_query_tokens if getattr(self.model.config, "image_token_index", None) is None else 0
        if max_len + visual_extra > self.model.config.text_config.max_position_embeddings:
            raise ValueError("input exceeds OPT context; silent answer truncation is forbidden")
        pad = self.processor.tokenizer.pad_token_id
        if pad is None:
            raise ValueError("tokenizer must have an explicit padding token")
        result = {"input_ids": torch.full((len(rows), max_len), pad, dtype=torch.long),
                  "attention_mask": torch.zeros((len(rows), max_len), dtype=torch.long),
                  "pixel_values": torch.cat([row["pixel_values"] for row in rows])}
        if answers is not None:
            result["labels"] = torch.full((len(rows), max_len), -100, dtype=torch.long)
        for index, row in enumerate(rows):
            length = row["input_ids"].shape[-1]
            result["input_ids"][index, :length] = row["input_ids"][0]
            result["attention_mask"][index, :length] = 1
            if answers is not None:
                result["labels"][index, :length] = row["labels"][0]
        return {key: value.to(self.device, dtype=self.dtype if key == "pixel_values" else value.dtype)
                for key, value in result.items()}

    def answer_forward(self, images, questions, answers):
        batch = self.build_batch(images, questions, answers)
        with self.autocast_context():
            output = self.model(**batch, return_dict=True)
        # The upstream function returns text-aligned logits when labels exist.
        scores = answer_log_scores(output.logits, batch["labels"])
        tokens = batch["labels"][:, 1:].ne(-100).sum()
        loss = -scores.sum() / tokens
        return {"loss": loss, "official_loss": output.loss, "logits": output.logits,
                "labels": batch["labels"], "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"], "scores": scores}

    def loss(self, images, questions, answers):
        return self.answer_forward(images, questions, answers)["loss"]

    def forward(self, images, questions, answers):
        return self.loss(images, questions, answers)

    @torch.no_grad()
    def score_candidates(self, images, questions, candidates, reduction: str = "sum"):
        if len(images) != len(questions) or len(candidates) != len(images):
            raise ValueError("candidate rows must align with images and questions")
        was_training = self.training
        self.eval()
        result = []
        try:
            for image, question, row_candidates in zip(images, questions, candidates):
                if not row_candidates:
                    raise ValueError("candidate set cannot be empty")
                # Sequential candidates avoid multiplying memory by answer count.
                row_scores = []
                for candidate in row_candidates:
                    output = self.answer_forward([image], [question], [candidate])
                    row_scores.append(answer_log_scores(output["logits"], output["labels"], reduction)[0].item())
                result.append(row_scores)
        finally:
            self.train(was_training)
        return result

    @torch.no_grad()
    def generate_answers(self, images, questions, max_new_tokens: int = 16):
        if len(images) != len(questions):
            raise ValueError("images and questions must align")
        was_training = self.training
        self.eval()
        result = []
        try:
            # Single-image inference avoids any decoder right-padding ambiguity.
            for image, question in zip(images, questions):
                batch = self.build_batch([image], [question])
                with self.autocast_context():
                    ids = self.model.generate(**batch, do_sample=False, num_beams=1,
                                              max_new_tokens=max_new_tokens)
                # Official 4.46.3 passes only inputs_embeds into OPT.generate;
                # its starting IDs have length zero, and BLIP-2 adds one BOS.
                if ids.shape[1] > max_new_tokens + 1:
                    raise ValueError("upstream generation return protocol changed: unexpected prompt positions")
                result.append(decode_generated_answers(self.processor, ids, includes_prompt=False)[0])
        finally:
            self.train(was_training)
        return result

    @torch.no_grad()
    def vision_features(self, images):
        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
        self.model.vision_model.eval()
        with self.autocast_context():
            return self.model.vision_model(pixel_values=pixels, return_dict=True).last_hidden_state

    def manual_forward_logits(self, batch):
        """Diagnostic implementation only; compare with official forward before use."""
        with torch.no_grad():
            visual = self.model.vision_model(pixel_values=batch["pixel_values"], return_dict=True).last_hidden_state
        query = self.model.qformer(query_embeds=self.model.query_tokens.expand(visual.shape[0], -1, -1),
                                  encoder_hidden_states=visual,
                                  encoder_attention_mask=torch.ones(visual.shape[:2], device=visual.device, dtype=torch.long),
                                  return_dict=True).last_hidden_state
        prefix = self.model.language_projection(query)
        embedded = self.model.language_model.get_input_embeddings()(batch["input_ids"])
        mask = batch["attention_mask"]
        image_index = getattr(self.model.config, "image_token_index", None)
        if image_index is not None:
            slots = batch["input_ids"].eq(image_index).unsqueeze(-1).expand_as(embedded)
            embedded = embedded.masked_scatter(slots, prefix.to(embedded.dtype))
        else:
            embedded = torch.cat((prefix, embedded), 1)
            mask = torch.cat((torch.ones(prefix.shape[:2], device=mask.device, dtype=mask.dtype), mask), 1)
        # Intentionally graph-enabled through the frozen OPT.
        return self.model.language_model(inputs_embeds=embedded, attention_mask=mask, return_dict=True).logits

    def adapter_state_dict(self):
        # Persist all adapter modules, including unchanged Q-Former in B1, so
        # each payload can be audited and loaded without a partial-module guess.
        return {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()
                if name == "query_tokens" or name.startswith(("qformer.", "language_projection."))}

    def load_adapter_state_dict(self, state):
        expected = {name for name in self.model.state_dict()
                    if name == "query_tokens" or name.startswith(("qformer.", "language_projection."))}
        if set(state) != expected:
            raise ValueError("adapter state keys mismatch: missing=%s extra=%s" %
                             (sorted(expected - set(state)), sorted(set(state) - expected)))
        self.model.load_state_dict(state, strict=False)

    def save_adapter(self, path, metadata=None):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        torch.save({"format_version": 1, "adapter": self.adapter_state_dict(),
                    "metadata": metadata or {}, "model_revision": self.source_model_revision,
                    "parameter_report": self.parameter_report()}, temporary)
        temporary.replace(target)
