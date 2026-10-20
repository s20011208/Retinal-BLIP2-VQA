"""Tiny random architectures and synthetic mechanism fixtures, never research results."""
import numpy as np
from PIL import Image
import torch
from retinal_blip2.models.blip2_adapter import RetinalBLIP2

def tiny_adapter(mode="M1", expanded=True):
    """Actual HF BLIP-2 architecture with tiny, untrained random weights."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from tokenizers.processors import TemplateProcessing
    from transformers import (Blip2Config, Blip2ForConditionalGeneration,
                              Blip2Processor, BlipImageProcessor, PreTrainedTokenizerFast)
    vocab = {"<pad>": 0, "<unk>": 1, "</s>": 2, "<image>": 3,
             "Question:": 4, "Answer:": 5, "Are": 6, "there": 7,
             "exudates?": 8, "yes": 9, "no": 10, "What": 11,
             "is": 12, "grade?": 13, "0": 14, "1": 15, "2": 16}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.post_processor = TemplateProcessing(single="</s> $A", special_tokens=[("</s>", 2)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="</s>", eos_token="</s>",
                                       unk_token="<unk>", pad_token="<pad>", additional_special_tokens=["<image>"])
    processor = Blip2Processor(BlipImageProcessor(size={"height": 16, "width": 16}), tokenizer,
                               num_query_tokens=4 if expanded else None)
    config = Blip2Config(vision_config={"hidden_size": 32, "intermediate_size": 64,
                                        "num_hidden_layers": 2, "num_attention_heads": 4,
                                        "image_size": 16, "patch_size": 4, "initializer_range": 0.02},
                         qformer_config={"hidden_size": 32, "intermediate_size": 64,
                                         "num_hidden_layers": 2, "num_attention_heads": 4,
                                         "encoder_hidden_size": 32, "cross_attention_frequency": 1,
                                         "hidden_dropout_prob": 0.1, "attention_probs_dropout_prob": 0.1},
                         text_config={"model_type": "opt", "hidden_size": 32, "word_embed_proj_dim": 32,
                                      "ffn_dim": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
                                      "vocab_size": len(tokenizer), "max_position_embeddings": 128,
                                      "bos_token_id": 2, "eos_token_id": 2, "pad_token_id": 0},
                         num_query_tokens=4, image_token_index=3 if expanded else None)
    return RetinalBLIP2(Blip2ForConditionalGeneration(config), processor, mode)


def synthetic_samples():
    a = np.zeros((24, 24, 3), dtype=np.uint8)
    a[:, :12, 0] = 200
    b = np.zeros((24, 24, 3), dtype=np.uint8)
    b[::2, :, 1] = 230
    return [Image.fromarray(a), Image.fromarray(b)], ["Are there exudates?", "Are there exudates?"], ["yes", "no"]
