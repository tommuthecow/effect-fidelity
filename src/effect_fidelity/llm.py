"""Minimal local-LLM scorer: returns the model's probability distribution over answer options.

Uses llama.cpp (GGUF, CPU) and reads next-token logits directly, restricted to the option
labels (A/B/C... or digits). Reuses the KV cache for shared prompt prefixes, so many personas
answering the same scenario are cheap.
"""
from __future__ import annotations
import numpy as np

TEMPLATES = {
    "chatml": ("<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"),
    "llama3": ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|>"
               "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
}


class LocalLM:
    def __init__(self, path: str, template: str, n_threads: int = 7, n_ctx: int = 2048):
        from llama_cpp import Llama
        self.llm = Llama(path, n_ctx=n_ctx, n_threads=n_threads, n_threads_batch=n_threads, n_batch=512, logits_all=False, verbose=False)
        self.template = TEMPLATES[template]
        self.n_vocab = self.llm.n_vocab()
        self._label_ids: dict[str, int] = {}

    def _label_id(self, label: str) -> int:
        if label not in self._label_ids:
            ids = self.llm.tokenize(label.encode(), add_bos=False, special=False)
            assert len(ids) == 1, f"label {label!r} is not a single token: {ids}"
            self._label_ids[label] = ids[0]
        return self._label_ids[label]

    def option_probs(self, system: str, user: str, labels: list[str]) -> np.ndarray:
        prompt = self.template.format(sys=system, user=user)
        toks = self.llm.tokenize(prompt.encode(), add_bos=False, special=True)
        llm = self.llm
        prev = list(llm._input_ids)
        k = 0
        for a, b in zip(prev, toks):
            if a != b:
                break
            k += 1
        k = min(k, len(toks) - 1)  # always eval at least one token to refresh logits
        if k > 0 and llm._ctx.kv_cache_seq_rm(-1, k, -1):
            llm.n_tokens = k
        else:
            llm.reset()
            k = 0
        llm.eval(toks[k:])
        logits = np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(self.n_vocab,)).astype(np.float64)
        sel = np.array([logits[self._label_id(l)] for l in labels])
        p = np.exp(sel - sel.max())
        return p / p.sum()
