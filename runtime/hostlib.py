#!/usr/bin/env python3
"""Host-side pieces of the Gemma-4-E2B A16W8 v79 runtime, matched to the gemma3n architecture.

Graph boundary (from decode_fixed.py / host_generate.py):
  host: token id -> inputs_embeds (embed_tokens[id] * sqrt(H))
                 -> per_layer_inputs (embed_tokens_per_layer[id].reshape(NL,PLD) * sqrt(PLD))
  NPU decode graph: takes inputs_embeds, per_layer_inputs, position_ids(int32), cache_position(int32),
                    full_mask, sliding_mask, 15x past_k/v  ->  hidden (final-normed), 15x present_k/v
  host: hidden -> logits = hidden @ embed_tokens.T  (tied, UNSCALED)
              -> softcap: 30*tanh(logits/30) -> argmax

Embedding scales are read straight from the gemma3n source:
  embed_tokens          embed_scale = hidden_size ** 0.5              (H=1536 -> ~39.1918)
  embed_tokens_per_layer embed_scale = hidden_size_per_layer ** 0.5   (PLD=256 -> 16.0)
lm_head is tied to embed_tokens.weight and applied WITHOUT the embed scale.
"""
import json, pathlib, numpy as np

HERE = pathlib.Path(__file__).resolve().parent
HM = HERE.parent / "host-model"

# dims
H = 1536
PLD = 256
NL = 35
CTX = 4096
VOCAB = 262144
SOFTCAP = 30.0
EMB_SCALE = float(np.sqrt(H))          # 39.19183...
PLE_SCALE = float(np.sqrt(PLD))        # 16.0
# Finite mask value — MUST match the value used at quantization calibration
# (decode_pipeline_v2.py NEG=-1e4). -inf/finfo.min cannot survive int16 activation
# quantization (blows out the range so real scores round to 0); -1e4 still zeroes
# softmax (exp(-1e4)=0) while leaving real scores (~+-50) well resolved.
NEG = -1e4

# Gemma-4 chat-template token ids (verified against transformers apply_chat_template).
BOS_ID = 2          # <bos>
TURN_START = 105    # <|turn>
TURN_END = 106      # <turn|>   -- also the generation stop token
NL_ID = 107         # '\n'
ROLE_USER = 2364    # 'user'
ROLE_MODEL = 4368   # 'model'
STOP_IDS = {TURN_END, 1}   # <turn|> or <eos>

# KV layout: head dim 512 for layers 4,9,14; else 256 (from decode-io.tsv, 15 non-shared layers)
KV_HD = [256]*4 + [512] + [256]*4 + [512] + [256]*4 + [512]
NC = 15


def _load_bf16(path, shape):
    raw = np.fromfile(path, dtype=np.uint16)
    f32 = (raw.astype(np.uint32) << 16).view(np.float32)
    return f32.reshape(shape)


def _bf16_row(mm, idx):
    """Convert one bf16 row (uint16 memmap slice) -> float32."""
    return (mm[idx].astype(np.uint32) << 16).view(np.float32)


class HostModel:
    def __init__(self):
        # memmap as uint16 so we never materialize the full float32 tensors (~11GB spike).
        self.embed = np.memmap(HM / "embed_tokens_weight.bf16", dtype=np.uint16,
                               mode="r", shape=(VOCAB, H))            # [V,H] bf16
        self.ple = np.memmap(HM / "embed_tokens_per_layer_weight.bf16", dtype=np.uint16,
                             mode="r", shape=(VOCAB, NL*PLD))         # [V, NL*PLD] bf16
        # tokenizer
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(HM / "tokenizer.json"))

    # ---- tokenization ----
    def encode(self, text):
        return self.tok.encode(text).ids

    def encode_chat(self, user_text):
        """Gemma-4 canonical chat template, built from raw token ids.

        The `tokenizers` library has no chat-template support, so we assemble the same
        id sequence transformers' apply_chat_template() produces. Verified byte-exact
        against transformers 5.12 for google/gemma-4-E2B-it:
          '<bos><|turn>user\\nTHE PROMPT<turn|>\\n<|turn>model\\n'
        Without this the instruct model degenerates ('France is France is ...'); with it
        plain greedy decoding is coherent.
        """
        return ([BOS_ID, TURN_START, ROLE_USER, NL_ID]
                + self.encode(user_text)
                + [TURN_END, NL_ID, TURN_START, ROLE_MODEL, NL_ID])

    def decode(self, ids):
        return self.tok.decode(ids)

    # ---- host embeddings for one token ----
    def embeds(self, token_id):
        ie = (_bf16_row(self.embed, token_id) * EMB_SCALE).reshape(1, 1, H)
        ple = (_bf16_row(self.ple, token_id) * PLE_SCALE).reshape(1, 1, NL, PLD)
        return ie.astype(np.float32), ple.astype(np.float32)

    # ---- masks (additive [1,1,1,CTX]) ----
    def masks(self, pos):
        j = np.arange(CTX)
        full = np.where(j <= pos, 0.0, NEG).astype(np.float32).reshape(1, 1, 1, CTX)
        slide = np.where((j <= pos) & (j > pos - 512), 0.0, NEG).astype(np.float32).reshape(1, 1, 1, CTX)
        return full, slide

    # ---- lm head (tied, unscaled) + softcap ----
    def _embed_f32(self):
        # Lazily materialize the tied word-embedding as float32 [V,H] for lm_head (~1.6GB).
        if getattr(self, "_ef32", None) is None:
            self._ef32 = (self.embed.astype(np.uint32) << 16).view(np.float32)
        return self._ef32

    def logits(self, hidden):
        h = np.asarray(hidden, np.float32).reshape(H)
        lg = self._embed_f32() @ h                       # [V,H] @ [H] -> [V]
        lg = SOFTCAP * np.tanh(lg / SOFTCAP)
        return lg

    def argmax_next(self, hidden):
        return int(self.logits(hidden).argmax())


if __name__ == "__main__":
    # smoke: load + embed a couple tokens, print shapes/norms
    m = HostModel()
    ids = m.encode("The capital of France is")
    print("prompt ids:", ids, "->", repr(m.decode(ids)))
    ie, ple = m.embeds(ids[0])
    print("inputs_embeds", ie.shape, "norm", float(np.linalg.norm(ie)))
    print("per_layer_inputs", ple.shape, "norm", float(np.linalg.norm(ple)))
    f, s = m.masks(3)
    print("full_mask nonneg count", int((f == 0).sum()), "sliding", int((s == 0).sum()))
