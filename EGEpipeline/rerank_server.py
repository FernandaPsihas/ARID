"""Cross-encoder rerank HTTP service for ARID.

Runs INSIDE QuarkyLab's apptainer/torch container -- the lightweight query
venv (answer.py/chat.py/search.py) has no torch, so the reranker can't live
in-process there. Instead it's a localhost HTTP service the query path reaches
over HTTP, exactly like Ollama (11434) and Qdrant (6333) already are. A
cross-encoder reads (query, passage) TOGETHER in one forward pass, so it scores
"does this chunk answer this query" rather than the bi-encoder embedder's
"do these look similar in the abstract" -- which is what lets it demote a
class's boilerplate constructor below the method that does the actual work
(the 7/24 pixel-map ranking bug).

Endpoints:
  POST /rerank  {"query": "...", "passages": ["...", ...]}
     -> {"scores": [float, ...]}   # higher = more relevant, input order preserved
  GET  /health  -> {"ok": true, "model": "..."}

Launch on QuarkyLab (torch lives in the container, not the query venv):
  apptainer exec --nv /workspace/containers/base.sif \
      python EGEpipeline/rerank_server.py --port 8095

Model: ARID_RERANK_MODEL/ARID_RERANK_KIND env or --model/--kind. Default is
Qwen/Qwen3-Reranker-0.6B (causal), which won the 7/24 bake-off -- it fixes the
pixel-map ranking pathology AND improves the aggregate (recall@1 0.5->0.575,
MRR 0.68->0.78). The general-domain BAAI/bge-reranker-base (--kind seqcls) is
still selectable but LOST that bake-off badly (regressed recall@1 to 0.25);
it's kept only for reproducing that comparison, not for production use.
"""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

MODEL_NAME = os.environ.get("ARID_RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
# Two reranker families need two scoring paths, both behind the same /rerank API:
#   "seqcls" -- bge-reranker-* etc.: a sequence-classification head, score = its logit.
#   "causal" -- Qwen3-Reranker-*: a causal LM judged on the yes/no next-token logits,
#               score = P(yes). Same-family as our qwen3-embedding, code-aware.
# DEFAULT is Qwen3-Reranker (causal): the 7/24 bake-off showed the general-domain
# bge-reranker-base fixed the target bug but REGRESSED the aggregate hard (recall@1
# 0.5->0.25) -- it doesn't understand code relevance. Qwen3-Reranker-0.6B instead
# beats baseline across the board (recall@1 0.5->0.575, MRR 0.68->0.78) AND fixes
# the pixel-map pathology. Don't swap back to a general-text reranker without
# re-checking the aggregate, not just one query.
MODEL_KIND = os.environ.get("ARID_RERANK_KIND", "causal")
MAX_LEN = int(os.environ.get("ARID_RERANK_MAXLEN", "512"))
BATCH = int(os.environ.get("ARID_RERANK_BATCH", "32"))
# Instruction handed to the causal reranker -- framed for THIS domain (code Q&A), which is
# the whole point of using a code-capable reranker over a general web-QA one.
CAUSAL_INSTRUCT = ("Given a question about a software codebase, judge whether the code "
                   "snippet contains the actual logic that answers it (not merely a class "
                   "declaration, constructor, or mention of the relevant name).")
_CAUSAL_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based "
                  "on the Query and the Instruct provided. Note that the answer can only be "
                  "\"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
_CAUSAL_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

_tok = None
_model = None
_yes_id = _no_id = None
_prefix_ids = _suffix_ids = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load() -> None:
    global _tok, _model, _yes_id, _no_id, _prefix_ids, _suffix_ids
    from transformers import AutoTokenizer
    if MODEL_KIND == "causal":
        from transformers import AutoModelForCausalLM
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16 if _device == "cuda" else torch.float32
        ).eval().to(_device)
        _yes_id = _tok.convert_tokens_to_ids("yes")
        _no_id = _tok.convert_tokens_to_ids("no")
        _prefix_ids = _tok.encode(_CAUSAL_PREFIX, add_special_tokens=False)
        _suffix_ids = _tok.encode(_CAUSAL_SUFFIX, add_special_tokens=False)
    else:
        from transformers import AutoModelForSequenceClassification
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval().to(_device)
    print(f"rerank: loaded {MODEL_NAME} (kind={MODEL_KIND}) on {_device}", flush=True)


def _score_seqcls(query: str, passages: list[str]) -> list[float]:
    out: list[float] = []
    for i in range(0, len(passages), BATCH):
        batch = passages[i:i + BATCH]
        enc = _tok([query] * len(batch), batch, padding=True, truncation=True,
                   return_tensors="pt", max_length=MAX_LEN)
        enc = {k: v.to(_device) for k, v in enc.items()}
        with torch.no_grad():
            logits = _model(**enc).logits.view(-1).float().tolist()
        out.extend(logits)
    return out


def _score_causal(query: str, passages: list[str]) -> list[float]:
    """Qwen3-Reranker: P(yes) from the last-token logits, per the model card's
    recipe. Body of each pair is truncated to leave room for the fixed
    instruction wrapper within MAX_LEN."""
    out: list[float] = []
    body_cap = MAX_LEN - len(_prefix_ids) - len(_suffix_ids)
    for i in range(0, len(passages), BATCH):
        batch = passages[i:i + BATCH]
        texts = [f"<Instruct>: {CAUSAL_INSTRUCT}\n<Query>: {query}\n<Document>: {d}" for d in batch]
        enc = _tok(texts, padding=False, truncation="longest_first",
                   return_attention_mask=False, max_length=body_cap)
        enc["input_ids"] = [_prefix_ids + ids + _suffix_ids for ids in enc["input_ids"]]
        enc = _tok.pad(enc, padding=True, return_tensors="pt")
        enc = {k: v.to(_device) for k, v in enc.items()}
        with torch.no_grad():
            last = _model(**enc).logits[:, -1, :]
            pair = torch.stack([last[:, _no_id], last[:, _yes_id]], dim=1)
            probs = torch.nn.functional.log_softmax(pair.float(), dim=1)[:, 1].exp()
        out.extend(probs.tolist())
    return out


def _score(query: str, passages: list[str]) -> list[float]:
    return _score_causal(query, passages) if MODEL_KIND == "causal" else _score_seqcls(query, passages)


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._send(200, {"ok": True, "model": MODEL_NAME, "device": _device})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/rerank":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            query = req["query"]
            passages = req["passages"]
            scores = _score(query, passages) if passages else []
            self._send(200, {"scores": scores})
        except Exception as e:  # a bad request must not take the whole server down
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *args):  # quiet the default per-request stderr spam
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--model", default=None, help="override ARID_RERANK_MODEL / default")
    ap.add_argument("--kind", default=None, choices=["seqcls", "causal"],
                    help="scoring path: seqcls (bge-*) or causal (Qwen3-Reranker-*)")
    args = ap.parse_args()
    global MODEL_NAME, MODEL_KIND
    if args.model:
        MODEL_NAME = args.model
    if args.kind:
        MODEL_KIND = args.kind
    _load()
    srv = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"rerank: serving on http://{args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
