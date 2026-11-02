"""CUDA-graph ByT5 decode step for transformers 4.30's T5Stack. Default ON; BYT5_CUDAGRAPH=0 disables.

Measured (benchmarks/cudagraph/, 200-clip OpenASL replay, 2026-11-02): decode 39.2 -> 14.7 ms/step,
generate() 2.47x faster, BLEU 20.09 -> 20.60 (paired bootstrap +0.52, CI [-0.10, +1.24]: no loss),
182/200 texts identical; the 18 that change do so from bf16 reduction order (they change the same way
with the static math run WITHOUT a graph). ~110 MB extra CUDA memory. Live A/B
(latency_runs/cudagraph_ab/): sequential post-cut 5.25 -> 3.41 s; overlap 0 refusals (was 7/40), 22 -> 3.7 s.

Why eager 4.30 cannot be graphed: the self-attention KV cache grows by torch.cat every step
(new shapes) and the relative position bias is rebuilt from Python ints. This module re-implements
ONE decoder step over preallocated buffers, reusing the model's own weights/modules:

  * self-attn K/V: [beams, heads, MAX_LEN, d_kv] per layer, written in place at a device-side step
    index; scores over all MAX_LEN slots, slots > t masked with finfo.min (as HF masks).
  * relative position bias: computed from the device step tensor with the model's own
    _relative_position_bucket / relative_attention_bias (decoder = unidirectional).
  * cross-attn K/V: computed ONCE per clip outside the graph, into buffers padded to an encoder
    length bucket (multiple of ENC_BUCKET), padded slots masked. One graph per bucket.
  * beam reorder: a static beam_idx buffer applied to the cache at the start of the next step
    (identity after each replay), so HF's _reorder_cache becomes one tiny copy.

HF 4.30's generate()/beam scorer is left untouched: install(model) swaps only forward(),
_reorder_cache() and the past sentinel, so the search itself is identical to eager.
Wired in by inference._get_cached_model() unless BYT5_CUDAGRAPH=0; all graphs are captured at load
(warm_all), so no clip pays a capture.
"""
import torch
from torch import nn
from transformers.modeling_outputs import Seq2SeqLMOutput

ENC_BUCKET = 64
# Decoder-length buckets: the graph for a step only touches cache slots [0, bucket). Beam reorder
# copies the whole touched cache every step, so a 768-slot graph costs ~8.5 ms/step of copies
# for a ~70-step clip; bucketing keeps that proportional to the real length. All buckets are
# VIEWS of one max_len buffer, so crossing a bucket copies nothing.
DEC_BUCKETS = (64, 128, 256, 512, 768)
# Cross-attn K/V capacity (encoder frames). All encoder buckets are VIEWS of one buffer of this
# size (separate buffers per bucket cost ~99 MB over 6 buckets). A clip whose encoder output is
# longer than this decodes on the normal eager path (install() falls back per clip).
ENC_CAP = 512


class _StaticPast(tuple):
    """Non-None past_key_values sentinel carrying the step count (HF only checks `is not None`)."""


class StaticDecoder:
    def __init__(self, model, beams=4, max_len=768):
        self.m = model
        dec = model.decoder
        cfg = dec.config
        self.L = len(dec.block)
        self.H, self.dk, self.D = cfg.num_heads, cfg.d_kv, cfg.d_model
        self.beams, self.max_len = beams, max_len
        p = next(model.parameters())
        self.dev, self.dt = p.device, p.dtype
        self.neg = torch.finfo(self.dt).min
        z = lambda *s: torch.zeros(*s, device=self.dev, dtype=self.dt)  # noqa: E731
        self.k = [z(beams, self.H, max_len, self.dk) for _ in range(self.L)]
        self.v = [z(beams, self.H, max_len, self.dk) for _ in range(self.L)]
        # static inputs
        self.ids = torch.zeros(beams, 1, dtype=torch.long, device=self.dev)
        self.t = torch.zeros((), dtype=torch.long, device=self.dev)
        self.ident = torch.arange(beams, device=self.dev)
        self.beam_idx = self.ident.clone()
        self.pos = torch.arange(max_len, device=self.dev)
        self.dec_buckets = sorted({min(b, max_len) for b in DEC_BUCKETS} | {max_len})
        self.enc_cap = ENC_CAP
        self.ck = [z(beams, self.H, self.enc_cap, self.dk) for _ in range(self.L)]
        self.cv = [z(beams, self.H, self.enc_cap, self.dk) for _ in range(self.L)]
        self.cross = {}           # enc bucket -> ((ck views, cv views), cmask)
        self.graphs = {}          # (enc bucket, dec bucket) -> (graph, logits out)
        self.pool = None
        self.cur = None
        self.steps = 0

    # ---- per clip -------------------------------------------------------------------------
    def _bufs(self, lp):
        if lp in self.cross:
            return self.cross[lp]
        ck = [x[:, :, :lp] for x in self.ck]
        cv = [x[:, :, :lp] for x in self.cv]
        cmask = torch.zeros(1, 1, 1, lp, device=self.dev, dtype=self.dt)
        self.cross[lp] = ((ck, cv), cmask)
        return self.cross[lp]

    @torch.no_grad()
    def prepare(self, enc_hidden):
        """enc_hidden: [1 or beams, Lenc, D]. Fill cross K/V for this clip; pick/capture the graph."""
        enc = enc_hidden[:1]
        le = enc.shape[1]
        lp = ((le + ENC_BUCKET - 1) // ENC_BUCKET) * ENC_BUCKET
        if lp > self.enc_cap:
            raise ValueError(f"encoder length {le} exceeds ENC_CAP {self.enc_cap}")
        (ck, cv), cmask = self._bufs(lp)
        for i, blk in enumerate(self.m.decoder.block):
            att = blk.layer[1].EncDecAttention
            k = att.k(enc).view(1, le, self.H, self.dk).transpose(1, 2)
            v = att.v(enc).view(1, le, self.H, self.dk).transpose(1, 2)
            ck[i].zero_(); cv[i].zero_()
            ck[i][:, :, :le].copy_(k.expand(self.beams, -1, -1, -1))
            cv[i][:, :, :le].copy_(v.expand(self.beams, -1, -1, -1))
        cmask.zero_()
        cmask[..., le:] = self.neg
        self.beam_idx.copy_(self.ident)
        self.cur = lp
        self.steps = 0

    # ---- the step ---------------------------------------------------------------------------
    def _step(self, cross, cmask, ld):
        dec = self.m.decoder
        ck, cv = cross
        B, H, dk = self.beams, self.H, self.dk
        # apply pending beam reorder (identity unless _reorder_cache set it)
        K = [x[:, :, :ld] for x in self.k]
        V = [x[:, :, :ld] for x in self.v]
        pos = self.pos[:ld]
        for i in range(self.L):
            K[i].copy_(K[i].index_select(0, self.beam_idx))
            V[i].copy_(V[i].index_select(0, self.beam_idx))
        self.beam_idx.copy_(self.ident)

        h = dec.embed_tokens(self.ids)                                   # [B,1,D]
        # self-attn position bias for query position t over keys 0..max_len-1, + causal mask
        att0 = dec.block[0].layer[0].SelfAttention
        rel = pos - self.t                                         # memory - query
        bucket = att0._relative_position_bucket(
            rel, bidirectional=False, num_buckets=att0.relative_attention_num_buckets,
            max_distance=att0.relative_attention_max_distance)
        bias = att0.relative_attention_bias(bucket)                     # [max_len, H]
        bias = bias.permute(1, 0)[None, :, None, :]                     # [1,H,1,max_len]
        future = (pos > self.t)[None, None, None, :]
        bias = bias.masked_fill(future, self.neg)

        for i, blk in enumerate(dec.block):
            # --- self attention
            sa = blk.layer[0]
            x = sa.layer_norm(h)
            a = sa.SelfAttention
            q = a.q(x).view(B, 1, H, dk).transpose(1, 2)                # [B,H,1,dk]
            kn = a.k(x).view(B, 1, H, dk).transpose(1, 2)
            vn = a.v(x).view(B, 1, H, dk).transpose(1, 2)
            # write slot t in place; the index is a device tensor, so no host sync (capture-safe)
            K[i].index_copy_(2, self.t.view(1), kn)
            V[i].index_copy_(2, self.t.view(1), vn)
            s = torch.matmul(q, K[i].transpose(3, 2)) + bias
            w = nn.functional.softmax(s.float(), dim=-1).type_as(s)
            o = torch.matmul(w, V[i]).transpose(1, 2).contiguous().view(B, 1, H * dk)
            h = h + a.o(o)
            # --- cross attention
            ca = blk.layer[1]
            x = ca.layer_norm(h)
            a = ca.EncDecAttention
            q = a.q(x).view(B, 1, H, dk).transpose(1, 2)
            s = torch.matmul(q, ck[i].transpose(3, 2)) + cmask
            w = nn.functional.softmax(s.float(), dim=-1).type_as(s)
            o = torch.matmul(w, cv[i]).transpose(1, 2).contiguous().view(B, 1, H * dk)
            h = h + a.o(o)
            # --- feed forward
            h = blk.layer[2](h)
        h = dec.final_layer_norm(h)
        if self.m.config.tie_word_embeddings:
            h = h * (self.m.model_dim ** -0.5)
        return self.m.lm_head(h)                                        # [B,1,V]

    @torch.no_grad()
    def _capture(self, lp, ld):
        cross, cmask = self.cross[lp]
        # ONE warmup stream for all captures: PyTorch keeps a cuBLAS workspace (~8 MB) alive per
        # stream it has seen, so a fresh stream per capture leaked 8 MB per graph (25 graphs = 200 MB).
        if getattr(self, "_side", None) is None:
            self._side = torch.cuda.Stream()
        s = self._side
        s.wait_stream(torch.cuda.current_stream())
        saved = [x.clone() for x in self.k + self.v]
        self.t.fill_(0)           # warmup must index inside [0, ld); replay sets the real t
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step(cross, cmask, ld)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(g, pool=self.pool):
            out = self._step(cross, cmask, ld)
        for x, y in zip(self.k + self.v, saved):
            x.copy_(y)
        self.beam_idx.copy_(self.ident)
        self.graphs[(lp, ld)] = (g, out)

    def _ld(self, t):
        for b in self.dec_buckets:
            if t < b:
                return b
        raise RuntimeError("static decoder: step beyond max_len")

    def capture_all(self, lp):
        """Pre-capture every decoder bucket for an encoder bucket (startup warmup)."""
        for ld in self.dec_buckets:
            if (lp, ld) not in self.graphs:
                self._capture(lp, ld)

    # ---- called from the patched forward ------------------------------------------------------
    def step(self, ids, t):
        key = (self.cur, self._ld(t))
        if key not in self.graphs:
            self._capture(*key)
        g, out = self.graphs[key]
        self.ids.copy_(ids)
        self.t.fill_(t)
        g.replay()
        self.steps += 1
        return out.clone()

    def eager_step(self, ids, t):
        """Same math without the graph (debug/control)."""
        cross, cmask = self.cross[self.cur]
        self.ids.copy_(ids)
        self.t.fill_(t)
        return self._step(cross, cmask, self._ld(t)).clone()


def install(model, beams=4, max_len=768, use_graph=True, sd=None):
    """Swap model.forward/_reorder_cache for the static path. Returns (decoder, uninstall).
    Pass an existing `sd` to reuse its buffers and captured graphs across calls."""
    sd = sd or StaticDecoder(model, beams=beams, max_len=max_len)
    orig_forward, orig_reorder = model.forward, model._reorder_cache
    state = {"eager": False, "fallbacks": 0}

    def forward(decoder_input_ids=None, encoder_outputs=None, past_key_values=None, **kw):
        enc = encoder_outputs[0]
        if past_key_values is None:
            # per-clip choice: too long for the static buffers, or a beam count the buffers were
            # not built for -> the whole clip runs on the normal eager path
            state["eager"] = (enc.shape[1] > sd.enc_cap or decoder_input_ids.shape[0] != beams)
            state["fallbacks"] += state["eager"]
        if state["eager"]:
            return orig_forward(decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_outputs,
                                past_key_values=past_key_values, **kw)
        if past_key_values is None:
            sd.prepare(enc)
            t = 0
        else:
            t = past_key_values[0] + 1
        if t >= max_len:
            raise RuntimeError("static decoder: step beyond max_len")
        logits = sd.step(decoder_input_ids[:, -1:], t) if use_graph else \
            sd.eager_step(decoder_input_ids[:, -1:], t)
        return Seq2SeqLMOutput(logits=logits, past_key_values=_StaticPast((t,)))

    def reorder(past, beam_idx):
        if state["eager"]:
            return orig_reorder(past, beam_idx)
        sd.beam_idx.copy_(beam_idx)
        return past

    sd.state = state
    model.forward = forward
    model._reorder_cache = reorder

    def uninstall():
        model.forward = orig_forward
        model._reorder_cache = orig_reorder
    return sd, uninstall


@torch.no_grad()
def warm_all(sd):
    """Capture every (encoder bucket, decoder bucket) graph up front. Returns the count."""
    for lp in range(ENC_BUCKET, sd.enc_cap + 1, ENC_BUCKET):
        sd.prepare(torch.zeros(1, lp, sd.D, device=sd.dev, dtype=sd.dt))
        sd.capture_all(lp)
    sd.cur = None
    return len(sd.graphs)
