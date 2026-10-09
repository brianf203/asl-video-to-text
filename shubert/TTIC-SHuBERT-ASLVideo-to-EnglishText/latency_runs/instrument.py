"""Measurement-only instrumentation for the latency study. Changes no pipeline file.

Everything here is a timing wrapper around an EXISTING module or object boundary:

  * `features.test`                          -> the whole ByT5 stage (wall)
  * `model.generate`                         -> generate wall + exact decode steps
  * `model.encoder`                          -> SHuBERT + ByT5-encoder wall (fwd hook)
  * `model.encoder.adapter.signhubert_adapter` -> SHuBERT alone (fwd hook)

No weight, threshold, dtype, device or control-flow change, so outputs are untouched --
which the 3-pass determinism check in analysis.py independently verifies.

HONEST LIMIT on the split: the hooks time CPU-side wall at module boundaries with no
mid-generate cuda.synchronize(), because inserting one WOULD change execution timing. The
decode step was measured on 2026-08-24 to be ~100% CPU dispatch with a 0.01 ms GPU tail,
so for decode this is the right instrument; for the encoder/SHuBERT split it can
under-attribute GPU work that lands after the boundary. Reported, not hidden.
"""
import threading
import time

# One slot per thread: only the worker thread translates, but keep it thread-local so a
# stray concurrent call can never cross-contaminate another utterance's numbers.
_local = threading.local()


def _slot():
    d = getattr(_local, "d", None)
    if d is None:
        d = _local.d = {}
    return d


def current():
    """The stage timings accumulated for the utterance this thread is translating."""
    return dict(_slot())


def reset():
    _slot().clear()


def install(config):
    """Wrap the boundaries. Call once, after the models are warm."""
    import features
    import inference

    model, _tokenizer, _device = inference._get_cached_model(
        config['slt_model_checkpoint'],
        config['slt_tokenizer_checkpoint'],
        config['temp_dir'],
    )

    # --- SHuBERT alone -------------------------------------------------------
    shubert = model.encoder.adapter.signhubert_adapter

    def _pre_sh(_m, _i):
        _slot()['_t_shubert'] = time.time()

    def _post_sh(_m, _i, _o):
        t0 = _slot().pop('_t_shubert', None)
        if t0 is not None:
            _slot()['shubert_s'] = time.time() - t0

    shubert.register_forward_pre_hook(_pre_sh)
    shubert.register_forward_hook(_post_sh)

    # --- encoder (SHuBERT + ByT5 encoder blocks) -----------------------------
    def _pre_enc(_m, _i):
        _slot()['_t_enc'] = time.time()

    def _post_enc(_m, _i, _o):
        t0 = _slot().pop('_t_enc', None)
        if t0 is not None:
            _slot()['encoder_total_s'] = time.time() - t0

    model.encoder.register_forward_pre_hook(_pre_enc)
    model.encoder.register_forward_hook(_post_enc)

    # --- generate: wall + exact decode steps ---------------------------------
    _real_generate = model.generate

    def _timed_generate(*a, **kw):
        t0 = time.time()
        ids = _real_generate(*a, **kw)
        d = _slot()
        d['generate_s'] = time.time() - t0
        try:
            d['decode_steps'] = int(ids.shape[-1])
        except Exception:
            pass
        enc = d.get('encoder_total_s')
        if enc is not None:
            # Decode is what is left of generate() once the single encoder pass is out.
            d['byt5_decode_s'] = max(0.0, d['generate_s'] - enc)
            sh = d.get('shubert_s')
            if sh is not None:
                d['byt5_encoder_s'] = max(0.0, enc - sh)
        return ids

    model.generate = _timed_generate

    # --- the whole ByT5 stage, as features.py invokes it ---------------------
    _real_test = features.test

    def _timed_test(*a, **kw):
        t0 = time.time()
        try:
            return _real_test(*a, **kw)
        finally:
            _slot()['byt5_stage_s'] = time.time() - t0

    features.test = _timed_test
    return model
