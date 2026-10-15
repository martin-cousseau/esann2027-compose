# ===================== CORE: calibration, OBS compression, evaluation =====================
import math, time, json, gc, random
import torch, torch.nn as nn

def tcr(sparsity, bits):
    """Theoretical compression rate (IJCNN 2025 definition), sparsity in [0,1], 16-bit baseline."""
    return 1.0 - (bits / 16.0) * (1.0 - sparsity)

# ---------- quantizer (asymmetric min-max, per output row, per group of input columns) ----------
def find_params(x, bits):
    maxq = 2 ** bits - 1
    xmin = torch.minimum(x.min(1).values, torch.zeros(x.shape[0], device=x.device))
    xmax = torch.maximum(x.max(1).values, torch.zeros(x.shape[0], device=x.device))
    both0 = (xmin == 0) & (xmax == 0)
    xmin[both0], xmax[both0] = -1.0, 1.0
    scale = (xmax - xmin) / maxq
    zero = torch.round(-xmin / scale)
    return scale, zero, maxq

def quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

# ---------- unified OBS sweep (SparseGPT/GPTQ family) ----------
@torch.no_grad()
def obs_sweep(W, H, sparsity=0.0, bits=16, fixed_mask=None, fixed_grid=None,
              blocksize=128, groupsize=128, percdamp=0.01):
    """
    One column-wise OBS pass over W (rows x cols) with Hessian H (cols x cols).
      sparsity>0 and fixed_mask None -> select an unstructured mask (SparseGPT, adaptive per block)
      fixed_mask given               -> keep exactly these weights at zero (mask preserved)
      bits<16                        -> quantize every column (GPTQ); grid per (row, group)
      fixed_grid given               -> re-use an existing grid {group_idx: (scale, zero)} instead of fitting one
    Returns (W_hat, mask, grid).
    """
    W = W.clone().float()
    rows, cols = W.shape
    H = H.clone()
    bad = int((~torch.isfinite(H)).sum().item())
    if bad:
        print(f"    WARNING: {bad} non-finite Hessian entries (fp16 overflow in calibration) -> set to 0")
        H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(H))
    idx = torch.arange(cols, device=W.device)
    Hinv = None
    for attempt in range(5):
        try:
            Hd = H.clone(); Hd[idx, idx] += damp * (10 ** attempt)
            L = torch.linalg.cholesky(Hd)
            Hinv = torch.cholesky_inverse(L)
            Hinv = torch.linalg.cholesky(Hinv, upper=True)
            break
        except Exception:
            Hinv = None
    if Hinv is None:
        print("    WARNING: Cholesky failed after 5 damping levels -> diagonal Hessian fallback")
        dg = torch.diag(H).clamp_min(0)
        Hinv = torch.diag(1.0 / torch.sqrt(dg + percdamp * dg.mean() + 1e-8))
    quant = bits < 16
    maxq = 2 ** bits - 1
    mask = torch.zeros_like(W, dtype=torch.bool)
    grid = {}
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols); count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1); Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        if fixed_mask is not None:
            mask1 = fixed_mask[:, i1:i2].clone()
        elif sparsity > 0:
            tmp = W1 ** 2 / (torch.diag(Hinv1).reshape(1, -1)) ** 2
            k = int(round(tmp.numel() * sparsity))
            mask1 = torch.zeros(tmp.numel(), dtype=torch.bool, device=W.device)
            if k > 0: mask1[torch.argsort(tmp.flatten())[:k]] = True   # exactly k weights (no tie overshoot)
            mask1 = mask1.reshape(tmp.shape)
        else:
            mask1 = torch.zeros_like(W1, dtype=torch.bool)
        for i in range(count):
            col = i1 + i
            w = W1[:, i]; d = Hinv1[i, i]
            q = w.clone(); q[mask1[:, i]] = 0
            if quant:
                if col % groupsize == 0:
                    g = col // groupsize
                    if fixed_grid is not None:
                        scale, zero = fixed_grid[g]
                    else:
                        blk = W1[:, i:i + groupsize] * (~mask1[:, i:i + groupsize])
                        scale, zero, _ = find_params(blk, bits)
                    grid[g] = (scale, zero)
                scale, zero = grid[col // groupsize]
                q = quantize(q, scale, zero, maxq)
                q[mask1[:, i]] = 0
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err
        W[:, i1:i2] = Q1
        mask[:, i1:i2] = mask1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    return W, mask, grid

def compress_weight(W, H, sparsity, bits, order, groupsize=128):
    """order in {'dense','prune','quant','joint','p_then_q','q_then_p'}"""
    if order == 'dense':
        return W.clone().float(), torch.zeros_like(W, dtype=torch.bool)
    if order == 'prune':
        Wh, m, _ = obs_sweep(W, H, sparsity=sparsity, bits=16, groupsize=groupsize); return Wh, m
    if order == 'quant':
        Wh, m, _ = obs_sweep(W, H, sparsity=0.0, bits=bits, groupsize=groupsize); return Wh, m
    if order == 'joint':
        Wh, m, _ = obs_sweep(W, H, sparsity=sparsity, bits=bits, groupsize=groupsize); return Wh, m
    if order == 'p_then_q':
        Wp, m, _ = obs_sweep(W, H, sparsity=sparsity, bits=16, groupsize=groupsize)
        Wh, m2, _ = obs_sweep(Wp, H, bits=bits, fixed_mask=m, groupsize=groupsize); return Wh, m2
    if order == 'q_then_p':
        Wq, _, grid = obs_sweep(W, H, sparsity=0.0, bits=bits, groupsize=groupsize)
        Wh, m, _ = obs_sweep(Wq, H, sparsity=sparsity, bits=bits, fixed_grid=grid, groupsize=groupsize); return Wh, m
    raise ValueError(order)

def rel_recon_error(W, Wh, H):
    D = (W.float() - Wh.float())
    num = torch.trace(D @ H @ D.T); den = torch.trace(W.float() @ H @ W.float().T)
    return (num / den).item()

def max_levels_per_group(Wh, groupsize, sample_rows=64):
    rows, cols = Wh.shape
    r = torch.linspace(0, rows - 1, min(rows, sample_rows)).long()
    m = 0
    for g0 in range(0, cols, groupsize):
        blk = Wh[r, g0:g0 + groupsize]
        for row in blk:
            m = max(m, torch.unique(row).numel())
    return m

# ---------- calibration data + layer-wise sequential compression ----------
def get_layers(model):
    return model.model.layers

@torch.no_grad()
def capture_layer0_inputs(model, calib, dev):
    layers = get_layers(model)
    store = {'x': [], 'kw': None, 'pos': False}
    class Stop(Exception): pass
    def hook(mod, args, kwargs):
        if len(args) > 0:
            store['x'].append(args[0].detach()); store['pos'] = True
        else:
            store['x'].append(kwargs['hidden_states'].detach())
        if store['kw'] is None:
            store['kw'] = {k: v for k, v in kwargs.items() if k != 'hidden_states'}
        raise Stop
    h = layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    for ids in calib:
        try: model(ids.to(dev), use_cache=False)
        except Stop: pass
    h.remove()
    return torch.cat(store['x'], 0), store['kw']

def layer_fwd(layer, x, kw):
    out = layer(x, **kw)
    return out[0] if isinstance(out, (tuple, list)) else out

@torch.no_grad()
def compress_model(model, calib, sparsity, bits, order, dev, groupsize=128, verbose=True, log=print):
    """Sequential layer-wise compression. Returns per-sublayer diagnostics list."""
    layers = get_layers(model)
    inps, kw = capture_layer0_inputs(model, calib, dev)
    diags = []
    for li, layer in enumerate(layers):
        lins = {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}
        Hs = {n: torch.zeros(m.in_features, m.in_features, device=dev) for n, m in lins.items()}
        cnt = {n: 0 for n in lins}
        def mk(n):
            def f(mod, inp, out):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                t = x.shape[0]
                Hs[n] *= cnt[n] / (cnt[n] + t); cnt[n] += t
                x = math.sqrt(2 / cnt[n]) * x
                Hs[n] += x.T @ x
            return f
        hs = [m.register_forward_hook(mk(n)) for n, m in lins.items()]
        for j in range(inps.shape[0]):
            layer_fwd(layer, inps[j:j + 1], kw)
        for h in hs: h.remove()
        for n, m in lins.items():
            t0 = time.time()
            W = m.weight.data
            Wh, mask = compress_weight(W, Hs[n], sparsity, bits, order, groupsize)
            e = rel_recon_error(W, Wh, Hs[n])
            m.weight.data = Wh.to(W.dtype)
            Wc = m.weight.data
            mfrac = mask.float().mean().item()
            zfrac = (Wc == 0).float().mean().item()
            mask_kept = bool((Wc[mask] == 0).all().item()) if mask.any() else True
            lev = max_levels_per_group(Wc.float(), groupsize) if bits < 16 else -1
            d = dict(layer=li, name=n, shape=list(W.shape), mask_frac=mfrac, zero_frac=zfrac, mask_kept=mask_kept,
                     max_levels=lev, rel_err=e, sec=time.time() - t0)
            diags.append(d)
            if verbose:
                log(f"  L{li:02d} {n:24s} {str(tuple(W.shape)):13s} mask={mfrac:.3f} zeros={zfrac:.3f} maskKept={mask_kept!s:5s} levels/grp={lev:3d} relErr={e:.3e} {d['sec']:.1f}s")
        del Hs
        for j in range(inps.shape[0]):
            inps[j:j + 1] = layer_fwd(layer, inps[j:j + 1], kw)
        if dev == 'cuda': torch.cuda.empty_cache()
    return diags

@torch.no_grad()
def layerwise_reproduces_model(model, calib, dev):
    """Validation: running layers one-by-one with captured kwargs reproduces the model's own last-layer output."""
    layers = get_layers(model)
    ref = {}
    h = layers[-1].register_forward_hook(lambda m, i, o: ref.__setitem__('y', (o[0] if isinstance(o, (tuple, list)) else o).detach()))
    model(calib[0].to(dev), use_cache=False); h.remove()
    x, kw = capture_layer0_inputs(model, calib[:1], dev)
    for layer in layers: x = layer_fwd(layer, x, kw)
    return (x.float() - ref['y'].float()).abs().max().item(), ref['y'].float().abs().max().item()

@torch.no_grad()
def perplexity(model, enc_ids, seqlen, dev, max_windows=None):
    n = enc_ids.numel() // seqlen
    if max_windows: n = min(n, max_windows)
    nll = 0.0
    for i in range(n):
        ids = enc_ids[:, i * seqlen:(i + 1) * seqlen].to(dev)
        logits = model(ids, use_cache=False).logits.float()
        loss = nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:], reduction='sum')
        nll += loss.item()
    return math.exp(nll / (n * (seqlen - 1))), n
