
import torch

def apply_duquant_to_model(model, duquant_params, device="cuda"):
    """
    smooth scale만 weight에 흡수.
    rotation/permutation은 activation도 함께 변환해야 하므로 현재 단계에서는 제외.
    """
    blocks = model.model.transformer["blocks"]
    print(f"Applying DuQuant smooth scales to {len(blocks)} layers ...")

    for idx, blk in enumerate(blocks):
        p   = duquant_params[idx]
        get = lambda k: p[k].to(device) if k in p else None

        # ── 1. QKV smooth scale ───────────────────────────────────────
        s = get("qkv_smooth_scale")
        if s is not None:
            blk.attn_norm.weight.data /= s
            for proj in [blk.q_proj, blk.k_proj, blk.v_proj]:
                proj.weight.data *= s.unsqueeze(0)

        # ── 2. Attn-out smooth scale ──────────────────────────────────
        s = get("out_smooth_scale")
        if s is not None:
            blk.v_proj.weight.data   /= s.unsqueeze(1)
            blk.attn_out.weight.data *= s.unsqueeze(0)

        # ── 3. FC1 smooth scale ───────────────────────────────────────
        s = get("fc1_smooth_scale")
        if s is not None:
            blk.ff_norm.weight.data /= s
            for proj in [blk.ff_proj, blk.up_proj]:
                proj.weight.data *= s.unsqueeze(0)

        # ── 4. Down smooth scale ──────────────────────────────────────
        s = get("down_smooth_scale")
        if s is not None:
            blk.up_proj.weight.data /= s.unsqueeze(1)
            blk.ff_out.weight.data  *= s.unsqueeze(0)

        if idx % 8 == 0:
            print(f"  layer {idx}/{len(blocks)-1} done")

    print("DuQuant smooth scale absorption complete ✓")
    print("(rotation/permutation은 다음 단계에서 runtime hook으로 적용 예정)")
    return model
