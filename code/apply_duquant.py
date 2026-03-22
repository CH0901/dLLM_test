<<<<<<< HEAD:code/apply_duquant.py

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
=======
import torch

def apply_duquant_to_model(model, duquant_params, device="cuda"):
    """DuQuant smooth scale을 weight에 흡수"""
    
    # 모델 타입에 따라 레이어 접근
    if hasattr(model, 'model'):
        # Case 1: model.transformer (LLaDA)
        if hasattr(model.model, 'transformer'):
            transformer = model.model.transformer
            
            if hasattr(transformer, 'blocks'):
                blocks = transformer.blocks
            elif isinstance(transformer, dict) and 'blocks' in transformer:
                blocks = transformer['blocks']
            elif hasattr(transformer, 'h'):
                blocks = transformer.h
            else:
                raise ValueError(f"알 수 없는 transformer 구조: {type(transformer)}")
        
        # Case 2: model.layers (DREAM)
        elif hasattr(model.model, 'layers'):
            blocks = model.model.layers
        
        else:
            raise ValueError("model.transformer도 model.layers도 없습니다")
    else:
        raise ValueError("model 속성이 없습니다")
    
    print(f"Applying DuQuant to {len(blocks)} blocks...")
    
    for idx, blk in enumerate(blocks):
        p = duquant_params[idx]
        get = lambda k: p[k].to(device) if k in p else None
        
        # 1. QKV smooth
        s = get("qkv_smooth_scale")
        if s is not None and hasattr(blk, 'attn_norm'):
            blk.attn_norm.weight.data /= s
            if hasattr(blk, 'q_proj'):
                blk.q_proj.weight.data *= s.unsqueeze(0)
            if hasattr(blk, 'k_proj'):
                blk.k_proj.weight.data *= s.unsqueeze(0)
            if hasattr(blk, 'v_proj'):
                blk.v_proj.weight.data *= s.unsqueeze(0)
        
        # 2. Attn-out smooth
        s = get("out_smooth_scale")
        if s is not None and hasattr(blk, 'v_proj') and hasattr(blk, 'attn_out'):
            blk.v_proj.weight.data /= s.unsqueeze(1)
            blk.attn_out.weight.data *= s.unsqueeze(0)
        
        # 3. FC1 smooth
        s = get("fc1_smooth_scale")
        if s is not None and hasattr(blk, 'ff_norm'):
            blk.ff_norm.weight.data /= s
            if hasattr(blk, 'ff_proj'):
                blk.ff_proj.weight.data *= s.unsqueeze(0)
            if hasattr(blk, 'up_proj'):
                blk.up_proj.weight.data *= s.unsqueeze(0)
        
        # 4. Down smooth
        s = get("down_smooth_scale")
        if s is not None and hasattr(blk, 'up_proj') and hasattr(blk, 'ff_out'):
            blk.up_proj.weight.data /= s.unsqueeze(1)
            blk.ff_out.weight.data *= s.unsqueeze(0)
        
        if idx % 8 == 0:
            print(f"  layer {idx}/{len(blocks)-1}")
    
    print("✅ DuQuant smooth scale absorption complete")
>>>>>>> 417ccdbe (Add masked recovery experiments and QDLM quantization debugging notes):llada/apply_duquant.py
    return model
