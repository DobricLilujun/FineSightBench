# Attention Visualization Methods for Vision-Language Models (VLMs)

## Research Summary

**Key Papers:**
- **"Quantifying Attention Flow in Transformers"** (Abnar & Zuidema, 2020) — arXiv:2005.00928. Proposes **Attention Rollout** and **Attention Flow** as post-hoc methods to approximate the attention to input tokens, accounting for information mixing across layers via residual connections.
- **"Transformer Interpretability Beyond Attention Visualization"** (Chefer, Gur & Wolf, CVPR 2021) — arXiv:2012.09838. Combines LRP relevance, gradient weighting, and rollout for class-specific explainability.
- **"Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers"** (Chefer et al., 2021) — Extends to cross-attention in multimodal models (CLIP, LXMERT, ViLBERT, DETR).

**Key Repositories:**
- `jacobgil/vit-explain` (1.1k stars) — Attention Rollout + Gradient Attention Rollout for ViT
- `hila-chefer/Transformer-Explainability` (2k stars) — CVPR 2021 method for ViT + BERT
- `hila-chefer/Transformer-MM-Explainability` — Cross-attention for CLIP, LXMERT, DETR
- `jacobgil/pytorch-grad-cam` (12.8k stars) — GradCAM family for CNNs and ViTs

---

## Part 1: Direct Attention Visualization

### 1.1 Extracting Raw Attention Weights

In a standard transformer, each attention layer produces:
```
attn_weights: shape [batch, num_heads, seq_len, seq_len]
```

For a ViT with 224×224 input and patch_size=16:
- `seq_len = (224/16)² + 1 = 197` (196 patches + 1 CLS token)
- The CLS token is at index 0

**Hook-based extraction** (from `jacobgil/vit-explain`):
```python
class AttentionExtractor:
    def __init__(self, model, attention_layer_name='attn_drop'):
        self.attentions = []
        for name, module in model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self._get_attention)

    def _get_attention(self, module, input, output):
        # output shape: [batch, num_heads, seq_len, seq_len]
        self.attentions.append(output.cpu().detach())

    def reset(self):
        self.attentions = []
```

### 1.2 Single Layer, Single Head Attention Map

```python
def single_head_attention_map(attn_weights, layer_idx, head_idx, 
                                query_token=0, image_size=224, patch_size=16):
    """
    Extract attention map for a single head at a single layer.
    
    Args:
        attn_weights: list of tensors, each [batch, num_heads, seq_len, seq_len]
        layer_idx: which layer to visualize
        head_idx: which head to visualize
        query_token: token index to query FROM (0 = CLS token)
        image_size: original image resolution
        patch_size: ViT patch size
    
    Returns:
        heatmap: np.ndarray of shape [image_size, image_size]
    """
    # Get attention at specified layer and head
    # Shape: [seq_len, seq_len]
    attn = attn_weights[layer_idx][0, head_idx]
    
    # Get attention FROM query_token TO all other tokens
    # Shape: [seq_len]
    attn_row = attn[query_token]
    
    # Remove CLS token (index 0), keep only patch tokens
    # Shape: [num_patches] = [196]
    patch_attn = attn_row[1:]
    
    # Reshape to 2D grid
    grid_size = int(patch_attn.shape[0] ** 0.5)  # 14
    heatmap = patch_attn.reshape(grid_size, grid_size).numpy()
    
    # Normalize to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Resize to image resolution
    heatmap = cv2.resize(heatmap, (image_size, image_size),
                         interpolation=cv2.INTER_LINEAR)
    
    return heatmap
```

### 1.3 Multi-Head Average Map

```python
def multi_head_average_map(attn_weights, layer_idx, query_token=0,
                            head_fusion='mean'):
    """
    Fuse across attention heads within a single layer.
    
    Args:
        attn_weights: list of tensors [batch, num_heads, seq_len, seq_len]
        layer_idx: which layer
        head_fusion: 'mean', 'max', or 'min'
    
    Returns:
        fused_attn: [seq_len, seq_len] tensor
    """
    attn = attn_weights[layer_idx]  # [batch, num_heads, seq_len, seq_len]
    
    if head_fusion == 'mean':
        fused = attn.mean(dim=1)        # [batch, seq_len, seq_len]
    elif head_fusion == 'max':
        fused = attn.max(dim=1)[0]      # [batch, seq_len, seq_len]
    elif head_fusion == 'min':
        fused = attn.min(dim=1)[0]      # [batch, seq_len, seq_len]
    
    # Extract CLS->patches attention and reshape
    patch_attn = fused[0, query_token, 1:]
    grid_size = int(patch_attn.shape[0] ** 0.5)
    heatmap = patch_attn.reshape(grid_size, grid_size).numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap
```

**Empirical findings** (from jacobgil's experiments):
- `mean` is what the original Attention Rollout paper suggests
- `min` removes noise by finding the common denominator across heads
- `max` combined with `discard_ratio` works best in practice

### 1.4 Multi-Layer Aggregation (Naive — NOT Rollout)

A simple approach: average attention maps across layers (not accounting for residual connections):

```python
def multi_layer_average(attn_weights, head_fusion='mean', query_token=0):
    """
    Simple average across all layers. Does NOT account for residual connections.
    Use Attention Rollout (Section 2) instead for proper multi-layer aggregation.
    """
    all_maps = []
    for layer_idx in range(len(attn_weights)):
        attn = attn_weights[layer_idx]
        if head_fusion == 'mean':
            fused = attn.mean(dim=1)
        elif head_fusion == 'max':
            fused = attn.max(dim=1)[0]
        elif head_fusion == 'min':
            fused = attn.min(dim=1)[0]
        
        patch_attn = fused[0, query_token, 1:]
        all_maps.append(patch_attn)
    
    # Simple average
    avg_map = torch.stack(all_maps).mean(dim=0)
    grid_size = int(avg_map.shape[0] ** 0.5)
    heatmap = avg_map.reshape(grid_size, grid_size).numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap
```

### 1.5 Text Token–Conditioned Cross-Attention Map

In VLMs with explicit cross-attention (e.g., LXMERT, Flamingo, decoder-based VLMs):

```python
def text_conditioned_cross_attention_map(cross_attn_weights, text_token_idx,
                                          num_image_patches, image_size=224):
    """
    Extract attention from a specific text token to visual patches.
    
    Args:
        cross_attn_weights: [batch, num_heads, text_seq_len, vis_seq_len]
            where Q=text tokens, K/V=visual tokens
        text_token_idx: which text token to visualize
        num_image_patches: number of visual tokens (e.g., 196 for 14x14)
        image_size: original image resolution
    
    Returns:
        heatmap: [image_size, image_size] attention map
    """
    # Average across heads
    # Shape: [text_seq_len, vis_seq_len]
    avg_attn = cross_attn_weights[0].mean(dim=0)
    
    # Get attention from the specified text token to all visual patches
    # Shape: [vis_seq_len]
    text_to_vis = avg_attn[text_token_idx]
    
    # Reshape to 2D and resize
    grid_size = int(num_image_patches ** 0.5)
    heatmap = text_to_vis.reshape(grid_size, grid_size).detach().cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = cv2.resize(heatmap, (image_size, image_size))
    return heatmap
```

For **self-attention VLMs** where text and image tokens are concatenated (e.g., VisualBERT), the attention matrix covers the joint `[text_tokens | image_patches]` sequence. Extract the text→image subblock:

```python
def cross_attention_from_joint_self_attention(attn_weights, layer_idx,
                                               text_token_idx, num_text_tokens,
                                               num_image_patches):
    """
    For models like VisualBERT where text and image share the same 
    self-attention. The attention matrix is:
    
        [text_to_text  | text_to_image ]
        [image_to_text | image_to_image]
    
    We extract text_to_image block for a specific text token.
    """
    attn = attn_weights[layer_idx][0].mean(dim=0)  # avg across heads
    
    # text token -> image patches block
    text_to_image = attn[text_token_idx, num_text_tokens:]
    
    grid_size = int(num_image_patches ** 0.5)
    heatmap = text_to_image[:num_image_patches].reshape(grid_size, grid_size)
    return heatmap.detach().cpu().numpy()
```

---

## Part 2: Attention Rollout

### 2.1 Theoretical Foundation

**Problem**: Raw attention weights at a single layer don't reflect the total flow of information across the network, because:
1. Information mixes across layers via matrix multiplication
2. Residual connections allow information to bypass attention layers

**Attention Rollout** (Abnar & Zuidema, 2020) accounts for both by:
1. Adding the identity matrix to each attention matrix (models residual connection)
2. Re-normalizing rows
3. Multiplying adjusted attention matrices across layers

**Formula**:

$$\hat{A}_l = 0.5 \cdot A_l + 0.5 \cdot I$$

$$\hat{A}_l = \frac{\hat{A}_l}{\hat{A}_l \cdot \mathbf{1}}$$

$$\text{Rollout}(L) = \hat{A}_L \cdot \hat{A}_{L-1} \cdots \hat{A}_1$$

The 0.5 factor comes from the residual: `output = attn(x) + x`, so half the information comes from attention, half from identity.

### 2.2 Visual Encoder Rollout (Self-Attention in ViT)

**Complete implementation** (from `jacobgil/vit-explain/vit_rollout.py`):

```python
import torch
import numpy as np

def rollout(attentions, discard_ratio=0.9, head_fusion='mean'):
    """
    Compute attention rollout across all layers of a ViT.
    
    Args:
        attentions: list of L tensors, each [batch, num_heads, seq_len, seq_len]
        discard_ratio: fraction of lowest attention values to zero out (0.0-1.0)
        head_fusion: how to combine heads - 'mean', 'max', 'min'
    
    Returns:
        mask: numpy array of shape [grid_size, grid_size], values in [0, 1]
    
    Algorithm:
        1. For each layer:
            a. Fuse attention heads (mean/max/min across head dim)
            b. Optionally discard lowest attention values (noise reduction)
            c. Add identity matrix (model residual connection)
            d. Re-normalize rows to sum to 1
            e. Multiply with running result
        2. Extract CLS token row, discard CLS self-attention
        3. Reshape to 2D grid
    """
    # Initialize: identity matrix (before any attention)
    result = torch.eye(attentions[0].size(-1))
    
    with torch.no_grad():
        for attention in attentions:
            # Step 1a: Fuse attention heads
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            
            # Step 1b: Discard lowest attentions (but keep CLS token)
            flat = attention_heads_fused.view(
                attention_heads_fused.size(0), -1
            )
            _, indices = flat.topk(
                int(flat.size(-1) * discard_ratio), -1, False  # lowest values
            )
            indices = indices[indices != 0]  # don't discard CLS
            flat[0, indices] = 0
            
            # Step 1c: Add identity (model residual connection)
            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2  # average with identity
            
            # Step 1d: Re-normalize rows to sum to 1
            a = a / a.sum(dim=-1)  # row-wise normalization
            
            # Step 1e: Matrix multiply with running result
            result = torch.matmul(a, result)
    
    # Step 2: Extract CLS token -> image patches attention
    # result[0, 0, :] is the CLS token's attention to all tokens
    # result[0, 0, 1:] removes self-attention to CLS
    mask = result[0, 0, 1:]
    
    # Step 3: Reshape to 2D grid
    width = int(mask.size(-1) ** 0.5)  # 14 for 224px/16px patches
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask
```

**Wrapper class** with hook registration:

```python
class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop',
                 head_fusion='mean', discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attentions = []
        
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self._get_attention)
    
    def _get_attention(self, module, input, output):
        self.attentions.append(output.cpu())
    
    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)
        return rollout(self.attentions, self.discard_ratio, self.head_fusion)
```

### 2.3 Gradient Attention Rollout (Class-Specific)

Gradient Attention Rollout weights attention heads by the gradient of the target class, enabling **class-specific** visualization:

```python
def grad_rollout(attentions, gradients, discard_ratio=0.9):
    """
    Gradient-weighted attention rollout for class-specific explainability.
    
    Key difference from vanilla rollout:
        Instead of fusing heads with mean/max/min, we weight each head's
        attention by the gradient of the target class w.r.t. that attention,
        then average. Negative attentions are masked out.
    
    Args:
        attentions: list of L tensors [batch, num_heads, seq_len, seq_len]
        gradients: list of L tensors [batch, num_heads, seq_len, seq_len]
                   (gradients of target class w.r.t. attention weights)
        discard_ratio: fraction of lowest values to discard
    
    Returns:
        mask: [grid_size, grid_size] numpy array
    """
    result = torch.eye(attentions[0].size(-1))
    
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):
            # Weight attention by gradient, then average across heads
            weights = grad
            attention_heads_fused = (attention * weights).mean(axis=1)
            
            # Remove negative attentions (don't contribute to target class)
            attention_heads_fused[attention_heads_fused < 0] = 0
            
            # Discard lowest attentions
            flat = attention_heads_fused.view(
                attention_heads_fused.size(0), -1
            )
            _, indices = flat.topk(
                int(flat.size(-1) * discard_ratio), -1, False
            )
            flat[0, indices] = 0
            
            # Add identity + normalize (same as vanilla rollout)
            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)
            result = torch.matmul(a, result)
    
    # Extract CLS -> patch attention
    mask = result[0, 0, 1:]
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask
```

### 2.4 Chefer et al. Method (Transformer Attribution)

From the CVPR 2021 paper — a cleaner formulation using gradient×attention as relevance:

```python
def avg_heads(cam, grad):
    """Rule 5: Gradient-weighted head averaging with ReLU."""
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)  # ReLU + mean across heads
    return cam

def apply_self_attention_rules(R_ss, cam_ss):
    """Rule 6: Self-attention update rule."""
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition

def generate_relevance(model, input, index=None):
    """
    Complete method from Chefer et al. (2021).
    
    Steps:
        1. Forward pass, collect attention weights per layer
        2. Backward pass on target class to get gradients
        3. For each layer: weight attention by gradient, ReLU, average heads
        4. Accumulate via matrix multiplication (rollout rule)
    """
    output = model(input, register_hook=True)
    
    if index is None:
        index = np.argmax(output.cpu().data.numpy(), axis=-1)
    
    # Backprop for target class
    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * output)
    model.zero_grad()
    one_hot.backward(retain_graph=True)
    
    # Get attention blocks
    blocks = list(dict(model.blocks.named_children()).values())
    num_tokens = blocks[0].attn.get_attn().shape[-1]
    
    # Initialize relevance as identity
    R = torch.eye(num_tokens, num_tokens, dtype=blocks[0].attn.get_attn().dtype).to(device)
    
    for blk in blocks:
        grad = blk.attn.get_attn_grad()
        cam = blk.attn.get_attn()
        cam = avg_heads(cam, grad)             # Rule 5
        R += apply_self_attention_rules(R, cam)  # Rule 6: R = R + cam @ R
    
    return R[0, 1:]  # CLS -> patches, excluding CLS self-attention
```

---

## Part 3: Cross-Attention Rollout for Multimodal Models

### 3.1 How Cross-Attention Rollout Differs from Self-Attention Rollout

In multimodal models with separate text and image streams (LXMERT, ViLBERT) or encoder-decoder models (DETR), we need to track **four relevance matrices**:

```
R_t_t: text → text    (text self-attention rollout)
R_i_i: image → image  (image self-attention rollout)
R_t_i: text → image   (how text tokens attend to image regions)
R_i_t: image → text   (how image regions attend to text tokens)
```

### 3.2 Cross-Attention Rollout Algorithm

From `hila-chefer/Transformer-MM-Explainability`:

```python
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    """
    Generic rollout computation for a sequence of attention matrices.
    
    Args:
        all_layer_matrices: list of [seq_len, seq_len] attention matrices
        start_layer: which layer to start from
    
    Returns:
        joint_attention: [seq_len, seq_len] aggregated attention
    """
    # Add residual consideration (identity matrix)
    num_tokens = all_layer_matrices[0].shape[1]
    eye = torch.eye(num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [
        all_layer_matrices[i] + eye 
        for i in range(len(all_layer_matrices))
    ]
    
    # Normalize rows
    all_layer_matrices = [
        all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
        for i in range(len(all_layer_matrices))
    ]
    
    # Sequential matrix multiplication
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer + 1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].matmul(joint_attention)
    
    return joint_attention
```

### 3.3 Complete Multimodal Rollout (LXMERT-style)

For models with separate text/image self-attention + cross-attention layers:

```python
class MultimodalRollout:
    """
    Tracks relevance flow through:
    1. Text self-attention layers
    2. Image self-attention layers  
    3. Cross-attention layers (text↔image)
    """
    
    def generate_rollout(self, model, input):
        output = model(input)
        
        text_tokens = num_text_tokens
        image_bboxes = num_image_patches
        
        # Initialize relevancy matrices
        R_t_t = torch.eye(text_tokens, text_tokens).to(device)
        R_i_i = torch.eye(image_bboxes, image_bboxes).to(device)
        R_t_i = torch.zeros(text_tokens, image_bboxes).to(device)
        R_i_t = torch.zeros(image_bboxes, text_tokens).to(device)
        
        cams_text = []
        cams_image = []
        
        # Process text self-attention layers
        for blk in model.text_self_attn_layers:
            cam = blk.attention.self.get_attn().detach()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
            cams_text.append(cam)
        
        # Process image self-attention layers
        for blk in model.image_self_attn_layers:
            cam = blk.attention.self.get_attn().detach()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
            cams_image.append(cam)
        
        # Process cross-attention layers
        for blk in model.cross_attn_layers:
            # Text self-attention within cross-attention block
            cam_text_self = blk.lang_self_att.self.get_attn().detach()
            cam_text_self = cam_text_self.reshape(-1, cam_text_self.shape[-2],
                                                   cam_text_self.shape[-1]).mean(dim=0)
            cams_text.append(cam_text_self)
            
            # Image self-attention within cross-attention block
            cam_img_self = blk.visn_self_att.self.get_attn().detach()
            cam_img_self = cam_img_self.reshape(-1, cam_img_self.shape[-2],
                                                 cam_img_self.shape[-1]).mean(dim=0)
            cams_image.append(cam_img_self)
        
        # Get cross-attention from last layer
        # text queries attending to image keys
        cam_t_i = model.cross_attn_layers[-1].visual_attention.att.get_attn().detach()
        cam_t_i = cam_t_i.reshape(-1, cam_t_i.shape[-2], cam_t_i.shape[-1]).mean(dim=0)
        
        # Compute self-attention rollout separately
        R_t_t = compute_rollout_attention(cams_text)
        R_i_i = compute_rollout_attention(cams_image)
        
        # Cross-attention: compose text rollout × cross-attn × image rollout
        # R_t_i = R_t_t^T @ cam_t_i @ R_i_i
        R_t_i = torch.matmul(
            R_t_t.t(),
            torch.matmul(cam_t_i, R_i_i)
        )
        
        # Zero out CLS self-attention
        R_t_t[0, 0] = 0
        
        return R_t_t, R_t_i
```

### 3.4 CLIP Visualization (Contrastive Model)

CLIP uses a ViT visual encoder with no explicit cross-attention. The "cross-modal" signal comes from the contrastive loss gradient flowing back through the visual encoder. From `hila-chefer/Transformer-MM-Explainability/CLIP/example.py`:

```python
def interpret_clip(image, text, model, device, index=None):
    """
    CLIP visualization using gradient-weighted attention rollout
    in the visual encoder.
    
    The gradient of the image-text similarity w.r.t. visual attention
    weights tells us which image regions are important for matching
    a specific text description.
    """
    logits_per_image, logits_per_text = model(image, text)
    
    if index is None:
        index = np.argmax(logits_per_image.cpu().data.numpy(), axis=-1)
    
    # Backprop for target text class
    one_hot = np.zeros((1, logits_per_image.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()
    one_hot.backward(retain_graph=True)
    
    # Collect attention + gradient from each visual transformer block
    image_attn_blocks = list(
        dict(model.visual.transformer.resblocks.named_children()).values()
    )
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens,
                  dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    
    for blk in image_attn_blocks:
        grad = blk.attn_grad
        cam = blk.attn_probs
        
        # Reshape and fuse heads with gradient weighting
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        
        # Accumulate: R = R + cam @ R
        R += torch.matmul(cam, R)
    
    # Remove CLS self-attention
    R[0, 0] = 0
    image_relevance = R[0, 1:]
    
    # Reshape to 2D and resize
    # For ViT-B/32 with 224 input: grid = 7×7
    grid_size = int(image_relevance.shape[0] ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, grid_size, grid_size)
    image_relevance = torch.nn.functional.interpolate(
        image_relevance, size=224, mode='bilinear'
    )
    image_relevance = image_relevance.reshape(224, 224).cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / \
                      (image_relevance.max() - image_relevance.min())
    
    return image_relevance
```

### 3.5 Encoder-Decoder Cross-Attention Rollout (DETR-style)

For encoder-decoder models where decoder queries attend to encoder image features:

```python
class EncoderDecoderRollout:
    """
    For models like DETR where:
    - Encoder: self-attention over image features
    - Decoder: self-attention over queries + cross-attention to image
    """
    
    def generate(self, model, image, target_index):
        outputs = model(image)
        
        decoder_blocks = model.transformer.decoder.layers
        encoder_blocks = model.transformer.encoder.layers
        
        image_bboxes = encoder_blocks[0].self_attn.get_attn().shape[-1]
        queries_num = decoder_blocks[0].self_attn.get_attn().shape[-1]
        
        # Initialize relevance matrices
        R_i_i = torch.eye(image_bboxes, image_bboxes).to(device)
        R_q_q = torch.eye(queries_num, queries_num).to(device)
        R_q_i = torch.zeros(queries_num, image_bboxes).to(device)
        
        # Process encoder self-attention (image-to-image)
        for blk in encoder_blocks:
            grad = blk.self_attn.get_attn_gradients().detach()
            cam = blk.self_attn.get_attn().detach()
            cam = avg_heads(cam, grad)
            R_i_i = torch.matmul(cam, R_i_i)  # accumulate
        
        # Process decoder blocks
        for blk in decoder_blocks:
            # Decoder self-attention (query-to-query)
            grad = blk.self_attn.get_attn_gradients().detach()
            cam = blk.self_attn.get_attn().detach()
            cam = avg_heads(cam, grad)
            R_q_q += torch.matmul(cam, R_q_q)
            
            # Decoder cross-attention (query-to-image)
            grad = blk.multihead_attn.get_attn_gradients().detach()
            cam = blk.multihead_attn.get_attn().detach()
            cam = avg_heads(cam, grad)
            R_q_i += torch.matmul(cam, R_i_i)  # compose with image rollout
        
        # Get relevance for target query
        aggregated = R_q_i[target_index, :].unsqueeze(0)
        return aggregated
```

---

## Part 4: Resizing and Heatmap Overlay

### 4.1 Resizing Attention Maps to Image Resolution

```python
import cv2
import torch.nn.functional as F

def resize_attention_map(attn_map, image_size, method='bilinear'):
    """
    Resize a patch-level attention map to full image resolution.
    
    Args:
        attn_map: numpy array [grid_h, grid_w] or torch tensor
        image_size: tuple (H, W) or int
        method: 'bilinear', 'nearest', 'bicubic'
    
    Returns:
        resized: numpy array [H, W]
    """
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    
    if isinstance(attn_map, np.ndarray):
        # OpenCV method
        interp = {
            'bilinear': cv2.INTER_LINEAR,
            'nearest': cv2.INTER_NEAREST,
            'bicubic': cv2.INTER_CUBIC,
        }[method]
        resized = cv2.resize(attn_map.astype(np.float32), 
                             (image_size[1], image_size[0]),
                             interpolation=interp)
    else:
        # PyTorch method (better for differentiable pipelines)
        attn_4d = attn_map.reshape(1, 1, *attn_map.shape)
        resized = F.interpolate(attn_4d, size=image_size, 
                                mode=method, 
                                align_corners=False if method != 'nearest' else None)
        resized = resized.squeeze().cpu().numpy()
    
    return resized
```

**Best practices for resizing:**
- **Bilinear** is the standard choice — smooth and fast
- **Bicubic** gives slightly smoother results but slower
- **Nearest** preserves exact patch boundaries (useful for analysis, not pretty for display)
- Use PyTorch's `F.interpolate` with `scale_factor=patch_size` for exact upscaling:
  ```python
  # For ViT with 14×14 patches -> 224×224
  attn_map = attn_map.reshape(1, 1, 14, 14)
  attn_map = F.interpolate(attn_map, scale_factor=16, mode='bilinear')
  ```

### 4.2 Overlaying Heatmaps on Images

**Standard method** (used across all major repos):

```python
def show_mask_on_image(img, mask, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """
    Overlay attention heatmap on the original image.
    
    Args:
        img: numpy array [H, W, 3], float32 in [0, 1] or uint8 in [0, 255]
        mask: numpy array [H, W], float32 in [0, 1]
        colormap: OpenCV colormap constant
        alpha: blending factor (0=only image, 1=only heatmap)
    
    Returns:
        visualization: numpy array [H, W, 3], uint8 in [0, 255]
    """
    # Ensure image is float32 in [0, 1]
    if img.dtype == np.uint8:
        img = np.float32(img) / 255.0
    
    # Ensure mask is in [0, 1]
    mask = np.clip(mask, 0, 1)
    
    # Apply colormap to mask
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    heatmap = np.float32(heatmap) / 255.0
    
    # Method 1: Simple additive blending (jacobgil's method)
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)  # normalize to prevent saturation
    return np.uint8(255 * cam)
    
    # Method 2: Alpha blending (more controllable)
    # cam = alpha * heatmap + (1 - alpha) * np.float32(img)
    # cam = np.clip(cam, 0, 1)
    # return np.uint8(255 * cam)
```

**Matplotlib-based overlay** (for Jupyter notebooks):

```python
import matplotlib.pyplot as plt

def show_attention_overlay_matplotlib(image, attention_map, 
                                       title='', cmap='jet', alpha=0.5):
    """
    Display attention overlay using matplotlib.
    
    Args:
        image: PIL Image or numpy array [H, W, 3]
        attention_map: numpy array [H, W], values in [0, 1]
        title: plot title
        cmap: matplotlib colormap name
        alpha: heatmap transparency
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Attention map only
    axes[1].imshow(attention_map, cmap=cmap)
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(attention_map, cmap=cmap, alpha=alpha)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig
```

---

## Part 5: Color Map Choices and Visualization Best Practices

### 5.1 Recommended Colormaps

| Colormap | Use Case | Notes |
|----------|----------|-------|
| `jet` / `cv2.COLORMAP_JET` | Most common default | Blue→Green→Yellow→Red. High contrast but perceptually uneven. Used by all major repos. |
| `viridis` | Perceptually uniform | Best for scientific visualization. Blue→Green→Yellow. |
| `inferno` / `magma` | Perceptually uniform, high contrast | Dark→Orange→Yellow. Good for dark backgrounds. |
| `hot` | Intensity focus | Black→Red→Yellow→White. Good for highlighting strong activations. |
| `coolwarm` | Diverging (positive/negative) | Blue→White→Red. Good for showing attention above/below average. |
| `turbo` | High contrast, perceptually improved | Improved version of jet. |

**OpenCV colormaps** (for `cv2.applyColorMap`):
```python
cv2.COLORMAP_JET        # Most popular, standard in papers
cv2.COLORMAP_VIRIDIS    # Perceptually uniform
cv2.COLORMAP_INFERNO    # High contrast
cv2.COLORMAP_HOT        # Thermal
cv2.COLORMAP_TURBO      # Improved jet
```

### 5.2 Visualization Best Practices

1. **Normalization**: Always normalize to [0, 1] before applying colormap:
   ```python
   mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
   ```

2. **Gaussian smoothing** (optional, for cleaner results):
   ```python
   from scipy.ndimage import gaussian_filter
   mask = gaussian_filter(mask, sigma=1.0)
   ```

3. **Thresholding** to highlight top-k attention:
   ```python
   threshold = np.percentile(mask, 90)  # top 10%
   mask[mask < threshold] = 0
   ```

4. **Discard ratio** (from jacobgil's experiments): Set the lowest `discard_ratio` fraction of attention values to zero before rollout. Values of 0.8-0.95 work well empirically. This is crucial for clean results.

5. **Side-by-side comparison**: Always show original image alongside the heatmap.

6. **Multi-head visualization**: Show individual heads in a grid to understand what each head attends to:
   ```python
   fig, axes = plt.subplots(3, 4, figsize=(16, 12))
   for head_idx in range(num_heads):
       ax = axes[head_idx // 4, head_idx % 4]
       ax.imshow(head_maps[head_idx], cmap='viridis')
       ax.set_title(f'Head {head_idx}')
       ax.axis('off')
   ```

7. **Color space**: OpenCV uses BGR by default. Convert to RGB for matplotlib:
   ```python
   vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
   ```

8. **Smoothing methods** (from pytorch-grad-cam):
   - `aug_smooth=True`: Test-time augmentation (horizontal flips + scale [0.9, 1.0, 1.1]), averages 6 forward passes. Better centering around objects.
   - `eigen_smooth=True`: First principal component of activations×weights. Reduces noise significantly.

---

## Summary: Method Comparison

| Method | Multi-Layer | Class-Specific | Cross-Modal | Complexity |
|--------|-------------|----------------|-------------|------------|
| Raw attention (single layer) | No | No | N/A | O(1) |
| Multi-head average | No | No | N/A | O(H) |
| Multi-layer average | Yes (naive) | No | No | O(L) |
| Attention Rollout | Yes | No | No | O(L·N²) |
| Gradient Rollout | Yes | Yes | No | O(L·N² + backprop) |
| Chefer et al. (Transformer Attribution) | Yes | Yes | No | O(L·N² + backprop) |
| Cross-Attention Rollout | Yes | Depends | Yes | O(L·N² + backprop) |
| CLIP Gradient Rollout | Yes | Yes (text-conditioned) | Implicit | O(L·N² + backprop) |

**Recommended approach**: 
- For **quick visualization**: Attention Rollout with `discard_ratio=0.9, head_fusion='max'`
- For **class-specific**: Gradient Attention Rollout or Chefer et al.
- For **multimodal text→image**: CLIP-style gradient rollout or explicit cross-attention rollout
- For **production/evaluation**: pytorch-grad-cam library with metrics (ROAD, faithfulness)
