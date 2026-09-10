# DLLM Tutorial

本教程将深入讲解基于FastDLLM框架的DLLM（深度学习语言模型）推理过程。重点介绍**双向注意力机制**、**单步多token解码**以及**双向KV缓存**的实现原理和技术细节。

### 1. DLLM原生推理过程

DLLM的推理过程结合了双向注意力机制与多token解码。与传统的自回归推理方法相比，DLLM能够更高效地生成序列。其核心优势在于每一步解码不仅利用之前的token，还能同时参考后续token，从而加快推理过程。

#### 1.1 双向注意力机制

DLLM的关键特性之一是使用**双向自注意力机制**。与传统的自回归模型不同，DLLM在生成过程中允许每个token同时关注序列中的所有其他token，无论这些token位于当前token之前还是之后。

具体来说：

- DLLM的自注意力机制**没有使用因果掩码**（causal mask），因此在计算每个token的表示时，它会考虑整个序列的所有token，而不仅仅是之前的token。
- **双向注意力**是通过应用多头自注意力机制实现的，保证每个token在生成过程中可以查看到整个序列。

以下是双向注意力的核心代码实现：

```python
def attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    attention_bias: Optional[torch.Tensor] = None,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    replace_position: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    B, T, C = q.size()  # batch size, sequence length, d_model
    dtype = k.dtype

    # Optionally apply layer norm to keys and queries.
    if self.q_norm is not None and self.k_norm is not None: #self.q_norm: None, self.k_norm: None
        q = self.q_norm(q).to(dtype=dtype)
        k = self.k_norm(k).to(dtype=dtype)

    # Move head forward to be next to the batch dim.
    # shape: (B, nh, T, hs)
    # self.config.n_heads: 32
    q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
    # shape: (B, n_kv_h, T, hs)
    k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
    # shape: (B, n_kv_h, T, hs)
    v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

    present = (k, v) if use_cache else None #present: None
    query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None

    if self.config.rope:
        # Apply rotary embeddings.
        if replace_position is None:
            q, k = self.rotary_emb(q, k)
        else:
            # For batched replace_position, use the maximum position across all batches
            max_replace_pos = replace_position.nonzero(as_tuple=True)[1].max() + 1 if replace_position.any() else key_len
            q, k = self.rotary_emb(q, k, max_replace_pos)

    # Get the attention scores.
    # shape: (B, nh, T, hs)
    att = self._scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0 if not self.training else self.config.attention_dropout,
        is_causal=False,
    )
    # Re-assemble all head outputs side-by-side.
    att = att.transpose(1, 2).contiguous().view(B, T, C)

    # Apply output projection.
    return self.attn_out(att), present
```

#### 1.2 单步多token解码

DLLM的另一个重要特性是能够**在一次推理中生成多个token**。这得益于**`generate`**函数的实现，该函数通过按块生成token来加速推理。

`get_transfer_index`函数在此过程中起到了关键作用。它负责控制每一步生成的token，并根据置信度或其他标准选择需要更新的位置。通过这种方式，DLLM可以在单次推理中生成多个token。

**核心代码：**

```python
def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x) # 将掩码位置的token替换为模型预测的token
    confidence = torch.where(mask_index, x0_p, -np.inf) # 将非掩码位置的token替换为-inf

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        # 如果是阈值的话，会将k提高到当前块内尚为掩码的token数量
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        # j是batch size
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            # 保证至少有一个token被揭示
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.):
    '''
    完整的生成过程，逐块生成token。
    `block_length`是每块生成的token数，`steps`是每块的解码步数。
    `gen_length`是生成的总长度。
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length  # 计算总共需要多少个块

    steps_per_block = steps // num_blocks  # 每个块的解码步数

    nfe = 0
    for num_block in range(num_blocks):  # 对每个块进行生成
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)  # 计算每一步需要生成多少个token

        i = 0
        while True:
            nfe += 1  # 每次迭代都会增加计算量
            mask_index = (x == mask_id)  # 查找mask位置
            logits = model(x).logits  # 获取模型的logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0  # 避免更新超出块的部分

            x0, transfer_index = get_transfer_index(logits, temperature, 'low_confidence', mask_index, x, num_transfer_tokens[:, i])
            x[transfer_index] = x0[transfer_index]  # 更新x中的token
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x, nfe
```

### 2. 双向KV缓存实现

DLLM中的**双向KV缓存**技术用于加速推理过程，通过缓存每个token的键值对（key-value pair），在生成新的token时可以复用之前计算过的结果，而不需要重新计算整个序列。

这种缓存机制不仅能提高生成速度，还能够保留整个序列的上下文信息，从而更好地捕捉长距离依赖关系。

KV缓存管理的核心代码实现如下：

```python
if layer_past is not None:
    past_key, past_value = layer_past
    if replace_position is None:
        k = torch.cat((past_key, k), dim=-2)
        v = torch.cat((past_value, v), dim=-2)
    else:
        # 根据replace_position掩码替换缓存中的key和value
        B = replace_position.shape[0]
        for batch_idx in range(B):
            batch_replace_indices = replace_position[batch_idx].nonzero(as_tuple=True)[0]
            if len(batch_replace_indices) > 0:
                past_key[batch_idx, :, batch_replace_indices] = k[batch_idx, :, :len(batch_replace_indices)]
                past_value[batch_idx, :, batch_replace_indices] = v[batch_idx, :, :len(batch_replace_indices)]
                
        k = past_key
        v = past_value
```

这段代码负责管理和更新KV缓存。如果当前token需要更新，它会将新的key和value与缓存中的旧值进行合并或替换，从而避免每次推理都重新计算。

# 3. 稀疏Attention简易实现

在 DLLM 的双向注意力机制基础上，我们可以引入**稀疏 Attention**进一步优化了推理效率。

#### 3.1 稀疏 Attention 核心思路

我们这里实现一个建议版本的稀疏 Attention，其核心是 **“聚焦关键上下文”**：不让每个查询 token 关注所有键 token，而是仅选择对当前生成任务最重要的部分。结合 DLLM 的双向特性与块生成逻辑，实现思路如下：

1. **拆分上下文**：将完整序列的键值对（k/v）拆分为两部分 —— 当前正在生成的`current_block`（当前块）和其余的`external`（外部块）。
2. **筛选重要外部块**：通过计算查询与外部块的相似度，筛选出 Top-K 个最相关的外部 token，避免无差别关注所有历史 / 未来上下文。
3. **拼接关键上下文**：将筛选后的外部块与当前块的 k/v 拼接，仅基于这些关键上下文计算注意力，减少无效计算。

#### 3.2 稀疏Attention实现流程

##### 3.2.1 输入处理与 KV 缓存集成

稀疏 Attention 无缝衔接前文的双向 KV 缓存机制，在位置编码（Rotary Embeddings）完成后执行筛选逻辑：

- 首先通过`index_select`从编码后的 k/v 中，截取当前块的索引（`current_block_indices`）对应的`current_block_k`和`current_block_v`。
- 若未指定当前块索引或不启用稀疏模式，默认初始化空张量，确保代码兼容性。

```python
if current_block_indices is not None:
    current_block_k = k.index_select(dim=2, index=current_block_indices)
    current_block_v = v.index_select(dim=2, index=current_block_indices)
```

##### 3.2.2 外部块相关性评分

为了筛选重要的外部上下文，需要计算查询与外部块的相似度：

- 先对查询 q 在序列维度（dim=2）求平均，得到`q_avg`（形状：[B, nh, 1, hs]），用全局查询特征代表当前块的关注需求。
- 通过矩阵乘法`torch.matmul(q_avg, k.transpose(-2, -1))`计算`q_avg`与所有 k 的相似度得分，再对批次、头维度求和，得到每个 token 的全局相关性得分`total_scores`。

```python
q_avg = q.mean(dim=2, keepdim=True)
scores = torch.matmul(q_avg, k.transpose(-2, -1))
total_scores = scores.sum(dim=[0, 1, 2])
```

##### 3.2.3 Top-K 外部块筛选

基于相关性得分筛选关键外部 token：

- 构建`external_indices_mask`掩码，排除当前块的索引，仅保留外部块的 token。
- 从外部块得分中选取前 50%（`num_external_to_keep = min(int(0.5 * scores_external.shape[0]), ...)`）的 token。（这个比例可以自行调整，也可以设置为参数，从外部传入）
- 对筛选出的索引排序，保证序列顺序一致性，避免上下文混乱。

```python
external_indices_mask = torch.ones_like(total_scores, dtype=torch.bool)
external_indices_mask[current_block_indices] = False
scores_external = total_scores[external_indices_mask]

if num_external_to_keep > 0:
    _, top_external_relative_indices = torch.topk(scores_external, k=num_external_to_keep)
    original_indices = torch.arange(total_scores.shape[0], device=q.device)
    sparse_context_indices = original_indices[external_indices_mask][top_external_relative_indices]
    sparse_context_indices, _ = torch.sort(sparse_context_indices)
else:
    sparse_context_indices = torch.tensor([], dtype=torch.long, device=q.device)
```

##### 3.2.4 稀疏注意力计算

最后基于筛选后的关键上下文计算注意力：

- 用筛选出的`sparse_context_indices`截取对应的`k_external`和`v_external`。
- 将外部关键块与当前块的 k/v 拼接（`torch.cat([k_external, current_block_k], dim=2)`），形成最终的稀疏上下文。
- 调用`_scaled_dot_product_attention`，仅基于稀疏上下文计算注意力，完成从 “全量关注” 到 “精准关注” 的转换。

```python
k_external = k.index_select(dim=2, index=sparse_context_indices)
v_external = v.index_select(dim=2, index=sparse_context_indices)

k_for_attn = torch.cat([k_external, current_block_k], dim=2)
v_for_attn = torch.cat([v_external, current_block_v], dim=2)

att = self._scaled_dot_product_attention(
    q,
    k_for_attn,
    v_for_attn,
    attn_mask=None,
    dropout_p=0.0 if not self.training else self.config.attention_dropout,
    is_causal=False,
)
```

##### 完整代码

```python
def attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    attention_bias: Optional[torch.Tensor] = None,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    replace_position: Optional[torch.Tensor] = None,
    use_sparse: bool = False,
    current_block_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    B, T, C = q.size()  # batch size, sequence length, d_model
    dtype = k.dtype

    # 1. 可选的查询/键层归一化（当前配置下未启用）
    if self.q_norm is not None and self.k_norm is not None:
        q = self.q_norm(q).to(dtype=dtype)
        k = self.k_norm(k).to(dtype=dtype)

    # 2. 多头注意力维度调整（[B, T, C] → [B, nh, T, hs]）
    q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
    k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
    v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
    
    # 3. 双向KV缓存更新（复用历史/未来上下文，减少重复计算）
    if layer_past is not None: 
        past_key, past_value = layer_past
        if replace_position is None:
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        else:
            # 按replace_position掩码替换缓存中需要更新的位置
            B = replace_position.shape[0]
            for batch_idx in range(B):
                batch_replace_indices = replace_position[batch_idx].nonzero(as_tuple=True)[0]
                if len(batch_replace_indices) > 0:
                    past_key[batch_idx, :, batch_replace_indices] = k[batch_idx, :, :len(batch_replace_indices)]
                    past_value[batch_idx, :, batch_replace_indices] = v[batch_idx, :, :len(batch_replace_indices)]
            k = past_key
            v = past_value

    present = (k, v) if use_cache else None
    query_len, key_len = q.shape[-2], k.shape[-2]

    # 4. 旋转位置编码（为q/k添加位置信息，保证时序相关性）
    if self.config.rope:
        if replace_position is None:
            q, k = self.rotary_emb(q, k)
        else:
            max_replace_pos = replace_position.nonzero(as_tuple=True)[1].max() + 1 if replace_position.any() else key_len
            q, k = self.rotary_emb(q, k, max_replace_pos)

    # 5. 注意力偏置处理（适配AMP混合精度训练，避免NaN）
    if attention_bias is not None:
        attention_bias = self._cast_attn_bias(
            attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
        )

    # -------------------------- 稀疏Attention核心逻辑 --------------------------
    if use_sparse:
        # 5.1 截取当前块的k/v（从编码后的k/v中获取当前生成块的上下文）
        if current_block_indices is not None:
            current_block_k = k.index_select(dim=2, index=current_block_indices)  # [B, n_kv_h, block_len, hs]
            current_block_v = v.index_select(dim=2, index=current_block_indices)
        else:
            current_block_k = torch.empty(0, device=k.device, dtype=k.dtype)
            current_block_v = torch.empty(0, device=v.device, dtype=v.dtype)
        
        # 5.2 计算查询与所有键的全局相关性得分
        q_avg = q.mean(dim=2, keepdim=True)  # 对查询序列维度求平均，得到全局查询特征 [B, nh, 1, hs]
        scores = torch.matmul(q_avg, k.transpose(-2, -1))  # 相似度计算 [B, nh, 1, T]
        total_scores = scores.sum(dim=[0, 1, 2])  # 聚合所有批次和头的得分 [T]
        
        # 5.3 筛选Top-K重要外部块（排除当前块，仅保留最相关的外部上下文）
        external_indices_mask = torch.ones_like(total_scores, dtype=torch.bool)
        external_indices_mask[current_block_indices] = False  # 排除当前块索引
        scores_external = total_scores[external_indices_mask]  # 仅保留外部块得分
        num_external_to_keep = min(int(0.5 * scores_external.shape[0]), scores_external.shape[0])  # 筛选前50%
        
        if num_external_to_keep > 0:
            _, top_external_relative_indices = torch.topk(scores_external, k=num_external_to_keep)
            original_indices = torch.arange(total_scores.shape[0], device=q.device)
            sparse_context_indices = original_indices[external_indices_mask][top_external_relative_indices]
            sparse_context_indices, _ = torch.sort(sparse_context_indices)  # 排序保证序列一致性
        else:
            sparse_context_indices = torch.tensor([], dtype=torch.long, device=q.device)
        
        # 5.4 截取稀疏上下文并计算注意力
        k_external = k.index_select(dim=2, index=sparse_context_indices)  # 筛选后的外部块k
        v_external = v.index_select(dim=2, index=sparse_context_indices)  # 筛选后的外部块v
        k_for_attn = torch.cat([k_external, current_block_k], dim=2)  # 拼接外部关键块+当前块
        v_for_attn = torch.cat([v_external, current_block_v], dim=2)
        
        att = self._scaled_dot_product_attention(
            q, k_for_attn, v_for_attn,
            attn_mask=None,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=False,  # 保持双向注意力特性
        )
    else:
        # 非稀疏模式：使用全量上下文计算双向注意力
        att = self._scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=False,
        )
    # --------------------------------------------------------------------------

    # 6. 多头注意力输出重组与投影
    att = att.transpose(1, 2).contiguous().view(B, T, C)
    return self.attn_out(att), present
```

#### 3.3 稀疏前后结果比对

```bash
### 输入
prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
out = generate_with_dual_cache(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking='low_confidence')    

### 稀疏前
--- generate_with_origin_dLLM execution time: 3.5685 seconds ---
Lily can run 12 kilometers per hour for 4 hours, so she runs a total of 12 kilometers per hour x 4 hours = 48 kilometers.
After that, she runs 6 kilometers per hour for the remaining 4 hours, so she runs a total of 6 kilometers per hour x 4 hours = 24 kilometers.
Therefore, Lily can run a total of 48 kilometers + 24 kilometers = 72 kilometers in 8 hours.
Conclusively: 72

### 稀疏后
--- generate_with_origin_dLLM execution time: 5.0209 seconds ---
Lily can run 12 kilometers per hour for 4 hours, so she runs 12 * 4 = 48 kilometers.
After that, she runs 6 kilometers per hour, so she runs 6 * 4 = 24 kilometers.
The final result is 72
```



### 4. 课后实践

1. **调整稀疏方法，比较不同稀疏方案下的推理精度和速度**：可以参考如下方案，各位同学也可以发挥想象力和创造力进行开发和实现。

   1. 仅针对prompt进行稀疏
   2. 单个解码块内采用同一个稀疏索引（即仅在当前解码块的第一步进行稀疏索引的选取，后面每一步沿用该索引，类似于sparse-dllm：https://arxiv.org/abs/2508.02558 ）
   3. 浅层不进行稀疏，深层才开始稀疏
   4. 不同层引入不同的稀疏比例
   5. 不同解码步骤引入不同的稀疏比例

2. （可选）**优化代码实现**：通过算子融合，批处理兼容等方案优化稀疏Attention

   1. 实现算子融合，可以基于Triton实现稀疏访存和attention计算两个算子的融合以加速推理
   2. 批处理兼容，目前的实现可能不兼容批次推理，可以尝试在当前代码的基础上实现批处理并与原生推理方案比较速度和精度。

   
