# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLCausalLMOutputWithPast,
    Qwen2VLForConditionalGeneration,
)
from transformers.utils import is_flash_attn_greater_or_equal

from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_world_size,
    validate_ulysses_config,
)

def ulysses_self_flash_attn_forward(
    self,
    x: torch.Tensor,
    seq_lens, grid_sizes, freqs,  # will become mandatory in v4.46
    **kwargs,
):
    from wan.modules.model import rope_apply
    attention_mask=None
    # bsz, q_len, _ = x.size()  # q_len = seq_length / sp_size
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)

    f, h, w = grid_sizes[0,:]
    if ulysses_sp_size > 1:
        validate_ulysses_config(self.num_heads, ulysses_sp_size)
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)
        target = f*h*w
        q = gather_seq_scatter_heads(q, seq_dim=1, head_dim=2,unpadded_dim_size=target)
        k = gather_seq_scatter_heads(k, seq_dim=1, head_dim=2,unpadded_dim_size=target)
        v = gather_seq_scatter_heads(v, seq_dim=1, head_dim=2,unpadded_dim_size=target)
        #TODO:UNPAD
        # rank = torch.distributed.get_rank(group=sp_group)
        # world_size = torch.distributed.get_world_size(group=sp_group)

        # if rank == world_size - 1:
        #     padding_size = ulysses_sp_size- f*h*w % ulysses_sp_size
        #     q = _unpad_tensor(q, seq_dim=1, padding_size)
        #     k = _unpad_tensor(k, seq_dim=1, padding_size)
        #     v = _unpad_tensor(v, seq_dim=1, padding_size)
        # (batch_size, num_head / sp_size, seq_length, head_size)
        
        full_q_len = q.size(1)  # full_q_len = seq_length
    else:
        full_q_len = s

    import torch.nn.functional as F
    torch.backends.cuda.enable_cudnn_sdp(False)
    #unpad 

    q=rope_apply(q, grid_sizes, freqs)
    k=rope_apply(k, grid_sizes, freqs)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    attn_output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
    )  # b h s d

    if ulysses_sp_size > 1:
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=1, seq_dim=2)

    attn_output = attn_output.transpose(1, 2).flatten(2, 3).contiguous()
    attn_output = self.o(attn_output)
    return attn_output

_PAD_SIZE = None

def set_pad_size(pad_size):
    global _PAD_SIZE
    _PAD_SIZE=pad_size

def get_pad_size():
    return _PAD_SIZE
