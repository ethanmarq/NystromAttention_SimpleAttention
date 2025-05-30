"""
This is Nystorm Attention with RRLS
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
# ================================================
#  PyTorch Kernel Function (Gaussian/RBF) 
# ================================================
def _torch_gauss_kernel(X: torch.Tensor, Y: torch.Tensor = None, gamma: float = 0.01):

    X = X.float() 
    if Y is None:
        return torch.ones(X.shape[0], device=X.device, dtype=X.dtype)
    else:
        Y = Y.float()
        nsq_rows = torch.sum(X**2, dim=1, keepdim=True)
        nsq_cols = torch.sum(Y**2, dim=1, keepdim=True)
        Ksub = nsq_rows - 2 * torch.matmul(X, Y.T) + nsq_cols.T
        Ksub = torch.clamp(Ksub, min=0.0)
        return torch.exp(-gamma * Ksub)

# ================================================
# Recursive Nystrom PyTorch Implementation 
# ================================================
def _recursive_nystrom_pytorch(
    X: torch.Tensor,
    n_components: int,
    kernel_func,
    lmbda_0: float = 1e-6,
    random_seed: int = None,
    return_leverage_score: bool = False
):

    N = X.shape[0]
    device = X.device
    dtype = X.dtype

    if n_components <= 0:
        print("Warning: n_components <= 0. Returning empty tensor.") # debug
        empty_indices = torch.tensor([], dtype=torch.long, device=device)
        if return_leverage_score:
            empty_scores = torch.tensor([], dtype=dtype, device=device)
            return empty_indices, empty_scores
        return empty_indices
        
    if N == 0 : # If input X itself is empty
        print("Warning: Input X is empty. Returning empty tensor.") # debug
        empty_indices = torch.tensor([], dtype=torch.long, device=device)
        if return_leverage_score:
            empty_scores = torch.tensor([], dtype=dtype, device=device)
            return empty_indices, empty_scores
        return empty_indices

    if n_components >= N:
        print(f"Warning: n_components ({n_components}) >= N ({N}). Returning all indices.") # debug
        indices = torch.arange(N, device=device)
        if return_leverage_score:
            scores = torch.ones(N, device=device, dtype=dtype) * (n_components / N) # Or just 1s
            return indices, scores
        return indices

    if random_seed is not None:
        generator = torch.Generator(device=device).manual_seed(random_seed)
    else:
        generator = None

    log_n_components = torch.log(torch.tensor(n_components, dtype=dtype, device=device))
    n_oversample = torch.clamp(log_n_components, min=0.1).item()
    k = max(1, int(torch.ceil(torch.tensor(n_components / (4 * n_oversample))).item()))
    log2_ratio = torch.log2(torch.tensor(N / n_components, dtype=dtype, device=device))
    n_levels = max(0, int(torch.ceil(log2_ratio).item()))

    perm = torch.randperm(N, device=device, generator=generator)
    
    size_list = [N]
    for _ in range(n_levels):
        next_size = int(torch.ceil(torch.tensor(size_list[-1] / 2.0)).item())
        size_list.append(max(1, next_size))

    initial_sample_size = min(size_list[-1], N)
    if initial_sample_size <= 0: # Should be caught by N==0 earlier, but as safeguard
        print("Warning: Calculated initial sample size is non-positive.") # debug
        empty_indices = torch.tensor([], dtype=torch.long, device=device)
        if return_leverage_score:
            empty_scores = torch.tensor([], dtype=dtype, device=device)
            return empty_indices, empty_scores
        return empty_indices
        
    sample_in_smallest_subset = torch.arange(initial_sample_size, device=device)
    indices = perm[sample_in_smallest_subset]
    weights = torch.ones(indices.shape[0], device=device, dtype=dtype)

    try:
        k_diag = kernel_func(X, None).squeeze()
        if k_diag.shape[0] != N:
            raise ValueError(f"kernel_func(X, None) returned shape {k_diag.shape}, expected ({N},)")
        if not torch.isfinite(k_diag).all():
            print("Warning: Non-finite values in kernel diagonal. Clamping.") # debug
            k_diag = torch.nan_to_num(k_diag, nan=1.0, posinf=1.0, neginf=0.0)
    except Exception as e:
        raise RuntimeError(f"Error calling kernel_func(X, None): {e}")

    for l in reversed(range(n_levels + 1)):
        if indices.numel() == 0:
            print(f"Error: Landmark set became empty at level {l}. Returning empty.") # debug
            final_empty_indices = torch.tensor([], dtype=torch.long, device=device)
            if return_leverage_score:
                final_empty_scores = torch.zeros(N, device=device, dtype=dtype) # Return zero scores for all
                return final_empty_indices, final_empty_scores
            return final_empty_indices


        current_subset_size = min(size_list[l], N)
        if current_subset_size <= 0: continue
        current_indices_in_perm = perm[:current_subset_size]
        
        X_current = X[current_indices_in_perm, :]
        X_landmarks = X[indices, :]

        try:
            KS = kernel_func(X_current, X_landmarks)
            SKS = kernel_func(X_landmarks, X_landmarks)
            if not torch.isfinite(KS).all() or not torch.isfinite(SKS).all():
                print(f"Warning: Non-finite values in KS or SKS at level {l}. Clamping.") # debug
                KS = torch.nan_to_num(KS)
                SKS = torch.nan_to_num(SKS)
        except Exception as e:
            raise RuntimeError(f"Error calling kernel_func at level {l}: {e}")

        num_landmarks_in_sample = SKS.shape[0]
        current_k = min(k, num_landmarks_in_sample)
        lmbda_val = torch.tensor(1e-6, device=device, dtype=dtype)

        if current_k > 0 and num_landmarks_in_sample > 0:
            try:
                weighted_SKS = SKS * torch.outer(weights, weights)
                diag_SKS = torch.diag(SKS)
                trace_weighted_SKS = torch.sum(diag_SKS * (weights**2))
                if not torch.isfinite(weighted_SKS).all():
                    print("Warning: Non-finite values in weighted_SKS. Clamping.") # debug
                    weighted_SKS = torch.nan_to_num(weighted_SKS)
                if not torch.allclose(weighted_SKS, weighted_SKS.T, atol=1e-5):
                    print("Warning: weighted_SKS is not symmetric. Symmetrizing.") # debug
                    weighted_SKS = (weighted_SKS + weighted_SKS.T) / 2.0
                eigvals = torch.linalg.eigvalsh(weighted_SKS)
                if not torch.isfinite(eigvals).all():
                    print(f"Warning: Non-finite eigenvalues detected at level {l}. Using fallback lambda.") # debug
                    lmbda_val = torch.maximum(torch.tensor(lmbda_0 * num_landmarks_in_sample, device=device, dtype=dtype), trace_weighted_SKS / current_k)
                else:
                    sum_largest_k_eigvals = torch.sum(eigvals[-current_k:])
                    lmbda_calc = (trace_weighted_SKS - sum_largest_k_eigvals) / current_k
                    lmbda_calc = torch.clamp(lmbda_calc, min=0.0)
                    lmbda_val = torch.maximum(torch.tensor(lmbda_0 * num_landmarks_in_sample, device=device, dtype=dtype), lmbda_calc)
            except torch.linalg.LinAlgError: # Removed 'as e' as e is not used
                print(f"Warning: Eigenvalue computation failed at level {l}. Using fallback lambda.") # debug
                lmbda_val = torch.tensor(lmbda_0 * num_landmarks_in_sample + 1e-5, device=device, dtype=dtype)
            except Exception:
                print(f"Warning: Unexpected error during lambda calculation at level {l}. Using fallback.") # debug
                lmbda_val = torch.tensor(lmbda_0 * num_landmarks_in_sample + 1e-5, device=device, dtype=dtype)
        
        lmbda = torch.maximum(lmbda_val, torch.tensor(1e-8, device=device, dtype=dtype))
        leverage_score_values = None
        try:
            diag_reg = torch.diag(lmbda / torch.clamp(weights**2, min=1e-10))
            inv_term = torch.linalg.inv(SKS + diag_reg)
            R = torch.matmul(KS, inv_term)
            row_sums_R_KS = torch.sum(R * KS, dim=1)
            current_k_diag = k_diag[current_indices_in_perm]
            if current_k_diag.shape != row_sums_R_KS.shape:
                raise ValueError(f"Shape mismatch in RLS: k_diag {current_k_diag.shape} vs row_sums {row_sums_R_KS.shape}")
            leverage_score_unscaled = torch.clamp(current_k_diag - row_sums_R_KS, min=0.0)
            leverage_score_values = (1.0 / lmbda) * leverage_score_unscaled
            if not torch.isfinite(leverage_score_values).all():
                print(f"Warning: Non-finite leverage scores computed at level {l}. Clamping.") # debug
                leverage_score_values = torch.nan_to_num(leverage_score_values, nan=0.0, posinf=1.0)
        except torch.linalg.LinAlgError: # Removed 'as e'
            print(f"Warning: Linear algebra error during RLS computation at level {l}. Falling back to uniform.") # debug
            leverage_score_values = None
        except Exception:
            print(f"Warning: Unexpected error during RLS computation at level {l}. Falling back to uniform.") # debug
            leverage_score_values = None

        if leverage_score_values is None:
            leverage_score_values = torch.ones(current_subset_size, device=device, dtype=dtype) * (n_components / max(1, current_subset_size))

        if l == 0:
            p = torch.clamp(leverage_score_values, min=0.0)
            p_sum = torch.sum(p)
            if p_sum <= 1e-10:
                print("Warning: Final leverage scores sum to <= 0. Using uniform probabilities.") # debug
                p = torch.ones_like(p) / max(1, p.numel())
            else:
                p = p / p_sum
            
            final_n_components = min(n_components, p.numel())
            if final_n_components < n_components:
                print(f"Warning: Requested n_components ({n_components}) > available points ({p.numel()}). Sampling {final_n_components}.") # debug
            
            if p.numel() > 0 and final_n_components > 0:
                sampled_relative_indices = torch.multinomial(p, num_samples=final_n_components, replacement=False, generator=generator)
                final_indices = perm[sampled_relative_indices] 
            else:
                final_indices = torch.tensor([], dtype=torch.long, device=device)

            if return_leverage_score:
                final_leverage_scores = torch.zeros(N, device=device, dtype=dtype)
                final_leverage_scores[perm] = leverage_score_values
                return final_indices, final_leverage_scores
            return final_indices
        else:
            sampling_prob = torch.clamp(n_oversample * leverage_score_values, min=0.0, max=1.0)
            rand_vals = torch.rand(current_subset_size, device=device, dtype=dtype, generator=generator)
            sampled_relative_indices = torch.where(rand_vals < sampling_prob)[0]

            if sampled_relative_indices.numel() == 0:
                print(f"Warning: No points sampled via RLS at level {l}. Falling back to uniform.") # debug
                num_fallback_samples = min(max(1, n_components // (n_levels + 1 if n_levels > -1 else 1)), current_subset_size)
                if current_subset_size > 0:
                    fallback_perm = torch.randperm(current_subset_size, device=device, generator=generator)
                    sampled_relative_indices = fallback_perm[:num_fallback_samples]
                    sampling_prob.fill_(0.0)
                    if current_subset_size > 0 : # Avoid division by zero
                         sampling_prob[sampled_relative_indices] = num_fallback_samples / current_subset_size
                else:
                    sampled_relative_indices = torch.tensor([], dtype=torch.long, device=device)
            
            if sampled_relative_indices.numel() > 0:
                indices = current_indices_in_perm[sampled_relative_indices]
                sample_probs = sampling_prob[sampled_relative_indices]
                weights = 1.0 / torch.sqrt(torch.clamp(sample_probs, min=1e-10))
            else:
                indices = torch.tensor([], dtype=torch.long, device=device)
                weights = torch.tensor([], dtype=dtype, device=device)
    
    # Fallback return (should ideally be caught by the l==0 branch)
    print("Warning: Recursive Nystrom loop finished unexpectedly (reached end of function).") # debug
    final_fallback_indices = torch.tensor([], dtype=torch.long, device=device)
    if 'final_indices' in locals() and final_indices.numel() > 0: return final_indices
    elif indices.numel() > 0: return indices # Return current sample if loop exited early
    
    if return_leverage_score:
        final_fallback_scores = torch.zeros(N, device=device, dtype=dtype)
        return final_fallback_indices, final_fallback_scores
    return final_fallback_indices


"""
Nystrom Code
"""
class NystromSelfAttention(nn.Module):
    def __init__(self, embed_dim, d_k, d_v, num_landmarks, mask=False, CUDA=False, conv_kernel_size=None,
                 rrls_gamma=0.01, rrls_lmbda_0=1e-6): 
        super(NystromSelfAttention, self).__init__()
        self.query_embed = nn.Linear(embed_dim, d_k)
        self.key_embed = nn.Linear(embed_dim, d_k)
        self.value_embed = nn.Linear(embed_dim, d_v)
        self.d_k = d_k
        self.mask = mask
        self.num_landmarks = num_landmarks 
        self.dropout = nn.Dropout(0.1)
        self.CUDA = CUDA
        self.device = torch.device("cuda:0" if CUDA else "cpu")
        self.init_option = "original"

        self.rrls_gamma = rrls_gamma
        self.rrls_lmbda_0 = rrls_lmbda_0
        self.rrls_seed_counter = 0 # Optional: Line 289 and 290 must be uncommented

        self.use_conv = conv_kernel_size is not None
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(conv_kernel_size, 1),
                padding=(conv_kernel_size // 2, 0),
                bias=False
            ).to(self.device)

    def _sample_landmarks_rrls(self, tensor_3d, target_num_landmarks):
        batch_size = tensor_3d.shape[0]
        # seq_len_dim = tensor_3d.shape[1] 
        d_features = tensor_3d.shape[2]
        
        landmark_list = []
        # Optional
        current_seed = self.rrls_seed_counter if hasattr(self, 'rrls_seed_counter') else None
        if current_seed is not None: self.rrls_seed_counter +=1
            #

        for i in range(batch_size):
            current_item = tensor_3d[i]  # Shape: (seq_len, d_features)
            current_seq_len = current_item.shape[0]

            if current_seq_len == 0: # Handle empty sequence for this batch item
                selected_indices_item = torch.tensor([], dtype=torch.long, device=self.device)
            else:
                # RRLS handles n_components >= N by returning all N indices.
                # It also handles n_components == 0 or N == 0.
                selected_indices_item = _recursive_nystrom_pytorch(
                    X=current_item,
                    n_components=target_num_landmarks,
                    kernel_func=lambda x, y=None: _torch_gauss_kernel(x, y, gamma=self.rrls_gamma),
                    lmbda_0=self.rrls_lmbda_0,
                    random_seed=None, 
                    return_leverage_score=False
                )

            num_selected = selected_indices_item.shape[0]
            if num_selected > 0:
                landmarks_item = current_item[selected_indices_item, :]
            else: # No indices selected
                landmarks_item = torch.zeros((0, d_features), device=self.device, dtype=tensor_3d.dtype)

            # Pad if fewer landmarks were selected/available
            if num_selected < target_num_landmarks:
                padding_size = target_num_landmarks - num_selected
                padding = torch.zeros((padding_size, d_features), device=self.device, dtype=tensor_3d.dtype)
                landmarks_item = torch.cat([landmarks_item, padding], dim=0)
            elif num_selected > target_num_landmarks:
                landmarks_item = landmarks_item[:target_num_landmarks, :]
                
            landmark_list.append(landmarks_item)

        return torch.stack(landmark_list, dim=0)


    def forward(self, query_in, key_in, value_in):
        batch_size = query_in.size(0)
        q_seq_len = query_in.size(1)
        k_seq_len = key_in.size(1)
        
        query = self.query_embed(query_in)
        key = self.key_embed(key_in)
        value = self.value_embed(value_in)
        
        scaling = 1.0 / math.sqrt(math.sqrt(self.d_k))
        query = query * scaling
        key = key * scaling
        
        mask_tensor = None
        if self.mask:
            mask_tensor = torch.ones(batch_size, q_seq_len, k_seq_len, device=self.device) # For standard attention fallback
            mask_tensor = torch.tril(mask_tensor)

        is_decoder_self_attn = (q_seq_len == 1 and k_seq_len > 1)
        use_nystrom = True 
        
        # `actual_landmarks_dim` will be self.num_landmarks due to padding in _sample_landmarks_rrls
        actual_landmarks_dim = self.num_landmarks

        if use_nystrom and actual_landmarks_dim > 0 : # Nystrom only if landmarks are requested
            try:
                # RRLS Sampling for k_landmarks
                k_landmarks = self._sample_landmarks_rrls(key, actual_landmarks_dim) # Shape: (B, M, Dk)

                # RRLS Sampling for q_landmarks (or direct use if decoder_self_attn)
                if is_decoder_self_attn:
                    q_landmarks = query # query is (B, 1, Dk)
                else:
                    # This will make q_landmarks (B, M, Dk)
                    q_landmarks = self._sample_landmarks_rrls(query, actual_landmarks_dim)
                
                # Nystrom Kernels
                # kernel_1: Q K_L^T -> (B, Q_len, M)
                kernel_1 = torch.matmul(query, k_landmarks.transpose(-1, -2))
                kernel_1 = F.softmax(kernel_1, dim=-1)
                
                # kernel_2: Q_L K_L^T -> (B, Q_L_dim, M)
                # If is_decoder_self_attn, Q_L_dim = 1. Else Q_L_dim = M.
                kernel_2 = torch.matmul(q_landmarks, k_landmarks.transpose(-1, -2))
                kernel_2 = F.softmax(kernel_2, dim=-1)
                
                # kernel_3: Q_L K^T -> (B, Q_L_dim, K_len)
                kernel_3 = torch.matmul(q_landmarks, key.transpose(-1, -2))
                
                if self.mask:
                    # Determine Q_L_dim for masking kernel_3
                    q_l_dim_for_mask = 1 if is_decoder_self_attn else actual_landmarks_dim
                    
                    if is_decoder_self_attn: 
                        seq_mask_k3 = torch.ones(batch_size, 1, k_seq_len, device=self.device)
                        
                        _mask_k3 = torch.ones(batch_size, q_l_dim_for_mask, k_seq_len, device=self.device)
                        if q_seq_len == 1: 
                            
                             indices_to_zero = torch.arange(1, k_seq_len, device=self.device)
                             if indices_to_zero.numel() > 0:
                                _mask_k3[:, :, indices_to_zero] = 0
                        else: 
                            _mask_k3 = torch.tril(_mask_k3)
                        kernel_3 = kernel_3.masked_fill(_mask_k3 == 0, float('-inf'))

                    else: 
                        _mask_k3 = torch.ones(batch_size, actual_landmarks_dim, k_seq_len, device=self.device)
                        _mask_k3 = torch.tril(_mask_k3) # (B, M, K_len)
                        kernel_3 = kernel_3.masked_fill(_mask_k3 == 0, float('-inf'))
                
                kernel_3 = F.softmax(kernel_3, dim=-1)
                
                # Moore-Penrose pseudoinverse of kernel_2
                # kernel_2 shape: (B, Q_L_dim, M)
                if kernel_2.size(-2) == 0 or kernel_2.size(-1) == 0: 
                     raise RuntimeError("kernel_2 has zero dimension, cannot compute pinverse.")
                kernel_2_inv = torch.pinverse(kernel_2)

                k3_v = torch.matmul(kernel_3, value)
                k1_k2inv = torch.matmul(kernel_1, kernel_2_inv)
                attention_weighted_value = torch.matmul(k1_k2inv, k3_v)

            except RuntimeError as e:
                print(f"Error in RRLS Nystrom attention: {e}. Falling back to standard attention.") # debug
                # Fallback to standard attention
                use_nystrom = False 

        if not use_nystrom or actual_landmarks_dim == 0: # Fallback or no landmarks to use
            print(f"Using standard attention with q_seq_len={q_seq_len}, k_seq_len={k_seq_len}") # debug
            key_transposed = torch.transpose(key, 1, 2)
            attention_weights = torch.matmul(query, key_transposed)
            if self.mask and mask_tensor is not None: # mask_tensor is (B, Q_len, K_len)
                attention_weights = attention_weights.masked_fill(
                    mask_tensor == 0, float('-inf') 
                )
            attention_weights = F.softmax(attention_weights, dim=-1)
            attention_weighted_value = torch.matmul(attention_weights, value)
        
        if self.use_conv:
            pass 

        attention_weighted_value = self.dropout(attention_weighted_value)
        return attention_weighted_value
    
    def iterative_inv(self, mat, n_iter=6):
        I = torch.eye(mat.size(-1), device=mat.device)
        K = mat
        if self.init_option == "original":
            V = 1 / torch.max(torch.sum(K, dim=-2)) * K.transpose(-1, -2)
        else:
            V = 1 / torch.max(torch.sum(K, dim=-2), dim=-1).values[:, :, None, None] * K.transpose(-1, -2)
        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V


class NystromMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, d_k, d_v, num_heads, num_landmarks, mask=False, conv_kernel_size=None, CUDA=False,
                 rrls_gamma=0.01, rrls_lmbda_0=1e-6):
        super(NystromMultiHeadAttention, self).__init__()
        self.attention_blocks = nn.ModuleList(
            [NystromSelfAttention(embed_dim, d_k, d_v, num_landmarks, mask, CUDA, conv_kernel_size, rrls_gamma, rrls_lmbda_0) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(num_heads * d_v, embed_dim) # Output projection
        self.norm = LayerNorm(embed_dim) 
        self.CUDA = CUDA
        self.device = torch.device("cuda:0" if CUDA else "cpu")

    def forward(self, query, key, value, residual_x):
        # query, key, value: (B, SeqLen, embed_dim)
        # residual_x: (B, SeqLen_Q, embed_dim)
        
        head_outputs = []
        for attention in self.attention_blocks:
            head_outputs.append(attention(query, key, value)) # Each output is (B, Q_len, d_v)
        
        # Concatenate head outputs: (B, Q_len, num_heads * d_v)
        attention_out_concat = torch.cat(head_outputs, dim=2)
        
        # Project back to embed_dim
        attention_out_projected = self.output_linear(attention_out_concat) # (B, Q_len, embed_dim)
        
        if attention_out_projected.shape != residual_x.shape:
             pass

        add_and_norm = self.norm(attention_out_projected + residual_x)
        return add_and_norm


class NystromTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_landmarks, mask=False, conv_kernel_size=None, CUDA=False,
                 rrls_gamma=0.01, rrls_lmbda_0=1e-6):
        super(NystromTransformerBlock, self).__init__()
        self.multi_head_attention = NystromMultiHeadAttention(
            embed_dim,
            embed_dim // num_heads, # d_k per head
            embed_dim // num_heads, # 
            num_heads,
            num_landmarks,
            mask,
            conv_kernel_size,
            CUDA=CUDA,
            rrls_gamma=rrls_gamma,
            rrls_lmbda_0=rrls_lmbda_0
        )
        self.feed_forward = PositionWiseFeedForward(embed_dim, embed_dim) 

    def forward(self, query, key, value, residual_x):

        attention_out = self.multi_head_attention(query, key, value, residual_x)
        feed_forward_out = self.feed_forward(attention_out, attention_out)
        return feed_forward_out

"""
Non-Nystrom Code
"""

class LayerNorm(nn.Module):
    "Taken from Annotated Transformer (HarvardNLP)"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(features))
        self.shift = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.scale * (x - mean) / (std + self.eps) + self.shift


class PositionWiseFeedForward(nn.Module):
    def __init__(self, embed_dim, ff_hidden_dim, dropout=0.1): 
        super(PositionWiseFeedForward, self).__init__()
        self.l1 = nn.Linear(embed_dim, ff_hidden_dim)
        self.RELU = nn.ReLU() 
        self.l2 = nn.Linear(ff_hidden_dim, embed_dim)
        self.norm = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual_x): 
        out = self.l1(x)
        out = self.RELU(out)
        out = self.dropout(out) 
        out = self.l2(out)
        out = self.dropout(out)
        return self.norm(out + residual_x)


class VocabLogits(nn.Module):
    def __init__(self, embed_dim, logit_dim):
        super(VocabLogits, self).__init__()
        self.linear = nn.Linear(embed_dim, logit_dim)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)


class Embeddings(nn.Module):
    "Taken from Annotated Transformer (HarvardNLP)"
    def __init__(self, vocab_length, embed_dim, CUDA=False, max_len=5000): 
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_length, embed_dim)

        self.pos_encode = PositionalEncoding(embed_dim, max_len=max_len, CUDA=CUDA) 
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(0.1) # May need to modify

    def forward(self, x):
        embed = self.lut(x) * math.sqrt(self.embed_dim)
        return self.dropout(self.pos_encode(embed))


class PositionalEncoding(nn.Module):
    "Modified From Annotated Transformer (HarvardNLP)"
    def __init__(self, embed_dim, max_len=5000, CUDA=False): 
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout) # Optional
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0) # Shape: (1, max_len, embed_dim)
        if CUDA: 
            pe = pe.to(torch.device("cuda:0")) 
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        # return self.dropout(x) # Optional
        return x