import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

def torch_gauss_kernel(X: torch.Tensor, Y: torch.Tensor = None, gamma: float = 0.01):
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

def recursive_nystrom_pytorch(
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
        print("Warning: n_components <= 0. Returning empty tensor.")
        empty_indices = torch.tensor([], dtype=torch.long, device=device)
        if return_leverage_score:
            empty_scores = torch.tensor([], dtype=dtype, device=device)
            return empty_indices, empty_scores
        return empty_indices
    
    if n_components >= N:
        # print(f"Warning: n_components ({n_components}) >= N ({N}). Returning all indices.")
        all_indices = torch.arange(N, device=device)
        if return_leverage_score:
            # Leverage scores are typically 1 or k/n
            scores = torch.ones(N, device=device, dtype=dtype) * (n_components / N) if N > 0 else torch.ones(N, device=device, dtype=dtype)
            return all_indices, scores
        return all_indices

    generator = None
    if random_seed is not None:
        generator = torch.Generator(device=device).manual_seed(random_seed)

    log_n_components = torch.log(torch.tensor(n_components, dtype=dtype, device=device))
    n_oversample = torch.clamp(log_n_components, min=0.1).item()
    k = max(1, int(torch.ceil(torch.tensor(n_components / (4 * n_oversample), dtype=dtype, device=device)).item()))
    log2_ratio = torch.log2(torch.tensor(N / n_components, dtype=dtype, device=device))
    n_levels = max(0, int(torch.ceil(log2_ratio).item()))
    perm = torch.randperm(N, device=device, generator=generator)

    size_list = [N]
    for _ in range(n_levels):
        next_size = int(torch.ceil(torch.tensor(size_list[-1] / 2.0, dtype=dtype, device=device)).item())
        size_list.append(max(1, next_size))

    initial_sample_size = size_list[-1]
    initial_sample_size = min(initial_sample_size, N)
    assert initial_sample_size > 0, "Initial sample size is non-positive"

    sample_in_smallest_subset = torch.arange(initial_sample_size, device=device)
    indices = perm[sample_in_smallest_subset]
    weights = torch.ones(indices.shape[0], device=device, dtype=dtype)

    k_diag = kernel_func(X, None).squeeze()
    assert k_diag.shape[0] == N, f"k_diag shape {k_diag.shape} mismatch with N {N}"
    if not torch.isfinite(k_diag).all(): # Changed from assert to allow recovery
        print("Warning: Non-finite values in kernel diagonal. Clamping.")
        k_diag = torch.nan_to_num(k_diag, nan=1.0, posinf=1.0, neginf=0.0)


    for l in reversed(range(n_levels + 1)):
        assert indices.numel() > 0, f"Landmark set became empty at level {l}"

        current_subset_size = size_list[l]
        current_subset_size = min(current_subset_size, N)
        if current_subset_size <= 0: continue

        current_indices_in_perm = perm[:current_subset_size]
        X_current = X[current_indices_in_perm, :]
        X_landmarks = X[indices, :]

        try:
            KS = kernel_func(X_current, X_landmarks)
            SKS = kernel_func(X_landmarks, X_landmarks)
            if not (torch.isfinite(KS).all() and torch.isfinite(SKS).all()): # Changed from assert
                print(f"Warning: Non-finite values in KS or SKS at level {l}. Clamping.")
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

                if not torch.isfinite(weighted_SKS).all(): # Changed from assert
                    print("Warning: Non-finite values in weighted_SKS. Clamping.")
                    weighted_SKS = torch.nan_to_num(weighted_SKS)
                if not torch.allclose(weighted_SKS, weighted_SKS.T, atol=1e-5): # Changed from assert
                     print("Warning: weighted_SKS is not symmetric. Symmetrizing.")
                     weighted_SKS = (weighted_SKS + weighted_SKS.T) / 2.0

                eigvals = torch.linalg.eigvalsh(weighted_SKS)

                if not torch.isfinite(eigvals).all(): # Changed from assert
                    print(f"Warning: Non-finite eigenvalues detected at level {l}. Using fallback lambda.")
                    lmbda_val = torch.maximum(
                        torch.tensor(lmbda_0 * num_landmarks_in_sample, device=device, dtype=dtype),
                        trace_weighted_SKS / max(1, current_k) # Avoid division by zero
                    )
                else:
                    sum_largest_k_eigvals = torch.sum(eigvals[-current_k:])
                    lmbda_calc = (trace_weighted_SKS - sum_largest_k_eigvals) / max(1, current_k)
                    lmbda_calc = torch.clamp(lmbda_calc, min=0.0)
                    lmbda_val = torch.maximum(torch.tensor(lmbda_0 * num_landmarks_in_sample, device=device, dtype=dtype), lmbda_calc)
            except torch.linalg.LinAlgError as e:
                print(f"Warning: Eigenvalue computation failed at level {l}: {e}. Using fallback lambda.")
                lmbda_val = torch.tensor(lmbda_0 * num_landmarks_in_sample + 1e-5, device=device, dtype=dtype)
            except Exception as e:
                print(f"Warning: Unexpected error during lambda calculation at level {l}: {e}. Using fallback.")
                lmbda_val = torch.tensor(lmbda_0 * num_landmarks_in_sample + 1e-5, device=device, dtype=dtype)
        
        lmbda = torch.maximum(lmbda_val, torch.tensor(1e-8, device=device, dtype=dtype))
        leverage_score = None
        try:
            diag_reg = torch.diag(lmbda / torch.clamp(weights**2, min=1e-10))
            inv_term = torch.linalg.inv(SKS + diag_reg)
            R = torch.matmul(KS, inv_term)
            row_sums_R_KS = torch.sum(R * KS, dim=1)
            current_k_diag = k_diag[current_indices_in_perm]
            
            assert current_k_diag.shape == row_sums_R_KS.shape, \
                f"Shape mismatch in RLS: k_diag {current_k_diag.shape} vs row_sums {row_sums_R_KS.shape}"

            leverage_score_unscaled = torch.clamp(current_k_diag - row_sums_R_KS, min=0.0)
            leverage_score = (1.0 / lmbda) * leverage_score_unscaled
            
            if not torch.isfinite(leverage_score).all():
                print(f"Warning: Non-finite leverage scores computed at level {l}. Clamping.")
                leverage_score = torch.nan_to_num(leverage_score, nan=0.0, posinf=1.0)
        except torch.linalg.LinAlgError as e:
            print(f"Warning: Linear algebra error during RLS computation at level {l}: {e}. Falling back to uniform.")
            leverage_score = None
        except Exception as e:
            print(f"Warning: Unexpected error during RLS computation at level {l}: {e}. Falling back to uniform.")
            leverage_score = None
        
        if leverage_score is None:
            leverage_score = torch.ones(current_subset_size, device=device, dtype=dtype) * (n_components / max(1, current_subset_size))

        if l == 0:
            p = torch.clamp(leverage_score, min=0.0)
            p_sum = torch.sum(p)
            if p_sum <= 1e-10:
                print("Warning: Final leverage scores sum to <= 0. Using uniform probabilities.")
                p = torch.ones_like(p) / max(1, p.numel())
            else:
                p = p / p_sum
            
            final_n_components = min(n_components, p.numel())
            # if final_n_components < n_components:
            #     print(f"Warning: Requested n_components ({n_components}) > available points ({p.numel()}). Sampling {final_n_components}.")
            
            final_indices = torch.tensor([], dtype=torch.long, device=device)
            final_leverage_scores = torch.zeros(N, device=device, dtype=dtype)

            if p.numel() > 0 and final_n_components > 0:
                sampled_relative_indices = torch.multinomial(p, num_samples=final_n_components, replacement=False, generator=generator)
                final_indices = perm[sampled_relative_indices] # These are indices relative to the original X
                if return_leverage_score: # Map scores for the current perm back to original X positions
                    final_leverage_scores[perm[:current_subset_size]] = leverage_score # leverage_score corresponds to current_indices_in_perm

            if return_leverage_score:
                return final_indices, final_leverage_scores
            return final_indices
        else:
            sampling_prob = torch.clamp(n_oversample * leverage_score, min=0.0, max=1.0)
            rand_vals = torch.rand(current_subset_size, device=device, dtype=dtype, generator=generator)
            sampled_relative_indices = torch.where(rand_vals < sampling_prob)[0]

            if sampled_relative_indices.numel() == 0:
                # print(f"Warning: No points sampled via RLS at level {l}. Falling back to uniform.")
                num_fallback_samples = min(max(1, n_components // max(1, (n_levels + 1))), current_subset_size)
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
                # The assertion at the start of the loop will catch this if indices becomes empty.
    
    # Fallback if loop finishes unexpectedly (should not happen with the assertion at the start)
    print("Warning: Recursive Nystrom loop finished unexpectedly (should have returned in l==0).")
    # Return the last valid indices if available
    # This part is unlikely to be reached if the l==0 block always returns.
    final_fallback_indices = indices if 'indices' in locals() and indices.numel() > 0 else torch.arange(min(N, n_components), device=device)
    if return_leverage_score:
        final_fallback_scores = torch.ones(N, device=device, dtype=dtype) / N if N > 0 else torch.ones(N, device=device, dtype=dtype)
        return final_fallback_indices, final_fallback_scores
    return final_fallback_indices
# --- End of placeholder ---


class NystromSelfAttention(nn.Module):
    def __init__(self, embed_dim, d_k, d_v, num_landmarks, mask=False, CUDA=False, conv_kernel_size=None, rls_gamma=0.01, rls_random_seed=None):
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
        self.init_option = "original" # For iterative_inv, if used

        self.rls_gamma = rls_gamma # Gamma for Gaussian kernel in RLS
        self.rls_random_seed = rls_random_seed # Optional seed for RLS

        self.use_conv = conv_kernel_size is not None
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(conv_kernel_size, 1),
                padding=(conv_kernel_size // 2, 0),
                bias=False
            ).to(self.device)

    def forward(self, query_in, key_in, value_in):
        batch_size = query_in.size(0)
        q_seq_len = query_in.size(1)
        k_seq_len = key_in.size(1)

        query = self.query_embed(query_in)
        key = self.key_embed(key_in)
        value = self.value_embed(value_in)

        scaling = 1.0 / math.sqrt(math.sqrt(self.d_k)) # Or math.sqrt(self.d_k) for standard attention
        query = query * scaling
        key = key * scaling

        mask_tensor = None
        if self.mask:
            # Standard causal mask for QK^T
            mask_tensor_qk_t = torch.ones(q_seq_len, k_seq_len, device=self.device)
            mask_tensor_qk_t = torch.tril(mask_tensor_qk_t)
            # For kernel_3 (Q_lm K^T), the mask might need different dimensions if q_landmarks are involved
            # This depends on how q_landmarks are formed and how masking is applied to kernel_3

        is_decoder_self_attn = (q_seq_len == 1 and k_seq_len > 1)
        use_nystrom = True # Force Nystrom

        if not use_nystrom: # Standard Attention Fallback (kept for completeness)
            # print(f"Using standard attention with q_seq_len={q_seq_len}, k_seq_len={k_seq_len}")
            key_transposed = torch.transpose(key, 1, 2) # B x d_k x k_L
            attention_scores = torch.matmul(query, key_transposed) # B x q_L x k_L

            if self.mask and mask_tensor_qk_t is not None:
                attention_scores = attention_scores.masked_fill(mask_tensor_qk_t == 0, float('-inf'))
            
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weighted_value = torch.matmul(attention_weights, value)
        else:
            # Nystrom attention implementation with RLS
            try:
                # --- 1. Select Landmark Tokens using RLS ---
                actual_num_landmarks = self.num_landmarks
                
                # Detach keys for RLS sampling to prevent gradients from flowing back through the sampling process
                keys_for_sampling = key.detach() # Shape: [batch_size, k_seq_len, d_k]

                batched_landmark_indices = []
                # RLS is not batch-aware, so we loop.
                # Consider vectorizing RLS if performance becomes an issue, though it's complex.
                for i in range(batch_size):
                    # Pass key[i] which has shape [k_seq_len, d_k]
                    # Ensure k_seq_len > 0 for RLS to work
                    if k_seq_len == 0:
                         # Handle empty key sequence case if necessary, e.g., return empty or fallback
                         batched_landmark_indices.append(torch.tensor([], dtype=torch.long, device=self.device))
                         continue

                    # Potentially use a different seed for each batch item or a sequence of seeds
                    current_rls_seed = self.rls_random_seed + i if self.rls_random_seed is not None else None

                    indices_for_item, _ = recursive_nystrom_pytorch(
                        keys_for_sampling[i],
                        n_components=actual_num_landmarks,
                        kernel_func=lambda x, y=None: torch_gauss_kernel(x, y, gamma=self.rls_gamma),
                        lmbda_0=1e-6,
                        random_seed=current_rls_seed, # Pass seed here
                        return_leverage_score=True # We don't use scores here, but it's fine to get them
                    )
                    batched_landmark_indices.append(indices_for_item)

                # k_landmarks: [batch_size, num_landmarks, d_k]
                # We need to gather based on the batched_landmark_indices.
                # This is a bit tricky as gather expects indices for a single dimension.
                # We can construct k_landmarks by iterating or by careful unsqueezing and gathering.
                
                k_landmarks_list = []
                max_selected_landmarks = 0
                for i in range(batch_size):
                    item_indices = batched_landmark_indices[i]
                    if item_indices.numel() > 0:
                        k_landmarks_list.append(key[i, item_indices, :])
                        if item_indices.numel() > max_selected_landmarks:
                            max_selected_landmarks = item_indices.numel()
                            #
                            #
                            #
                effective_num_landmarks = min(actual_num_landmarks, k_seq_len)
                if k_seq_len == 0: # Handle empty key sequence
                    k_landmarks = torch.empty(batch_size, 0, self.d_k, device=self.device, dtype=key.dtype)
                else:
                    # Pad landmark_indices if any batch item has fewer than effective_num_landmarks
                    # This can happen if k_seq_len < effective_num_landmarks for that item after RLS
                    padded_indices_list = []
                    for item_indices in batched_landmark_indices:
                        if item_indices.numel() < effective_num_landmarks:
                            if item_indices.numel() == 0 and effective_num_landmarks > 0 : # No landmarks selected, but we need some
                                # Fallback: take first 'effective_num_landmarks' if k_seq_len > 0
                                padding_indices = torch.arange(min(effective_num_landmarks, k_seq_len), device=self.device, dtype=torch.long)
                                padded_indices_list.append(padding_indices)
                            elif item_indices.numel() == 0 and effective_num_landmarks == 0:
                                padded_indices_list.append(torch.tensor([], dtype=torch.long, device=self.device))
                            else: # Some landmarks selected, but fewer than effective_num_landmarks
                                padding_needed = effective_num_landmarks - item_indices.numel()
                                # Pad by repeating the last selected index, or first if only one.
                                last_idx = item_indices[-1] if item_indices.numel() > 0 else torch.tensor(0, dtype=torch.long, device=self.device)
                                padding = last_idx.repeat(padding_needed)
                                padded_indices_list.append(torch.cat([item_indices, padding]))
                        else: # Sufficient landmarks selected
                            padded_indices_list.append(item_indices[:effective_num_landmarks]) # Ensure correct number

                    if not padded_indices_list: # If batch_size was 0 or k_seq_len was 0 for all
                         k_landmarks = torch.empty(batch_size, 0, self.d_k, device=self.device, dtype=key.dtype)
                    else:
                        landmark_indices_tensor = torch.stack(padded_indices_list, dim=0) # B x effective_num_landmarks
                        # Use gather to select landmarks
                        # landmark_indices_tensor needs to be B x num_landmarks x 1 and then expanded
                        k_landmarks = torch.gather(key, 1, landmark_indices_tensor.unsqueeze(-1).expand(-1, -1, self.d_k))


                # Select q_landmarks
                if is_decoder_self_attn:
                    # Query is [B, 1, Dk], k_landmarks is [B, num_landmarks, Dk]
                    # q_landmarks should probably be query itself for interaction with k_landmarks
                    q_landmarks = query # Shape: [B, 1, Dk]
                else:
""" # This part should never be run
                    if q_seq_len == k_seq_len and k_seq_len > 0 and landmark_indices_tensor.numel() > 0 :
                         q_landmarks = torch.gather(query, 1, landmark_indices_tensor.unsqueeze(-1).expand(-1, -1, self.d_k))
                    elif k_seq_len > 0 and landmark_indices_tensor.numel() > 0 : # q_seq_len != k_seq_len, or other mismatch
                        # Fallback: Use mean of queries as q_landmarks, similar to original Nystrom idea
                        # This is a simplification if RLS indices from keys don't apply well to queries.
                        # Ensure query has enough elements to segment or average.
                        if q_seq_len >= effective_num_landmarks and effective_num_landmarks > 0:
                            # Simple averaging if q_seq_len is divisible, otherwise more complex.
                            # For simplicity, let's use a strided slice or mean of all queries if not easily segmentable.
                            # This is a placeholder, proper q_landmark selection here is important.
                            # One simple approach: take the first 'effective_num_landmarks' queries if available
                            q_sel_indices = torch.arange(min(q_seq_len, effective_num_landmarks), device=self.device).unsqueeze(0).expand(batch_size, -1)
                            q_landmarks = torch.gather(query, 1, q_sel_indices.unsqueeze(-1).expand(-1, -1, self.d_k))
                        elif q_seq_len > 0 : # q_seq_len < effective_num_landmarks
                             q_landmarks = query.mean(dim=1, keepdim=True).expand(-1, effective_num_landmarks, -1) # Repeat mean query
                        else: # q_seq_len is 0
                             q_landmarks = torch.empty(batch_size, 0, self.d_k, device=self.device, dtype=query.dtype)

                    else: # k_seq_len is 0, so no k_landmarks, q_landmarks also effectively empty for matmuls
                        q_landmarks = torch.empty(batch_size, 0, self.d_k, device=self.device, dtype=query.dtype)
"""

                # --- 2. Compute the three Nystrom kernels ---
                if k_landmarks.shape[1] == 0: # No landmarks selected or k_seq_len was 0
                    attention_weighted_value = torch.zeros_like(query) 
                else:
                    # kernel_1: Q K_lm^T -> [B, q_seq_len, num_landmarks]
                    kernel_1 = torch.matmul(query, k_landmarks.transpose(-1, -2))
                    kernel_1 = F.softmax(kernel_1, dim=-1)

                    kernel_2 = torch.matmul(q_landmarks, k_landmarks.transpose(-1, -2))
                    kernel_2 = F.softmax(kernel_2, dim=-1) # Softmax over k_landmarks dim

                    # kernel_3: Q_lm K^T -> [B, num_q_landmarks, k_seq_len]
                    kernel_3 = torch.matmul(q_landmarks, key.transpose(-1, -2))
                    
                    # Masking for kernel_3 (causal)
                    if self.mask:
                        if k_seq_len > 0 and q_landmarks.shape[1] > 0:
                            mask_k3 = torch.ones(q_landmarks.shape[1], k_seq_len, device=self.device)
                            mask_k3 = torch.tril(mask_k3) # Q_lm_i cannot attend to K_j if j > i (assuming Q_lm are ordered)
                            kernel_3 = kernel_3.masked_fill(mask_k3.unsqueeze(0) == 0, float('-inf'))

                    kernel_3 = F.softmax(kernel_3, dim=-1) # Softmax over k_seq_len dim

                    # --- 3. Compute Nystrom approximation ---
                    kernel_2_inv = torch.pinverse(kernel_2)
                    
                    term_3_value = torch.matmul(kernel_3, value) # B x num_q_lm x D_v
                    term_2inv_3v = torch.matmul(kernel_2_inv, term_3_value) # B x num_k_lm x D_v (if K2inv is num_k_lm x num_q_lm)
                                                                        # Or B x num_q_lm x D_v if K2 is square
                    attention_weighted_value = torch.matmul(kernel_1, term_2inv_3v) # B x q_seq_len x D_v

            except RuntimeError as e:
                print(f"Error in Nystrom attention: {e}")
                print(f"Falling back to standard attention due to error")
                # Fall back to standard attention on error
                key_transposed = torch.transpose(key, 1, 2)
                attention_scores = torch.matmul(query, key_transposed)
                if self.mask and mask_tensor_qk_t is not None:
                    attention_scores = attention_scores.masked_fill(mask_tensor_qk_t == 0, float('-inf'))
                attention_weights = F.softmax(attention_scores, dim=-1)
                attention_weighted_value = torch.matmul(attention_weights, value)

        if self.use_conv:

            pass


        attention_weighted_value = self.dropout(attention_weighted_value)
        return attention_weighted_value

    def iterative_inv(self, mat, n_iter=6): # Kept from original
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
    def __init__(self, embed_dim, output_dim): # Typically output_dim is hidden_dim (e.g., 4*embed_dim)
        super(PositionWiseFeedForward, self).__init__()
        self.l1 = nn.Linear(embed_dim, output_dim)
        self.RELU = nn.ReLU() # Standard ReLU
        self.l2 = nn.Linear(output_dim, embed_dim)
        self.norm = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1) # Standard dropout rate

    def forward(self, x, residual_x): # x is input, residual_x is from before FFN
        # The original had torch.max(torch.zeros(x.shape), self.l1(x)) which is just F.relu(self.l1(x))
        output = self.l1(x)
        output = self.RELU(output)
        output = self.l2(output)
        output = self.dropout(output)
        # Add & Norm: Add residual connection *before* normalization
        return self.norm(output + residual_x)


class NystromMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, d_k, d_v, num_heads, num_landmarks, mask=False, conv_kernel_size=None, CUDA=False, rls_gamma=0.01, rls_random_seed=None):
        super(NystromMultiHeadAttention, self).__init__()
        self.attention_blocks = nn.ModuleList(
            [NystromSelfAttention(embed_dim, d_k, d_v, num_landmarks, mask, CUDA, conv_kernel_size, rls_gamma, rls_random_seed + i if rls_random_seed is not None else None) for i in range(num_heads)]
        )
        self.output_linear = nn.Linear(num_heads * d_v, embed_dim) # Project back to embed_dim
        self.norm = LayerNorm(embed_dim)
        # self.CUDA = CUDA # Not typically stored like this if device is handled by .to(device)
        # self.device = torch.device("cuda:0" if CUDA else "cpu") # Handled by module's device

    def forward(self, query, key, value, residual_x): # residual_x is the input query for skip connection
        attention_outputs = []
        for attention_block in self.attention_blocks:
            attention_outputs.append(attention_block(query, key, value))
        
        # Concatenate outputs from all heads along the feature dimension
        concatenated_output = torch.cat(attention_outputs, dim=-1) # B x q_seq_len x (num_heads * d_v)
        
        # Project back to embed_dim
        projected_output = self.output_linear(concatenated_output) # B x q_seq_len x embed_dim
        
        # Add residual and then LayerNorm
        return self.norm(projected_output + residual_x)


class NystromTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_landmarks, mask=False, conv_kernel_size=None, CUDA=False, rls_gamma=0.01, rls_random_seed=None, ff_hidden_dim=None):
        super(NystromTransformerBlock, self).__init__()
        # d_k and d_v are typically embed_dim // num_heads
        head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.multi_head_attention = NystromMultiHeadAttention(
            embed_dim=embed_dim,
            d_k=head_dim,
            d_v=head_dim,
            num_heads=num_heads,
            num_landmarks=num_landmarks,
            mask=mask,
            conv_kernel_size=conv_kernel_size,
            CUDA=CUDA, # Pass CUDA for device setting within NystromSelfAttention if it uses it
            rls_gamma=rls_gamma,
            rls_random_seed=rls_random_seed
        )
        if ff_hidden_dim is None:
            ff_hidden_dim = embed_dim * 4 # Common practice for FFN hidden dim
        self.feed_forward = PositionWiseFeedForward(embed_dim, ff_hidden_dim) # Pass embed_dim and hidden_dim

    def forward(self, query, key, value): # Standard transformer block takes x (query,key,value are same for self-attn)
        # For self-attention, query, key, value are all the same input x
        # residual_x is the input to the MHA block
        attention_out = self.multi_head_attention(query, key, value, residual_x=query) # Pass query as residual
        
        # The residual for FFN is the output of the attention block (attention_out)
        feed_forward_out = self.feed_forward(attention_out, residual_x=attention_out)
        return feed_forward_out

