import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

def __main__():
    # Debugging main function to test the recursive Nystrom implementation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.randn(10, 100, 20, device=device)  # 10 batches, 100 samples, 20 features
    n_components = 10
    kernel_func = _torch_gauss_kernel  # Using the Gaussian kernel function
    random_seed = 42

    class MockSelf:
        def __init__(self, device, rrls_gamma, rrls_lmbda_0):
            self.device = device
            self.rrls_gamma = rrls_gamma
            self.rrls_lmbda_0 = rrls_lmbda_0
            self.rrls_seed_counter = 0  # Initialize seed counter

    mock_self = MockSelf(device, rrls_gamma=0.01, rrls_lmbda_0=1e-6)

    k_landmarks = _sample_landmarks_rrls(
            self=mock_self,
            tensor_3d=X,
            target_num_landmarks=n_components
            )


    print("Sampled landmarks shape:", k_landmarks.shape)
    # Is nan?
    if torch.isnan(k_landmarks).any():
        print("Warning: Sampled landmarks contain NaN values.")


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


if __name__ == "__main__":
    __main__()
