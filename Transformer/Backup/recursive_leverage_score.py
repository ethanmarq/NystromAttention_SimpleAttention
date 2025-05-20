"""
This is a similar iteration of standalone RRLS that is more concise
but it will not run currently, there is no main.



"""

import torch
import math

# Gaussian Kernel
# -----------------
# This is a simple base kernel
# however different kernels might
# function better depending on key vectors
# X: tensor (seq_len, d_k)
# Y: optional tensor 
# 
# Returns tensor (N,) of k(X_i, X_i) [1.0 for RBF].
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

# Recursive Nstrom PyTorch 
# --------------------------
# X: tensor (seq_len, d_k)
# n_components: Number of landmarks 
# kernel_func: A kernel function that takes two tensors, returning a kernel matrix
# lambda_0: minimum regularization value factor
#  - lambda = max(lambda_calc, lambda_0 * num_landmarks)
# return_leverage_score: If True, returns leverage scores along with indices
# 
# Returns indices: Tensor of selected landmark indices (shape: [n_components])
# leverage_score (optional): Tensor of leverage scores for all points
def recursive_nystrom_pytorch(X: torch.Tensor, n_components: int, kernel_func, lmbda_0: float = 1e-6, random_seed: int = None, return_leverage_score: bool = False):
    N = X.shape[0]
    device = X.device
    dtype = X.dtype

    # Make sure num_landmarks > seq_len & num_landmarks > 0
    if (n_components >= N or n_components <= 0):
        print("Warning, Line 42")
        return

    # Random Seed
    if random_seed is not None:
        generator = torch.Generator(device=device).manual_seed(random_seed)
    else:
        generator = None

    # Parameters 
    log_n_components = torch.log(torch.tensor(n_components, dtype=dtype, device=device))
    n_oversample = torch.clamp(log_n_components, min=0.1).item()
    k = max(1, int(torch.ceil(torch.tensor(n_components / (4 * n_oversample))).item()))
    log2_ratio = torch.log2(torch.tensor(N / n_components, dtype=dtype, device=device))
    n_levels = max(0, int(torch.ceil(log2_ratio).item()))
    perm = torch.randperm(N, device=device, generator=generator)

    # set up sizes for recursive levels
    size_list = [N]
    for _ in range(n_levels):
        next_size = int(torch.ceil(torch.tensor(size_list[-1] / 2.0)).item())
        size_list.append(max(1, next_size))

    # indices of points selected at previous level of recursion
    # at the base level it's just a uniform sample of ~ n_component points
    initial_sample_size = size_list[-1]
    initial_sample_size = min(initial_sample_size, N)
    assert not (initial_sample_size <= 0), "Initial smaple size is non-positive"

    sample_in_smallest_subset = torch.arange(initial_sample_size, device=device)
    indices = perm[sample_in_smallest_subset]
    weights = torch.ones(indices.shape[0], device=device, dtype=dtype)

    # compute diagonal of the whole kernel matrix upfront
    k_diag = kernel_func(X, None).squeeze()
    assert not (k_diag.shape[0] != N), "Wrong shape at dim 0 for kernel matrix"
    assert (torch.isfinite(k_diag).all()), "Non-finite values in kernel diagonal"


    # Unrolled Recursion Loop
    # Iterate from n_levels down to 0 (Full dataset)
    for l in reversed(range(n_levels + 1)):
        assert not (indices.numel() == 0), f"landmark empty at level {l}"

        # Indicies of current subset within permutation
        current_subset_size = size_list[l]
        current_subset_size = min(current_subset_size, N)
        if current_subset_size <= 0: continue # Skip if subset empty

        current_indices_in_perm = perm[:current_subset_size] # Absolute indiices

        # Data points for current levels and current landmarks
        X_current = X[current_indices_in_perm, :] # Shape (current_subset_size, d)
        X_landmarks = X[indices, :]                 # Shape (len(indices), d)

        # Build/Computer Sampled Kernels
        try:
            KS = kernel_func(X_Current, X_landmarks)
            SKS = kernel_func(X_landmarks, X_landmarks)
            assert (torch.isfinite(KS).all() and torch.isfinite(SKS).all()), "Nonfinite Values in KS or SKS"
    except Exception as e:
        raise RuntimeError(F"Error calling kernel_func at level {l}: {e}")

    # Adaptive Regularization (Lambda)
    num_landmarks_in_sample = SKS.shape[0]
    current_k = min(k, num_landmarks_in_sample)

    lmbda_val = torch.tensor(1e-6, device=device, dtype=dtype)
    if current_k > 0 and num_landmarks_in_sample > 0:
        try:
            weighted_SKS = SKS * torch.outer(weights, weights)
            diag_SKS = torch.diag(SKS)
            trace_weighted_SKS = torch.sum(diag_SKS * (weights**2))

            assert (torch.isfinite(weighted_SKS).all()), "Non-finite values in weighted_SKS"
            assert (torch.allclose(weighted_SKS, weighted_SKS.T, atol=1e-5)), "weighted_SKS is not symmetric"

            eigvals = torch.linalg.eigvalsh(weighted_SKS)

            assert (torch.isfinite(eigvals).all()), "Non-finite eigvalues at level {l}"

            sum_largest_k_eigvals = torch.sum(eigvals[-current_k:])
            lmbda_calc = (trace_weighted_SKS - sum_largest_k_eigvals) / current_k
            # Ensure calculated lambda is non-negative
            lmbda_calc = torch.clamp(lmbda_calc, min=0.0) 
            lmbda_val = torch.maximum(torch.tensor(lmbda_0 * num_landmarks_in_sample, device=device, dtype=dtype), lmbda_calc)
    
        except torch.linalg.LinAlgError as e:
                print(f"Warning: Eigenvalue computation failed at level {l}: {e}. Using fallback lambda.")
                lmbda_val = torch.tensor(lmbda_0 * num_landmarks_in_sample + 1e-5, device=device, dtype=dtype) # Ensure slightly positive
        except Exception as e:
                 print(f"Warning: Unexpected error during lambda calculation at level {l}: {e}. Using fallback.")
                 lmbda_val = torch.tensor(lmbda_0 * num_landmarks_in_sample + 1e-5, device=device, dtype=dtype)


    # Ensure lambda is positive for stability
    lmbda = torch.maximum(lmbda_val, torch.tensor(1e-8, device=device, dtype=dtype))
    
    # --- Compute Ridge Leverage Scores (RLS) ---
    leverage_score = None
    try:
        # R = KS @ inv(SKS + lambda * diag(1/weights^2))
        diag_reg = torch.diag(lmbda / torch.clamp(weights**2, min=1e-10)) # Clamp weight^2
        inv_term = torch.linalg.inv(SKS + diag_reg)
        R = torch.matmul(KS, inv_term) # Shape (current_subset_size, len(indices))
    
        # leverage_score = (1/lambda) * max(0, k_diag[subset] - sum(R * KS, axis=1))
        row_sums_R_KS = torch.sum(R * KS, dim=1)
        current_k_diag = k_diag[current_indices_in_perm]
    
        if current_k_diag.shape != row_sums_R_KS.shape:
             raise ValueError(f"Shape mismatch in RLS: k_diag {current_k_diag.shape} vs row_sums {row_sums_R_KS.shape}")
    
        leverage_score_unscaled = torch.clamp(current_k_diag - row_sums_R_KS, min=0.0)
        leverage_score = (1.0 / lmbda) * leverage_score_unscaled
        
        if not torch.isfinite(leverage_score).all():
             print(f"Warning: Non-finite leverage scores computed at level {l}. Clamping.")
             leverage_score = torch.nan_to_num(leverage_score, nan=0.0, posinf=1.0) # Replace bad scores
    
    except torch.linalg.LinAlgError as e:
         print(f"Warning: Linear algebra error during RLS computation at level {l}: {e}. Falling back to uniform.")
         leverage_score = None # Signal fallback
    except Exception as e:
         print(f"Warning: Unexpected error during RLS computation at level {l}: {e}. Falling back to uniform.")
         leverage_score = None # Signal fallback
    
    # Use uniform scores if RLS failed
    if leverage_score is None:
        leverage_score = torch.ones(current_subset_size, device=device, dtype=dtype) * (n_components / max(1, current_subset_size))
    
    
    # --- Sampling ---
    if l == 0: # Final level (full dataset)
        # Normalize scores to get probabilities
        p = torch.clamp(leverage_score, min=0.0) # Ensure non-negative
        p_sum = torch.sum(p)
    
        if p_sum <= 1e-10: # Handle sum zero/negative case
            print("Warning: Final leverage scores sum to <= 0. Using uniform probabilities.")
            p = torch.ones_like(p) / max(1, p.numel()) # Avoid division by zero if p is empty
        else:
            p = p / p_sum
    
        # Final sample size
        final_n_components = min(n_components, p.numel())
        if final_n_components < n_components:
             print(f"Warning: Requested n_components ({n_components}) > available points ({p.numel()}). Sampling {final_n_components}.")
    
        if p.numel() > 0 and final_n_components > 0:
             # Sample indices relative to the current subset (which is the full dataset)
             sampled_relative_indices = torch.multinomial(p, num_samples=final_n_components, replacement=False, generator=generator)
             # Get absolute indices from permutation
             final_indices = perm[sampled_relative_indices]
        else:
             final_indices = torch.tensor([], dtype=torch.long, device=device)
    
    
        if return_leverage_score:
            # Return scores in original order
            final_leverage_scores = torch.zeros(N, device=device, dtype=dtype)
            # Use perm to map scores back to original positions
            # leverage_score corresponds to perm[:current_subset_size] which is perm for l=0
            final_leverage_scores[perm] = leverage_score 
            return final_indices, final_leverage_scores
        else:
            return final_indices
    
    else: # Intermediate levels
        # Bernoulli sampling probabilities
        sampling_prob = torch.clamp(n_oversample * leverage_score, min=0.0, max=1.0)
    
        # Sample based on probabilities
        rand_vals = torch.rand(current_subset_size, device=device, dtype=dtype, generator=generator)
        sampled_relative_indices = torch.where(rand_vals < sampling_prob)[0]
    
        # Fallback: if no points sampled, take a uniform sample
        if sampled_relative_indices.numel() == 0:
            print(f"Warning: No points sampled via RLS at level {l}. Falling back to uniform.")
            num_fallback_samples = min(max(1, n_components // (n_levels+1)), current_subset_size) # Sample fraction or 1
            if current_subset_size > 0:
                 # Sample indices relative to the current subset
                 fallback_perm = torch.randperm(current_subset_size, device=device, generator=generator)
                 sampled_relative_indices = fallback_perm[:num_fallback_samples]
                 # Update sampling_prob for weight calculation (uniform for selected)
                 sampling_prob.fill_(0.0) # Reset probs
                 sampling_prob[sampled_relative_indices] = num_fallback_samples / current_subset_size
            else:
                 sampled_relative_indices = torch.tensor([], dtype=torch.long, device=device) # Handle empty subset
    
    
        # Update indices and weights for the next level (coarser level)
        if sampled_relative_indices.numel() > 0:
             # Get absolute indices of the sample for the next round
             indices = current_indices_in_perm[sampled_relative_indices]
             # Calculate weights: sqrt(1 / p_i) for p_i = sampling_prob[sampled_relative_indices]
             sample_probs = sampling_prob[sampled_relative_indices]
             weights = 1.0 / torch.sqrt(torch.clamp(sample_probs, min=1e-10))
        else:
             # If still no samples after fallback (e.g., subset was empty)
             indices = torch.tensor([], dtype=torch.long, device=device)
             weights = torch.tensor([], dtype=dtype, device=device)
             # Loop will terminate in the check at the beginning of the next iteration
    
    # Fallback return if loop finishes unexpectedly (shouldn't happen)
    print("Warning: Recursive Nystrom loop finished unexpectedly.")
    # Return the last valid indices if available
    if 'final_indices' in locals(): return final_indices
    elif 'indices' in locals(): return indices
    else: return torch.tensor([], dtype=torch.long, device=device)
    
    
    
    
    
    
    
    
    





































    