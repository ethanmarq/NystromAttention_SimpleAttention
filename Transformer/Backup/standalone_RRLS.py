"""
This script is RRLS and can be run using: python standalone_RRLS.py

It runs on random data using the following parameters:    
    N_points = 1000
    d_features = 64
    n_landmarks = 50
    gamma_value = 0.01
"""
import torch
import math # For standard math functions if needed

# ================================================
#  PyTorch Kernel Function (Gaussian/RBF)
# ================================================
def torch_gauss_kernel(X: torch.Tensor, Y: torch.Tensor = None, gamma: float = 0.01):
    """
    Computes the Gaussian (RBF) kernel between X and Y using PyTorch.
    K(x, y) = exp(-gamma * ||x - y||^2)

    Args:
        X: Tensor of shape (N, d)
        Y: Optional tensor of shape (M, d). If None, computes diagonal K(X, X).
        gamma: Kernel bandwidth parameter.

    Returns:
        If Y is None: Tensor of shape (N,) containing K(X_i, X_i) [which is 1.0 for RBF].
        If Y is not None: Tensor of shape (N, M) containing K(X_i, Y_j).
    """
    X = X.float() # Ensure float type for calculations
    if Y is None:
        # For RBF kernel, K(x, x) = exp(-gamma * ||x-x||^2) = exp(0) = 1
        return torch.ones(X.shape[0], device=X.device, dtype=X.dtype)
    else:
        Y = Y.float()
        nsq_rows = torch.sum(X**2, dim=1, keepdim=True)
        nsq_cols = torch.sum(Y**2, dim=1, keepdim=True)
        # Use matmul for batch compatibility if needed, but direct broadcasting works for (N,d) @ (M,d).T
        Ksub = nsq_rows - 2 * torch.matmul(X, Y.T) + nsq_cols.T
        # Clamp Ksub to avoid potential numerical issues with very small negative values before exp
        Ksub = torch.clamp(Ksub, min=0.0) 
        return torch.exp(-gamma * Ksub)

# ================================================
# Recursive Nystrom PyTorch Implementation
# ================================================
def recursive_nystrom_pytorch(
    X: torch.Tensor,
    n_components: int,
    kernel_func, # Must accept and return PyTorch tensors
    lmbda_0: float = 1e-6,
    random_seed: int = None,
    return_leverage_score: bool = False
):
    """
    PyTorch implementation of Recursive Nystrom using Ridge Leverage Scores.

    Selects n_components landmark indices from X based on RLS computed
    recursively.

    Args:
        X: Input data tensor (N, d). Assumed to be on a specific device.
        n_components: Number of landmarks to select (m).
        kernel_func: A function like torch_gauss_kernel that takes two PyTorch
                     tensors (X1, X2) and returns the kernel matrix K(X1, X2).
                     If X2 is None, it must return the diagonal K(X1, X1).
        lmbda_0: Minimum regularization value factor. lambda = max(lambda_calc, lmbda_0 * num_landmarks).
        random_seed: Optional integer seed for reproducibility via torch.Generator.
        return_leverage_score: If True, returns leverage scores along with indices.

    Returns:
        indices: Tensor of selected landmark indices (shape: [n_components]).
        leverage_score (optional): Tensor of leverage scores for all points (shape: [N]),
                                   in the original order of X.

    Note: This function processes a single X matrix (N, d). Batching must be
          handled externally by the caller.
    """

    N = X.shape[0]
    device = X.device
    dtype = X.dtype # Use input tensor's dtype

    if n_components >= N:
        print("Warning: n_components >= N. Returning all indices.")
        indices = torch.arange(N, device=device)
        if return_leverage_score:
            # Leverage scores are typically 1 in this case (or k/n if defined differently)
            scores = torch.ones(N, device=device, dtype=dtype)
            return indices, scores
        else:
            return indices
            
    if n_components <= 0:
         print("Warning: n_components <= 0. Returning empty tensor.")
         return torch.tensor([], dtype=torch.long, device=device)

    if random_seed is not None:
        generator = torch.Generator(device=device).manual_seed(random_seed)
    else:
        generator = None # Use default PyTorch RNG

    # Parameters (ensure calculations use tensors for device consistency where needed)
    log_n_components = torch.log(torch.tensor(n_components, dtype=dtype, device=device))
    # Avoid division by zero if n_components is 1 -> log(1)=0. Use a small floor.
    n_oversample = torch.clamp(log_n_components, min=0.1).item() 

    # Ensure k is at least 1
    k = max(1, int(torch.ceil(torch.tensor(n_components / (4 * n_oversample))).item()))
    # Ensure n_levels is at least 0
    log2_ratio = torch.log2(torch.tensor(N / n_components, dtype=dtype, device=device))
    n_levels = max(0, int(torch.ceil(log2_ratio).item()))

    # Random permutation of data indices
    perm = torch.randperm(N, device=device, generator=generator)

    # Set up sizes for recursive levels
    size_list = [N]
    for _ in range(n_levels): # n_levels means n_levels divisions
        next_size = int(torch.ceil(torch.tensor(size_list[-1] / 2.0)).item())
        # Prevent size from becoming 0 if N is very small relative to n_levels
        size_list.append(max(1, next_size)) 

    # Base case: initial uniform sample indices (relative to perm)
    initial_sample_size = size_list[-1]
    # Ensure initial_sample_size <= N
    initial_sample_size = min(initial_sample_size, N) 
    if initial_sample_size <= 0:
         raise ValueError("Calculated initial sample size is non-positive.")
         
    # Initial sample: indices within the smallest permuted subset
    sample_in_smallest_subset = torch.arange(initial_sample_size, device=device)

    # Absolute data indices corresponding to the initial sample
    indices = perm[sample_in_smallest_subset]
    weights = torch.ones(indices.shape[0], device=device, dtype=dtype)

    # Precompute diagonal of the kernel matrix K(X, X)
    try:
         k_diag = kernel_func(X, None).squeeze() # Should return shape (N,)
         if k_diag.shape[0] != N:
              raise ValueError(f"kernel_func(X, None) returned shape {k_diag.shape}, expected ({N},)")
         if not torch.isfinite(k_diag).all():
              print("Warning: Non-finite values in kernel diagonal. Clamping.")
              k_diag = torch.nan_to_num(k_diag, nan=1.0, posinf=1.0, neginf=0.0) # Replace non-finite with typical values
    except Exception as e:
         raise RuntimeError(f"Error calling kernel_func(X, None): {e}")


    # --- Main recursion loop (unrolled) ---
    # Iterate from level n_levels (smallest subset > initial) down to level 0 (full dataset)
    for l in reversed(range(n_levels + 1)):
        
        if indices.numel() == 0:
             print(f"Error: Landmark set became empty at level {l}. Returning empty.")
             return torch.tensor([], dtype=torch.long, device=device)

        # Indices for the current subset within the permutation
        current_subset_size = size_list[l]
        # Ensure size doesn't exceed N
        current_subset_size = min(current_subset_size, N)
        if current_subset_size <= 0: continue # Skip if subset becomes empty

        current_indices_in_perm = perm[:current_subset_size] # Absolute indices

        # Get data points for current level and current landmarks
        X_current = X[current_indices_in_perm, :] # Shape (current_subset_size, d)
        X_landmarks = X[indices, :]             # Shape (len(indices), d)

        # --- Compute Kernels ---
        try:
             # KS = K(current_subset, landmarks) | Shape (current_subset_size, len(indices))
             KS = kernel_func(X_current, X_landmarks)
             # SKS = K(landmarks, landmarks) | Shape (len(indices), len(indices))
             SKS = kernel_func(X_landmarks, X_landmarks)
             
             if not torch.isfinite(KS).all() or not torch.isfinite(SKS).all():
                 print(f"Warning: Non-finite values in KS or SKS at level {l}. Clamping.")
                 KS = torch.nan_to_num(KS)
                 SKS = torch.nan_to_num(SKS)

        except Exception as e:
             raise RuntimeError(f"Error calling kernel_func at level {l}: {e}")


        # --- Adaptive Regularization (Lambda) ---
        num_landmarks_in_sample = SKS.shape[0]
        # Ensure k doesn't exceed current number of landmarks
        current_k = min(k, num_landmarks_in_sample)

        lmbda_val = torch.tensor(1e-6, device=device, dtype=dtype) # Default
        if current_k > 0 and num_landmarks_in_sample > 0:
            try:
                weighted_SKS = SKS * torch.outer(weights, weights)
                diag_SKS = torch.diag(SKS) # Use unweighted diag for trace calculation per original code
                trace_weighted_SKS = torch.sum(diag_SKS * (weights**2))

                # Ensure matrix is finite and symmetric before eigvalsh
                if not torch.isfinite(weighted_SKS).all():
                    print("Warning: Non-finite values in weighted_SKS. Clamping.")
                    weighted_SKS = torch.nan_to_num(weighted_SKS)
                
                # Check for symmetry (important for eigvalsh)
                if not torch.allclose(weighted_SKS, weighted_SKS.T, atol=1e-5):
                     print("Warning: weighted_SKS is not symmetric. Symmetrizing.")
                     weighted_SKS = (weighted_SKS + weighted_SKS.T) / 2.0

                eigvals = torch.linalg.eigvalsh(weighted_SKS) # Ascending

                if not torch.isfinite(eigvals).all():
                     print(f"Warning: Non-finite eigenvalues detected at level {l}. Using fallback lambda.")
                     lmbda_val = torch.maximum(torch.tensor(lmbda_0 * num_landmarks_in_sample, device=device, dtype=dtype), trace_weighted_SKS / current_k)
                else:
                     sum_largest_k_eigvals = torch.sum(eigvals[-current_k:])
                     lmbda_calc = (trace_weighted_SKS - sum_largest_k_eigvals) / current_k
                     # Ensure calculated lambda is non-negative
                     lmbda_calc = torch.clamp(lmbda_calc, min=0.0) 
                     lmbda_val = torch.maximum(torch.tensor(lmbda_0 * num_landmarks_in_sample, device=device, dtype=dtype), lmbda_calc)

            except torch.linalg.LinAlgError as e:
                print(f"Warning: Eigenvalue computation failed at level {l}: {e}. Using fallback lambda.")
                lmbda_val = torch.tensor(lmbda_0 * num_landmarks_in_sample + 1e-5, device=device, dtype=dtype) # Ensure slightly positive
            except Exception as e: # Catch other potential errors
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


# Main test
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create synthetic data
    N_points = 1000
    d_features = 64
    n_landmarks = 50
    gamma_value = 0.01

    X_data = torch.randn(N_points, d_features, device=device)

    print(f"Input data shape: {X_data.shape}")
    print(f"Selecting {n_landmarks} landmarks...")

    # Select landmarks using the PyTorch function
    selected_indices, scores = recursive_nystrom_pytorch(
        X_data,
        n_components=n_landmarks,
        kernel_func=lambda x, y=None: torch_gauss_kernel(x, y, gamma=gamma_value),
        lmbda_0=1e-6,
        random_seed=42, # For reproducible results
        return_leverage_score=True
    )

    print(f"Selected indices shape: {selected_indices.shape}")
    print(f"Returned leverage scores shape: {scores.shape}")
    print("Selected indices (first 10):", selected_indices[:10].tolist())
    
    # Verify indices are unique and within bounds
    assert selected_indices.unique().shape[0] == selected_indices.shape[0], "Indices are not unique"
    assert selected_indices.max() < N_points, "Index out of bounds"
    assert selected_indices.min() >= 0, "Negative index selected"
    
    print("Indices verified (unique and within bounds).")
    
    # Example: Select the landmark points
    landmark_points = X_data[selected_indices, :]
    print(f"Selected landmark points shape: {landmark_points.shape}")