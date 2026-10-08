import torch

def lowrank_decompose(A: torch.Tensor, rank: int = None) -> tuple:
    """
    SVD low-rank approximation. 
    If rank is None, automatically calculates effective rank based on PyTorch's default tolerance.
    
    Args:
        A: Input matrix of shape (M, N).
        rank: Target rank to truncate. Uses full rank if None.
        
    Returns:
        tuple: (U, s_VT) where U @ s_VT approximates A.
    """
    U, s, VT = torch.linalg.svd(A, full_matrices=False)

    if rank is None:
        tol = S > S[0] * torch.finfo(S.dtype).eps * max(M, N)
        rank = (S > tol).sum().item()

    U = U[:, :rank]
    s = s[:rank]
    VT = VT[:rank, :]

    return U, s[:, None] @ VT