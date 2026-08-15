import torch


# macierz dolnotrójkątna -> rho, zoptymalizowane pod trening
def rho_from_params(params, N):
    B = params.shape[0]
    n_tri = N * (N + 1) // 2
    real, imag = params[:, :n_tri], params[:, n_tri:]

    L = torch.zeros(B, N, N, dtype=torch.cfloat, device=params.device)
    i, j = torch.tril_indices(N, N, device=params.device)
    L[:, i, j] = torch.complex(real, imag)

    LL = L @ L.mH
    tr = torch.einsum("bii->b", LL).real.clamp_min(1e-12)
    return LL / tr.view(B, 1, 1)
