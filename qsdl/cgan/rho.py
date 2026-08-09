import torch

# L -> rho
def rho_from_params(params: list, N=2):
    Im_idx = len(params) // 2

    L = torch.zeros(N, N, dtype=torch.complex64)

    # dziwne
    L_idx = torch.tril_indices(N,N)
    L_row = L_idx[0]
    L_col = L_idx[1]

    complexList = []
    for i in range(Im_idx):
        z = torch.complex(params[i], params[i + Im_idx])
        complexList.append(z)

    for i in range(len(L_row)):
        L[L_row[i], L_col[i]] = complexList[i]

    L_dag = L.mH
    LL = L @ L_dag
    rho = LL / torch.trace(LL)

    return rho
