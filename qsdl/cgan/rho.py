import torch


# macierz dolnotrójkątna -> rho
def rho_from_params(params, N):
    if params.ndim != 2:
        raise ValueError("Error: rho.py -- number of L tensor dimenstions is not 2")

    rhos = []
    for param_tensor in params:

        Im_idx = len(param_tensor) // 2

        L = torch.zeros(N, N, dtype=torch.complex64)

        # dziwne
        L_idx = torch.tril_indices(N,N)
        L_row = L_idx[0]
        L_col = L_idx[1]

        complexList = []
        for i in range(Im_idx):
            z = torch.complex(param_tensor[i], param_tensor[i + Im_idx])
            complexList.append(z)

        for i in range(len(L_row)):
            L[L_row[i], L_col[i]] = complexList[i]

        L_dag = L.mH
        LL = L @ L_dag
        rho = LL / torch.trace(LL)

        rhos.append(rho)

    return torch.stack(rhos,dim=0)
