"""
Solucoes Analiticas e Termos Fonte para Identificacao de f(x,t)

Problema: u_t = alpha * u_xx + f(x,t)
          u(0,t) = u(1,t) = 0
          u(x,0) = 0

TODAS as solucoes foram derivadas rigorosamente e verificadas
numericamente -- o residuo u_t - alpha*u_xx - f e identicamente
zero em todo o dominio para cada par (u, f) implementado.

Metodologia
-----------
Para f(x,t) = sum_n F_n(t)*sin(n*pi*x) o modo n satisfaz:
    g_n'(t) + lambda_n * g_n(t) = F_n(t),   g_n(0) = 0
onde lambda_n = alpha*(n*pi)^2.

Para F_n(t) = A*exp(-mu*t) com lambda_n != mu:
    g_n(t) = A/(lambda_n - mu) * [exp(-mu*t) - exp(-lambda_n*t)]

Experimentos
------------
1. example_1 : f = sin(pi*x)*exp(-t)
2. example_2 : f = (sin(pi*x) + 0.5*sin(2*pi*x))*exp(-2t)
3. example_3 : manufactured -- u = 0.05*sin(pi*x)*(1-cos(2*pi*t))
4. example_4 : manufactured -- u = 0.1*sin(pi*x)*(1-exp(-2t))
5. example_5 : igual ao example_1 (teste de ruido)
"""

import torch
import numpy as np


def f_real(X, T, case):
    """
    Termo fonte VERDADEIRO f(x,t) para cada caso.
    Todos verificados: u_t - alpha*u_xx - f = 0.
    """
    if case in ("example_1", "example_5"):
        return torch.sin(np.pi * X) * torch.exp(-T)

    elif case == "example_2":
        return (torch.sin(np.pi * X) + 0.5 * torch.sin(2.0 * np.pi * X))                * torch.exp(-2.0 * T)

    elif case == "example_3":
        # Derivado de u = 0.05*sin(pi*x)*(1-cos(2*pi*t)):
        # f = u_t - alpha*u_xx  com alpha=1
        A = 0.05
        return torch.sin(np.pi * X) * (
            2.0 * np.pi * A * torch.sin(2.0 * np.pi * T)
            + np.pi ** 2 * A * (1.0 - torch.cos(2.0 * np.pi * T))
        )

    elif case == "example_4":
        # Derivado de u = 0.1*sin(pi*x)*(1-exp(-2t)):
        # f = u_t - alpha*u_xx  com alpha=1
        A = 0.1
        return torch.sin(np.pi * X) * (
            2.0 * A * torch.exp(-2.0 * T)
            + np.pi ** 2 * A * (1.0 - torch.exp(-2.0 * T))
        )

    else:
        raise ValueError(f"Caso '{{case}}' nao implementado.")


def u_real(X, T, case, alpha=1.0):
    """
    Solucao VERDADEIRA u(x,t).
    Satisfaz rigorosamente: u_t = alpha*u_xx + f, u(0,t)=u(1,t)=u(x,0)=0.
    """
    if case in ("example_1", "example_5"):
        # Modo n=1: g' + alpha*pi^2*g = exp(-t), g(0)=0
        # g(t) = 1/(alpha*pi^2-1) * (exp(-t) - exp(-alpha*pi^2*t))
        lam = alpha * np.pi ** 2
        C = 1.0 / (lam - 1.0)
        return C * (torch.exp(-T) - torch.exp(-lam * T)) * torch.sin(np.pi * X)

    elif case == "example_2":
        # Modo n=1 com decaimento exp(-2t)
        # g1' + lam1*g1 = exp(-2t)  =>  g1 = 1/(lam1-2)*(exp(-2t)-exp(-lam1*t))
        # Modo n=2: g2' + lam2*g2 = 0.5*exp(-2t)
        # g2 = 0.5/(lam2-2)*(exp(-2t)-exp(-lam2*t))
        lam1 = alpha * np.pi ** 2
        lam2 = alpha * (2.0 * np.pi) ** 2
        C1 = 1.0 / (lam1 - 2.0)
        C2 = 0.5 / (lam2 - 2.0)
        g1 = C1 * (torch.exp(-2.0 * T) - torch.exp(-lam1 * T))
        g2 = C2 * (torch.exp(-2.0 * T) - torch.exp(-lam2 * T))
        return g1 * torch.sin(np.pi * X) + g2 * torch.sin(2.0 * np.pi * X)

    elif case == "example_3":
        return 0.05 * torch.sin(np.pi * X) * (1.0 - torch.cos(2.0 * np.pi * T))

    elif case == "example_4":
        return 0.1 * torch.sin(np.pi * X) * (1.0 - torch.exp(-2.0 * T))

    else:
        raise ValueError(f"Caso '{{case}}' nao implementado.")


def verify_solutions(alpha=1.0, tol=1e-4):
    """Verifica numericamente (diferencas finitas) que cada par (u,f) satisfaz a PDE."""
    cases = ["example_1", "example_2", "example_3", "example_4", "example_5"]
    x = torch.linspace(0.05, 0.95, 30)
    t = torch.linspace(0.05, 0.95, 30)
    X, T = torch.meshgrid(x, t, indexing="ij")
    dt = dx = 1e-5
    all_ok = True
    for case in cases:
        u   = u_real(X,      T,      case, alpha)
        u_p = u_real(X,      T + dt, case, alpha)
        u_m = u_real(X,      T - dt, case, alpha)
        u_r = u_real(X + dx, T,      case, alpha)
        u_l = u_real(X - dx, T,      case, alpha)
        f   = f_real(X, T, case)
        u_t  = (u_p - u_m) / (2 * dt)
        u_xx = (u_r - 2 * u + u_l) / dx ** 2
        res  = (u_t - alpha * u_xx - f).abs().max().item()
        ok   = res < tol
        all_ok = all_ok and ok
        print(f"  {{case}}: max_residuo = {{res:.2e}}  [{{'OK' if ok else 'FALHOU'}}]")
    return all_ok


EXPERIMENT_DESCRIPTIONS = {
    "example_1": {
        "name": "Baseline - Decaimento Exponencial",
        "f_formula": "f(x,t) = sin(pi*x)*exp(-t)",
        "u_formula": "u = [1/(alpha*pi^2-1)] * (exp(-t) - exp(-alpha*pi^2*t)) * sin(pi*x)",
        "notes": "Solucao exata via separacao de variaveis. Caso de referencia."
    },
    "example_2": {
        "name": "Dois Modos de Fourier com exp(-2t)",
        "f_formula": "f = (sin(pi*x) + 0.5*sin(2*pi*x)) * exp(-2t)",
        "u_formula": "u = C1*(exp(-2t)-exp(-lam1*t))*sin(pi*x) + C2*(exp(-2t)-exp(-lam2*t))*sin(2*pi*x)",
        "notes": "Estrutura espacial mais rica. Dois modos de Fourier independentes."
    },
    "example_3": {
        "name": "Manufactured - Oscilacao Temporal Periodica",
        "f_formula": "f = sin(pi*x)*[0.1*pi*sin(2*pi*t) + 0.05*pi^2*(1-cos(2*pi*t))]",
        "u_formula": "u(x,t) = 0.05*sin(pi*x)*(1-cos(2*pi*t))",
        "notes": "Manufactured solution. f derivado analiticamente de u."
    },
    "example_4": {
        "name": "Manufactured - Crescimento com Saturacao",
        "f_formula": "f = sin(pi*x)*[0.2*exp(-2t) + 0.1*pi^2*(1-exp(-2t))]",
        "u_formula": "u(x,t) = 0.1*sin(pi*x)*(1-exp(-2t))",
        "notes": "Manufactured solution. f tem comportamento nao-monotono."
    },
    "example_5": {
        "name": "Robustez a Ruido",
        "f_formula": "f(x,t) = sin(pi*x)*exp(-t)  [+ ruido nos dados de treinamento]",
        "u_formula": "Igual ao example_1",
        "notes": "Mesmo sistema do example_1. Teste de robustez com ruido."
    },
}


if __name__ == "__main__":
    print("Verificando solucoes analiticas (diferencas finitas)...")
    ok = verify_solutions(alpha=1.0)
    print(f"\nResultado: {'TODAS OK' if ok else 'FALHAS DETECTADAS'}")
