"""
Gerador de Dados Sinteticos para Identificacao de Termo Fonte f(x,t)

Problema: u_t = alpha * u_xx + f(x,t)
          u(0,t) = u(1,t) = 0

Gera datasets para treinamento e avaliacao da PINN incluindo:
    - Pontos de colocacao PDE
    - Condicoes de contorno e inicial
    - Observacoes interiores de u(x,t)
    - Observacoes esparsas de f(x,t) (opcional)
    - Dados de teste em malha fina
"""

import os
import torch
import numpy as np
from datetime import datetime

from analytical_solutions import u_real, f_real, EXPERIMENT_DESCRIPTIONS


class SyntheticDataGenerator:
    """
    Gerador de dados sinteticos para o problema inverso de identificacao
    do termo fonte f(x,t) na equacao do calor 1D.
    """

    def __init__(self, config):
        """
        Parametros
        ----------
        config : dict
            Configuracao do experimento (compativel com JSON de configs/).
            Campos relevantes:
                experiment.case    : str   'example_1' ... 'example_5'
                experiment.domain  : list  [x_min, x_max, t_min, t_max]
                experiment.alpha   : float coeficiente de difusao
                experiment.noise_level : float nivel de ruido
        """
        exp  = config["experiment"]
        self.case        = exp["case"]
        self.alpha       = exp.get("alpha", 1.0)
        self.noise_level = exp.get("noise_level", 0.0)

        domain = exp["domain"]
        self.x_min, self.x_max = domain[0], domain[1]
        self.t_min, self.t_max = domain[2], domain[3]

        desc = EXPERIMENT_DESCRIPTIONS.get(self.case, {})
        print(f"\n{'='*60}")
        print(f"  Experimento: {desc.get('name', self.case)}")
        print(f"  f(x,t)    : {desc.get('f_formula', 'N/A')}")
        print(f"  alpha     : {self.alpha}")
        print(f"  Ruido     : {self.noise_level*100:.1f}%")
        print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Dados de TESTE (malha uniforme densa)
    # ------------------------------------------------------------------

    def get_test_dataset(self, N_test_x=200, N_test_t=200):
        """
        Gera dataset de teste em malha uniforme fina.

        Retorna
        -------
        u_real_mesh : (N_test_x, N_test_t) tensor  solucao verdadeira
        f_real_mesh : (N_test_x, N_test_t) tensor  termo fonte verdadeiro
        x_test      : (N*N, 2) tensor  pontos (x, t) linearizados
        u_test      : (N*N, 1) tensor
        f_test      : (N*N, 1) tensor
        X, T        : tensors de meshgrid
        """
        x = torch.linspace(self.x_min, self.x_max, N_test_x)
        t = torch.linspace(self.t_min, self.t_max, N_test_t)
        X, T = torch.meshgrid(x, t, indexing='ij')

        u_mesh = u_real(X, T, self.case, self.alpha)
        f_mesh = f_real(X, T, self.case)

        x_test = torch.hstack([X.flatten().view(-1, 1),
                                T.flatten().view(-1, 1)])
        u_test = u_mesh.flatten().view(-1, 1)
        f_test = f_mesh.flatten().view(-1, 1)

        return u_mesh, f_mesh, x_test, u_test, f_test, X, T

    # ------------------------------------------------------------------
    # Pontos de COLOCACAO PDE
    # ------------------------------------------------------------------

    def get_pde_dataset(self, N_train_x=150, N_train_t=150):
        """
        Pontos interiores para o residuo da PDE.

        Retorna
        -------
        torch.Tensor de shape (N_train_x * N_train_t, 2)
        """
        x = torch.linspace(self.x_min, self.x_max, N_train_x + 2)[1:-1]
        t = torch.linspace(self.t_min, self.t_max, N_train_t + 2)[1:-1]
        X, T = torch.meshgrid(x, t, indexing='ij')
        return torch.hstack([X.flatten().view(-1, 1),
                              T.flatten().view(-1, 1)])

    # ------------------------------------------------------------------
    # Dados de CONTORNO e CONDICAO INICIAL
    # ------------------------------------------------------------------

    def get_bc_dataset(self, N_bc=1000):
        """
        Gera pontos de condicao inicial e de contorno (Dirichlet).

        Inclui:
            - Condicao inicial: u(x, 0) = u0(x)
            - Contorno esquerdo: u(0, t) = 0
            - Contorno direito:  u(1, t) = 0

        Retorna
        -------
        X_bc : (M, 2) tensor
        Y_bc : (M, 1) tensor
        """
        x_bc = torch.linspace(self.x_min, self.x_max, N_bc)
        t_bc = torch.linspace(self.t_min, self.t_max, N_bc)
        X, T = torch.meshgrid(x_bc, t_bc, indexing='ij')

        # Condicao inicial  u(x, t=0)
        ci_X = torch.hstack([X[:, 0].view(-1, 1), T[:, 0].view(-1, 1)])
        ci_Y = u_real(X[:, 0], T[:, 0], self.case, self.alpha).view(-1, 1)

        # Contorno x=0  u(0, t) = 0
        cb_left_X = torch.hstack([X[0, :].view(-1, 1), T[0, :].view(-1, 1)])
        cb_left_Y = torch.zeros(cb_left_X.shape[0], 1)

        # Contorno x=1  u(1, t) = 0
        cb_right_X = torch.hstack([X[-1, :].view(-1, 1), T[-1, :].view(-1, 1)])
        cb_right_Y = torch.zeros(cb_right_X.shape[0], 1)

        X_bc = torch.vstack([ci_X, cb_left_X, cb_right_X])
        Y_bc = torch.vstack([ci_Y, cb_left_Y, cb_right_Y])

        # Adicionar ruido
        if self.noise_level > 0.0:
            std = Y_bc.std().item()
            Y_bc = Y_bc + self.noise_level * std * torch.randn_like(Y_bc)

        # Sub-amostragem aleatoria
        idx = np.random.choice(X_bc.shape[0],
                               min(N_bc, X_bc.shape[0]),
                               replace=False)
        return X_bc[idx], Y_bc[idx]

    # ------------------------------------------------------------------
    # Observacoes INTERIORES de u(x,t)  -- Experimento 3
    # ------------------------------------------------------------------

    def get_interior_observations(self, N_obs_x=5, N_obs_t=50):
        """
        Observacoes de u(x,t) em posicoes de sensores fixos.

        Retorna
        -------
        X_obs    : (N_obs_x * N_obs_t, 2) tensor
        U_obs    : (N_obs_x * N_obs_t, 1) tensor
        x_sensors: (N_obs_x,) tensor  posicoes dos sensores
        """
        x_sensors = torch.linspace(self.x_min + 0.1,
                                   self.x_max - 0.1,
                                   N_obs_x)
        t_obs = torch.linspace(self.t_min, self.t_max, N_obs_t)
        X, T = torch.meshgrid(x_sensors, t_obs, indexing='ij')

        U = u_real(X, T, self.case, self.alpha)
        if self.noise_level > 0.0:
            std = U.std().item()
            U = U + self.noise_level * std * torch.randn_like(U)

        X_obs = torch.hstack([X.flatten().view(-1, 1),
                               T.flatten().view(-1, 1)])
        U_obs = U.flatten().view(-1, 1)
        return X_obs, U_obs, x_sensors

    # ------------------------------------------------------------------
    # Observacoes esparsas de f(x,t)  -- auxiliam a identificacao
    # ------------------------------------------------------------------

    def get_source_observations(self, N_obs=200):
        """
        Observacoes ESPARSAS de f(x,t) para auxiliar a identificacao.
        (Simula medicoes indiretas do termo fonte)

        Retorna
        -------
        X_f_obs : (N_obs^2, 2) tensor
        F_obs   : (N_obs^2, 1) tensor
        """
        x_obs = self.x_min + (self.x_max - self.x_min) * torch.rand(N_obs)
        t_obs = self.t_min + (self.t_max - self.t_min) * torch.rand(N_obs)
        X, T = torch.meshgrid(x_obs, t_obs, indexing='ij')

        F = f_real(X, T, self.case)
        if self.noise_level > 0.0:
            std = max(F.std().item(), 1e-8)
            F = F + self.noise_level * std * torch.randn_like(F)

        X_f_obs = torch.hstack([X.flatten().view(-1, 1),
                                 T.flatten().view(-1, 1)])
        F_obs = F.flatten().view(-1, 1)
        return X_f_obs, F_obs
    # ------------------------------------------------------------------
    # Observacoes interiores de u(x,t) -- ESSENCIAIS para fixar escala
    # ------------------------------------------------------------------

    def get_u_observations(self, N_obs_x=50, N_obs_t=50):
        """
        Observacoes INTERIORES de u(x,t) em grade uniforme.

        Usa grade uniforme (nao aleatoria) para garantir cobertura
        densa e homogenea do dominio, evitando minimos espurios onde
        a rede aprende formas temporais erradas.

        N_obs_x * N_obs_t pontos no total (padrão: 2500).

        CRITICO: o numero de pontos e os pesos devem ser comparaveis
        aos pontos PDE para que u_obs domine a loss de u e fixe
        tanto a escala quanto a forma temporal/espacial correta.

        Parametros
        ----------
        N_obs_x : int  pontos em x (padrão 50)
        N_obs_t : int  pontos em t (padrão 50)

        Retorna
        -------
        X_u_obs : (N_obs_x * N_obs_t, 2) tensor
        U_obs   : (N_obs_x * N_obs_t, 1) tensor
        """
        # Grade interior (excluindo bordas -- ja cobertas pelas BCs)
        x_obs = torch.linspace(self.x_min, self.x_max, N_obs_x + 2)[1:-1]
        t_obs = torch.linspace(self.t_min, self.t_max, N_obs_t + 2)[1:-1]
        X, T = torch.meshgrid(x_obs, t_obs, indexing="ij")

        U = u_real(X, T, self.case, self.alpha)
        if self.noise_level > 0.0:
            std = max(U.std().item(), 1e-8)
            U = U + self.noise_level * std * torch.randn_like(U)

        X_u_obs = torch.hstack([X.flatten().view(-1, 1),
                                 T.flatten().view(-1, 1)])
        U_obs   = U.flatten().view(-1, 1)
        return X_u_obs, U_obs



    # ------------------------------------------------------------------
    # Salvar dataset em .npz
    # ------------------------------------------------------------------

    def save_dataset(self, output_dir="data_synthetic", prefix="exp",
                     N_test_x=200, N_test_t=200,
                     N_train_x=150, N_train_t=150,
                     N_bc=1000, N_f_obs=200, N_obs_ux=50, N_obs_ut=50):
        """
        Gera e salva todos os conjuntos de dados em arquivo .npz.

        Parametros
        ----------
        output_dir : str   pasta de saida
        prefix     : str   prefixo do arquivo

        Retorna
        -------
        str  caminho do arquivo salvo
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"{prefix}_{timestamp}.npz")

        print(f"\nGerando dataset completo...")

        # Dados de teste
        u_mesh, f_mesh, x_test, u_test, f_test, X, T = \
            self.get_test_dataset(N_test_x, N_test_t)

        # Pontos PDE
        x_pde = self.get_pde_dataset(N_train_x, N_train_t)

        # Contorno / CI
        x_bc, y_bc = self.get_bc_dataset(N_bc)

        # Observacoes de f
        x_f_obs, f_obs = self.get_source_observations(N_f_obs)

        # Observacoes interiores de u (fixam escala e forma de u)
        x_u_obs, u_obs = self.get_u_observations(N_obs_ux, N_obs_ut)

        np.savez_compressed(
            filename,
            # Teste
            u_mesh=u_mesh.numpy(),
            f_mesh=f_mesh.numpy(),
            x_test=x_test.numpy(),
            u_test=u_test.numpy(),
            f_test=f_test.numpy(),
            X=X.numpy(),
            T=T.numpy(),
            # Treino
            x_pde=x_pde.numpy(),
            x_bc=x_bc.numpy(),
            y_bc=y_bc.numpy(),
            # f observado
            x_f_obs=x_f_obs.numpy(),
            f_obs=f_obs.numpy(),
            # u observado interior (para fixar escala de amplitude)
            x_u_obs=x_u_obs.numpy(),
            u_obs=u_obs.numpy(),
            # Metadados
            case=np.array([self.case]),
            alpha=np.array([self.alpha]),
            noise_level=np.array([self.noise_level]),
        )

        size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"  Salvo em: {filename}  ({size_mb:.2f} MB)")
        print(f"  PDE points  : {x_pde.shape[0]}")
        print(f"  BC points   : {x_bc.shape[0]}")
        print(f"  f obs points: {x_f_obs.shape[0]}")
        print(f"  u obs points: {x_u_obs.shape[0]}")
        print(f"  Test points : {x_test.shape[0]}")

        return filename


def load_dataset(filepath):
    """
    Carrega dataset salvo em .npz e converte para tensors.

    Parametros
    ----------
    filepath : str

    Retorna
    -------
    dict com tensors prontos para uso
    """
    data = np.load(filepath, allow_pickle=True)

    def to_tensor(key):
        return torch.from_numpy(data[key].astype(np.float32))

    # Compatibilidade retroativa: arquivos antigos podem nao ter x_u_obs
    def maybe_tensor(key):
        if key in data:
            return to_tensor(key)
        return None

    return {
        "u_mesh":   to_tensor("u_mesh"),
        "f_mesh":   to_tensor("f_mesh"),
        "x_test":   to_tensor("x_test"),
        "u_test":   to_tensor("u_test"),
        "f_test":   to_tensor("f_test"),
        "X":        to_tensor("X"),
        "T":        to_tensor("T"),
        "x_pde":    to_tensor("x_pde"),
        "x_bc":     to_tensor("x_bc"),
        "y_bc":     to_tensor("y_bc"),
        "x_f_obs":  to_tensor("x_f_obs"),
        "f_obs":    to_tensor("f_obs"),
        "x_u_obs":  maybe_tensor("x_u_obs"),
        "u_obs":    maybe_tensor("u_obs"),
        "case":     str(data["case"][0]),
        "alpha":    float(data["alpha"][0]),
        "noise_level": float(data["noise_level"][0]),
    }
