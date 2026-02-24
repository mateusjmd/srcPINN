"""
Arquitetura PINN para Identificacao de Termo Fonte f(x,t)

Problema: u_t = alpha * u_xx + f(x,t)
          u(0,t) = u(1,t) = 0

Implementa:
    - ActivationFactory : selecao de ativacoes por nome
    - FullyConnectedNN  : rede generica configuravel
    - SourceTermPINN    : modelo completo com duas redes (u e f)
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


# =============================================================================
# FABRICA DE ATIVACOES
# =============================================================================

class ActivationFactory:
    """Retorna uma funcao de ativacao pelo nome."""

    _MAP = {
        "tanh":     nn.Tanh,
        "relu":     nn.ReLU,
        "gelu":     nn.GELU,
        "elu":      nn.ELU,
        "silu":     nn.SiLU,
        "softplus": nn.Softplus,
        "sigmoid":  nn.Sigmoid,
    }

    @staticmethod
    def get(name: str) -> nn.Module:
        name = name.lower()
        if name == "sin":
            return _SinActivation()
        if name not in ActivationFactory._MAP:
            raise ValueError(
                f"Ativacao '{name}' desconhecida. "
                f"Opcoes: {list(ActivationFactory._MAP.keys()) + ['sin']}"
            )
        return ActivationFactory._MAP[name]()


class _SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)


# =============================================================================
# REDE TOTALMENTE CONECTADA GENERICA
# =============================================================================

class FullyConnectedNN(nn.Module):
    """
    Rede Neural Totalmente Conectada (feedforward).

    Parametros
    ----------
    layers     : list   [input_dim, h1, h2, ..., output_dim]
    activation : str    nome da ativacao (default 'tanh')
    """

    def __init__(self, layers: list, activation: str = "tanh"):
        super().__init__()
        self.layers_dims = layers

        net = OrderedDict()
        for i in range(len(layers) - 1):
            net[f"linear_{i}"] = nn.Linear(layers[i], layers[i + 1])
            if i < len(layers) - 2:
                net[f"act_{i}"] = ActivationFactory.get(activation)

        self.network = nn.Sequential(net)
        self._init_xavier()

    def _init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# MODELO PINN PARA IDENTIFICACAO DE TERMO FONTE
# =============================================================================

class SourceTermPINN(nn.Module):
    """
    PINN com duas redes:
        u_net  ->  prediz u(x,t)
        f_net  ->  prediz f(x,t)  [OBJETIVO: identificar o termo fonte]

    A PDE  u_t = alpha * u_xx + f(x,t)  e satisfeita via perda do residuo.

    Parametros
    ----------
    config : dict
        Sub-chaves usadas:
            model.domain        : [x_min, x_max, t_min, t_max]
            model.alpha         : float  difusividade (conhecida)
            model.net_u.layers  : list
            model.net_u.activation : str
            model.net_f.layers  : list
            model.net_f.activation : str
    """

    def __init__(self, config: dict):
        super().__init__()

        model_cfg = config["model"]
        domain = model_cfg["domain"]

        self.x_min = float(domain[0])
        self.x_max = float(domain[1])
        self.t_min = float(domain[2])
        self.t_max = float(domain[3])
        self.alpha = float(model_cfg.get("alpha", 1.0))

        # Construir redes
        net_u_cfg = model_cfg["net_u"]
        net_f_cfg = model_cfg["net_f"]

        self.u_net = FullyConnectedNN(
            layers=net_u_cfg["layers"],
            activation=net_u_cfg.get("activation", "tanh"),
        )
        self.f_net = FullyConnectedNN(
            layers=net_f_cfg["layers"],
            activation=net_f_cfg.get("activation", "tanh"),
        )


    # ------------------------------------------------------------------
    # Normalizacao
    # ------------------------------------------------------------------

    def _normalize(self, xt: torch.Tensor) -> torch.Tensor:
        """Normaliza (x, t) para o intervalo [-1, 1]."""
        x_norm = 2.0 * (xt[:, 0:1] - self.x_min) / (self.x_max - self.x_min) - 1.0
        t_norm = 2.0 * (xt[:, 1:2] - self.t_min) / (self.t_max - self.t_min) - 1.0
        return torch.cat([x_norm, t_norm], dim=1)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        """Prediz u(x,t)."""
        return self.u_net(self._normalize(xt))

    def source_term(self, xt: torch.Tensor) -> torch.Tensor:
        """Prediz f(x,t) -- TERMO FONTE A IDENTIFICAR."""
        return self.f_net(self._normalize(xt))

    # ------------------------------------------------------------------
    # Residuo da PDE
    # ------------------------------------------------------------------

    def pde_residual(self, xt: torch.Tensor) -> torch.Tensor:
        """
        Calcula r = u_t - alpha * u_xx - f(x,t)

        Parametros
        ----------
        xt : (N, 2) tensor com grad habilitado

        Retorna
        -------
        (N, 1) tensor  residuo
        """
        xt = xt.clone().detach().requires_grad_(True)

        u = self.forward(xt)
        f = self.source_term(xt)

        # Primeira ordem
        grad_u = torch.autograd.grad(
            u.sum(), xt, create_graph=True, retain_graph=True
        )[0]
        u_x = grad_u[:, 0:1]
        u_t = grad_u[:, 1:2]

        # Segunda ordem em x
        u_xx = torch.autograd.grad(
            u_x.sum(), xt, create_graph=True, retain_graph=True
        )[0][:, 0:1]

        return u_t - self.alpha * u_xx - f

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def loss_bc(self, x_bc: torch.Tensor, y_bc: torch.Tensor) -> torch.Tensor:
        """MSE nas condicoes de contorno / condicao inicial."""
        return torch.mean((self.forward(x_bc) - y_bc) ** 2)

    def loss_pde(self, x_pde: torch.Tensor) -> torch.Tensor:
        """MSE do residuo da PDE."""
        return torch.mean(self.pde_residual(x_pde) ** 2)

    def loss_source_reg(self, x_pde: torch.Tensor) -> torch.Tensor:
        """
        Regularizacao de suavidade em f(x,t):
        penaliza gradientes bruscos, nao a magnitude.
        """
        xt = x_pde.clone().detach().requires_grad_(True)
        f = self.source_term(xt)
        f_grad = torch.autograd.grad(
            f.sum(), xt, create_graph=True
        )[0]
        return torch.mean(f_grad ** 2)

    def loss_f_obs(self, x_f_obs, f_obs):
        """MSE nas observacoes esparsas de f(x,t)."""
        return torch.mean((self.source_term(x_f_obs) - f_obs) ** 2)

    def loss_u_obs(self, x_u_obs, u_obs):
        """
        MSE nas observacoes interiores de u(x,t).

        ESSENCIAL para quebrar a degenerescencia de escala:
        sem observacoes internas de u com magnitude real, o sistema
            u_t = alpha*u_xx + f,  u(0,t)=u(1,t)=u(x,0)=0
        e invariante por (u, f) -> (k*u, k*f) para qualquer k, e a rede
        pode convergir para uma escala arbitraria de u mantendo f correto.
        """
        return torch.mean((self.forward(x_u_obs) - u_obs) ** 2)

    def total_loss(self, x_bc, y_bc, x_pde,
                   w_bc=100.0, w_pde=10.0, w_reg=1e-4,
                   x_f_obs=None, f_obs=None, w_f=50.0,
                   x_u_obs=None, u_obs=None, w_u=100.0):
        """
        Loss total:
            L = w_bc*L_BC + w_pde*L_PDE + w_reg*L_reg
                [+ w_u*L_u_obs  observacoes interiores de u(x,t)]
                [+ w_f*L_f_obs  observacoes esparsas de f(x,t)]

        O termo L_u_obs e CRITICO para fixar a escala de amplitude de u
        quando a CI e as BCs sao homogeneas (= 0): sem ele, o sistema e
        invariante por escala e a rede converge para uma solucao errada.

        Retorna
        -------
        (total, dict com componentes escalares)
        """
        l_bc  = self.loss_bc(x_bc, y_bc)
        l_pde = self.loss_pde(x_pde)
        l_reg = self.loss_source_reg(x_pde)

        total = w_bc * l_bc + w_pde * l_pde + w_reg * l_reg

        components = {
            "loss_bc":  l_bc.item(),
            "loss_pde": l_pde.item(),
            "loss_reg": l_reg.item(),
        }

        # Observacoes interiores de u -- FIXAM A ESCALA DE AMPLITUDE
        if x_u_obs is not None and u_obs is not None:
            l_u = self.loss_u_obs(x_u_obs, u_obs)
            total = total + w_u * l_u
            components["loss_u_obs"] = l_u.item()

        # Observacoes esparsas de f
        if x_f_obs is not None and f_obs is not None:
            l_f = self.loss_f_obs(x_f_obs, f_obs)
            total = total + w_f * l_f
            components["loss_f_obs"] = l_f.item()

        components["loss_total"] = total.item()
        return total, components

    # ------------------------------------------------------------------
    # Metricas de erro
    # ------------------------------------------------------------------

    def l2_error_u(self, x_test, u_true):
        """Erro L2 relativo em u(x,t)."""
        with torch.no_grad():
            u_pred = self.forward(x_test)
            err = torch.linalg.norm(u_pred - u_true) / (torch.linalg.norm(u_true) + 1e-10)
        return err.item()

    def l2_error_f(self, x_test, f_true):
        """Erro L2 relativo em f(x,t) -- METRICA PRINCIPAL."""
        with torch.no_grad():
            f_pred = self.source_term(x_test)
            err = torch.linalg.norm(f_pred - f_true) / (torch.linalg.norm(f_true) + 1e-10)
        return err.item()

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def summary(self) -> str:
        n_u = self.u_net.count_params()
        n_f = self.f_net.count_params()
        lines = [
            "SourceTermPINN",
            f"  Dominio : [{self.x_min}, {self.x_max}] x [{self.t_min}, {self.t_max}]",
            f"  alpha   : {self.alpha}",
            f"  u_net   : {self.u_net.layers_dims}  ({n_u:,} params)",
            f"  f_net   : {self.f_net.layers_dims}  ({n_f:,} params)",
            f"  Total   : {n_u + n_f:,} parametros",
        ]
        return "\n".join(lines)

    def get_device(self) -> torch.device:
        return next(self.parameters()).device
