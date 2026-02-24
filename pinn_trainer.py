"""
Sistema de Treinamento para PINN de Identificacao de Termo Fonte

Funcionalidades:
    - Treinamento configuravel via JSON
    - Logging para arquivo e stdout
    - Checkpoint automatico
    - Historico de losses e erros
    - Suporte a learning-rate scheduler
    - Frames de f(x,t) E u(x,t) a cada plot_interval steps
    - Ablation study helper
"""

import os
import sys
import json
import time
import logging
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict

from pinn_architecture import SourceTermPINN


# =============================================================================
# TRAINER
# =============================================================================

class SourceTermTrainer:
    """
    Encapsula o loop de treinamento da PINN.

    Parametros
    ----------
    model      : SourceTermPINN
    config     : dict   configuracao completa (compativel com JSON)
    output_dir : str    pasta raiz do experimento
    """

    def __init__(self, model: SourceTermPINN, config: dict, output_dir: str):
        self.model      = model
        self.config     = config
        self.output_dir = output_dir
        self.device     = model.get_device()

        train_cfg = config.get("training", {})

        # Hiperparametros
        self.n_steps             = train_cfg.get("n_steps", 8000)
        self.batch_size          = train_cfg.get("batch_size", 1000)
        self.log_interval        = train_cfg.get("log_interval", 500)
        self.plot_interval       = train_cfg.get("plot_interval", 400)   # default mais fino
        self.checkpoint_interval = train_cfg.get("checkpoint_interval", 2000)

        # Pesos das losses
        w = train_cfg.get("loss_weights", {})
        self.w_bc  = float(w.get("bc",   100.0))
        self.w_pde = float(w.get("pde",   10.0))
        self.w_reg = float(w.get("reg",    1e-4))
        self.w_f   = float(w.get("f_obs",  50.0))
        self.w_u   = float(w.get("u_obs", 1000.0))

        # Otimizador
        opt_cfg  = train_cfg.get("optimizer", {})
        opt_type = opt_cfg.get("type", "adam").lower()
        lr       = float(opt_cfg.get("lr", 1e-3))

        if opt_type == "adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
                weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
            )
        elif opt_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr,
                weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
            )
        else:
            raise ValueError(f"Otimizador '{opt_type}' nao suportado.")

        # Scheduler
        sch_cfg  = train_cfg.get("scheduler", {})
        sch_type = sch_cfg.get("type", "none").lower()
        if sch_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(sch_cfg.get("step_size", 1000)),
                gamma=float(sch_cfg.get("gamma", 0.9)),
            )
        elif sch_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.n_steps,
                eta_min=float(sch_cfg.get("eta_min", 1e-6)),
            )
        else:
            self.scheduler = None

        # Logger
        self.logger = self._setup_logger()

        # Historico e contador de frames
        self.history    = defaultdict(list)
        self.plot_count = 0

        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------

    def _setup_logger(self) -> logging.Logger:
        log_path = os.path.join(self.output_dir, "training_log.txt")
        logger = logging.getLogger(f"pinn_trainer_{id(self)}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            fmt = logging.Formatter("%(message)s")
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(fmt)
            logger.addHandler(ch)
            fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        return logger

    # ------------------------------------------------------------------
    # Loop principal de treinamento
    # ------------------------------------------------------------------

    def train(self, data: dict, visualizer=None):
        """
        Executa o loop de treinamento.

        Parametros
        ----------
        data : dict
            x_pde, x_bc, y_bc, x_test, u_test, f_test  (obrigatorios)
            x_f_obs, f_obs                               (opcionais)
            X, T, u_mesh, f_mesh                         (para visualizacao)
        visualizer : PINNVisualizer | None
        """
        logger = self.logger
        self.model.train()

        logger.info("=" * 70)
        logger.info("INICIANDO TREINAMENTO")
        logger.info("=" * 70)
        logger.info(self.model.summary())
        logger.info("")
        logger.info(f"  Steps            : {self.n_steps}")
        logger.info(f"  Batch size       : {self.batch_size}")
        logger.info(f"  Plot interval    : {self.plot_interval}  "
                    f"(~{self.n_steps // self.plot_interval} frames por GIF)")
        logger.info(f"  w_bc / w_pde     : {self.w_bc} / {self.w_pde}")
        logger.info(f"  w_reg / w_f_obs  : {self.w_reg} / {self.w_f}")
        logger.info(f"  w_u_obs          : {self.w_u}  [fixa escala e forma de u]")
        logger.info("=" * 70)

        # Mover dados para device
        x_pde  = data["x_pde"].float().to(self.device)
        x_bc   = data["x_bc"].float().to(self.device)
        y_bc   = data["y_bc"].float().to(self.device)
        x_test = data["x_test"].float().to(self.device)
        u_test = data["u_test"].float().to(self.device)
        f_test = data["f_test"].float().to(self.device)

        x_f_obs = data.get("x_f_obs")
        f_obs   = data.get("f_obs")
        if x_f_obs is not None:
            x_f_obs = x_f_obs.float().to(self.device)
            f_obs   = f_obs.float().to(self.device)

        x_u_obs = data.get("x_u_obs")
        u_obs   = data.get("u_obs")
        if x_u_obs is not None:
            x_u_obs = x_u_obs.float().to(self.device)
            u_obs   = u_obs.float().to(self.device)
        logger.info(f"  u obs points     : {x_u_obs.shape[0] if x_u_obs is not None else 0}")

        Nx, Nt = data["X"].shape

        # Estado inicial (step 0)
        if visualizer is not None:
            visualizer.plot_ground_truth(
                data["X"], data["T"], data["u_mesh"], data["f_mesh"]
            )
            # Frame 0: predicoes antes do treinamento
            self.plot_count += 1
            self.model.eval()
            with torch.no_grad():
                f0 = self.model.source_term(x_test).reshape(Nx, Nt).cpu()
                u0 = self.model.forward(x_test).reshape(Nx, Nt).cpu()
            visualizer.plot_frames(
                data["X"], data["T"],
                data["f_mesh"], f0,
                data["u_mesh"], u0,
                step=0, frame_idx=self.plot_count,
                err_f=None, err_u=None,
            )

        start_time = time.time()

        for step in range(1, self.n_steps + 1):
            self.model.train()

            # Mini-batch dos pontos PDE
            idx = np.random.choice(x_pde.shape[0],
                                   min(self.batch_size, x_pde.shape[0]),
                                   replace=False)
            x_batch = x_pde[idx]

            # Mini-batch de u_obs (evita custo O(N^2) por step)
            if x_u_obs is not None and x_u_obs.shape[0] > self.batch_size:
                idx_u = np.random.choice(x_u_obs.shape[0],
                                         self.batch_size, replace=False)
                x_u_batch = x_u_obs[idx_u]
                u_batch   = u_obs[idx_u]
            else:
                x_u_batch = x_u_obs
                u_batch   = u_obs

            # Forward + loss
            total, components = self.model.total_loss(
                x_bc, y_bc, x_batch,
                w_bc=self.w_bc, w_pde=self.w_pde, w_reg=self.w_reg,
                x_f_obs=x_f_obs, f_obs=f_obs, w_f=self.w_f,
                x_u_obs=x_u_batch, u_obs=u_batch, w_u=self.w_u,
            )

            self.optimizer.zero_grad()
            total.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # Registrar historico
            for k, v in components.items():
                self.history[k].append(v)

            # Erros de validacao (a cada step -- leve com torch.no_grad)
            self.model.eval()
            err_u = self.model.l2_error_u(x_test, u_test)
            err_f = self.model.l2_error_f(x_test, f_test)
            self.history["err_u"].append(err_u)
            self.history["err_f"].append(err_f)

            # Log no console / arquivo
            if step % self.log_interval == 0 or step == 1:
                elapsed = (time.time() - start_time) / 60
                lr_now  = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Step {step:6d} | "
                    f"Loss {total.item():.5e} | "
                    f"BC {components['loss_bc']:.4e} | "
                    f"PDE {components['loss_pde']:.4e} | "
                    f"Err_u {err_u:.4e} | "
                    f"Err_f {err_f:.4e} | "
                    f"LR {lr_now:.2e} | "
                    f"{elapsed:.1f} min"
                )

            # Frames para os GIFs (f e u em paralelo)
            if visualizer is not None and (
                step % self.plot_interval == 0 or step == self.n_steps
            ):
                self.plot_count += 1
                with torch.no_grad():
                    f_pred = self.model.source_term(x_test).reshape(Nx, Nt).cpu()
                    u_pred = self.model.forward(x_test).reshape(Nx, Nt).cpu()

                visualizer.plot_frames(
                    data["X"], data["T"],
                    data["f_mesh"], f_pred,
                    data["u_mesh"], u_pred,
                    step=step,
                    frame_idx=self.plot_count,
                    err_f=err_f,
                    err_u=err_u,
                )

            # Checkpoint
            if step % self.checkpoint_interval == 0:
                self._save_checkpoint(step)

        elapsed_total = (time.time() - start_time) / 60
        logger.info("")
        logger.info("=" * 70)
        logger.info("TREINAMENTO FINALIZADO")
        logger.info(f"  Tempo total  : {elapsed_total:.2f} min")
        logger.info(f"  Err_u final  : {self.history['err_u'][-1]:.6f}")
        logger.info(f"  Err_f final  : {self.history['err_f'][-1]:.6f}")
        logger.info(f"  Frames GIF   : {self.plot_count}")
        logger.info("=" * 70)

        self._save_checkpoint(self.n_steps, tag="final")
        self._save_history()

        return self.history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, step: int, tag: str = ""):
        suffix = f"_{tag}" if tag else f"_step{step:06d}"
        path = os.path.join(
            self.output_dir, "checkpoints", f"checkpoint{suffix}.pt"
        )
        torch.save({
            "step":        step,
            "model_state": self.model.state_dict(),
            "opt_state":   self.optimizer.state_dict(),
            "history":     dict(self.history),
            "config":      self.config,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["opt_state"])
        self.history = defaultdict(list, ckpt["history"])
        return ckpt["step"]

    def _save_history(self):
        path = os.path.join(self.output_dir, "training_history.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(dict(self.history), fh, indent=2)

    def close(self):
        """Libera handlers do logger (necessario no Windows para evitar file-lock)."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


# =============================================================================
# ABLATION STUDY HELPER
# =============================================================================

def run_ablation_study(base_config: dict, ablations: list, data: dict,
                       output_base: str = "outputs/ablation"):
    """
    Executa um estudo de ablacao variando campos do config.

    Parametros
    ----------
    base_config : dict    configuracao base
    ablations   : list    lista de dicts com 'name' e overrides usando
                          notacao aninhada por ponto, ex.:
                          {"name": "w_pde_5", "training.loss_weights.pde": 5.0}
    data        : dict    dados de treinamento/teste
    output_base : str     pasta raiz de saida

    Retorna
    -------
    list de dicts com 'name', 'err_u_final', 'err_f_final'
    """
    import copy
    results = []

    for abl in ablations:
        name = abl.get("name", "run")
        print(f"\n{'='*60}\n  ABLACAO: {name}\n{'='*60}")

        cfg = copy.deepcopy(base_config)
        for key, val in abl.items():
            if key == "name":
                continue
            parts = key.split(".")
            obj = cfg
            for p in parts[:-1]:
                obj = obj.setdefault(p, {})
            obj[parts[-1]] = val

        out_dir = os.path.join(output_base, name)
        os.makedirs(out_dir, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model   = SourceTermPINN(cfg).to(device)
        trainer = SourceTermTrainer(model, cfg, out_dir)
        history = trainer.train(data, visualizer=None)
        trainer.close()

        results.append({
            "name":        name,
            "err_u_final": history["err_u"][-1],
            "err_f_final": history["err_f"][-1],
            "config":      cfg,
        })

    print("\n" + "=" * 60)
    print("  RESUMO ABLACAO")
    print("=" * 60)
    print(f"  {'Nome':<30}  {'Err_u':>10}  {'Err_f':>10}")
    print("-" * 60)
    for r in results:
        print(f"  {r['name']:<30}  {r['err_u_final']:>10.6f}  {r['err_f_final']:>10.6f}")
    print("=" * 60)

    return results
