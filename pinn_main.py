"""
Script Principal -- Identificacao de Termo Fonte via PINN

Uso:
    python pinn_main.py                                        # config padrao, gera dados
    python pinn_main.py --config configs/exp1.json             # via JSON
    python pinn_main.py --case example_2 --steps 6000         # overrides CLI
    python pinn_main.py --dataset data_synthetic/exp1_baseline_20260218.npz
                                                               # usa dataset pre-gerado

Ordem de prioridade para dados:
    1. --dataset  (carrega .npz existente, ignora geracao on-the-fly)
    2. geracao on-the-fly com os parametros do config / CLI

Nota: --dataset e --case sao mutuamente exclusivos.
"""

import os
import sys
import json
import copy
import argparse
import torch
import numpy as np
from datetime import datetime

from synthetic_data_generator import SyntheticDataGenerator, load_dataset
from pinn_architecture import SourceTermPINN
from pinn_trainer import SourceTermTrainer
from pinn_visualization import PINNVisualizer


# =============================================================================
# CONFIGURACAO PADRAO
# =============================================================================

DEFAULT_CONFIG = {
    "experiment": {
        "case":        "example_1",
        "domain":      [0, 1, 0, 1],
        "alpha":       1.0,
        "noise_level": 0.0,
    },
    "data": {
        "N_test_x":  200,
        "N_test_t":  200,
        "N_train_x": 150,
        "N_train_t": 150,
        "N_bc":      1000,
        "N_f_obs":   200,
        "N_obs_ux":  50,
        "N_obs_ut":  50,
    },
    "model": {
        "domain": [0, 1, 0, 1],
        "alpha":  1.0,
        "net_u": {
            "layers":     [2, 50, 50, 50, 50, 1],
            "activation": "tanh",
        },
        "net_f": {
            "layers":     [2, 50, 50, 50, 1],
            "activation": "tanh",
        },
    },
    "training": {
        "n_steps":             8000,
        "batch_size":          1000,
        "log_interval":        500,
        "plot_interval":       1000,
        "checkpoint_interval": 2000,
        "loss_weights": {
            "bc":    10.0,
            "pde":   10.0,
            "reg":   1e-4,
            "f_obs":  50.0,
            "u_obs": 100.0,
        },
        "optimizer": {
            "type": "adam",
            "lr":   1e-3,
        },
        "scheduler": {
            "type":      "step",
            "step_size": 2000,
            "gamma":     0.9,
        },
    },
    "output": {
        "base_dir":      "outputs",
        "gif_fps":       5,
        "remove_frames": False,
    },
}


# =============================================================================
# UTILIDADES
# =============================================================================

def merge_configs(base: dict, override: dict) -> dict:
    """Mescla override sobre base, recursivamente."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_configs(result[k], v)
        else:
            result[k] = v
    return result


def create_output_dir(base: str, tag: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(base, f"{tag}_{ts}")
    os.makedirs(out, exist_ok=True)
    return out


def save_config(config: dict, out_dir: str):
    path = os.path.join(out_dir, "config.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)


def print_device_info():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name()}")
    return device


# =============================================================================
# CARREGAMENTO DE DADOS
# =============================================================================

def load_data_from_npz(npz_path: str, config: dict) -> dict:
    """
    Carrega dataset pre-gerado de um arquivo .npz e atualiza o config
    com os metadados gravados no arquivo (case, alpha, noise_level).

    Parametros
    ----------
    npz_path : str   caminho para o arquivo .npz
    config   : dict  modificado in-place com metadados do arquivo

    Retorna
    -------
    dict com tensors prontos para o trainer
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Dataset nao encontrado: '{npz_path}'\n"
            f"  Gere datasets com:  python generate_all_experiments.py\n"
            f"  Ou via notebook:    data_generator.ipynb"
        )

    print(f"\n  Carregando: {npz_path}")
    ds = load_dataset(npz_path)

    # Atualizar config com metadados do arquivo
    config["experiment"]["case"]        = ds["case"]
    config["experiment"]["alpha"]       = ds["alpha"]
    config["experiment"]["noise_level"] = ds["noise_level"]
    config["model"]["alpha"]            = ds["alpha"]

    size_mb = os.path.getsize(npz_path) / (1024 * 1024)
    print(f"  Caso       : {ds['case']}")
    print(f"  Alpha      : {ds['alpha']}")
    print(f"  Ruido      : {ds['noise_level'] * 100:.1f}%")
    print(f"  Tamanho    : {size_mb:.2f} MB")
    print(f"  PDE points : {ds['x_pde'].shape[0]}")
    print(f"  BC points  : {ds['x_bc'].shape[0]}")
    print(f"  f obs pts  : {ds['x_f_obs'].shape[0]}")
    print(f"  Test points: {ds['x_test'].shape[0]}")

    return ds


def generate_data_on_the_fly(config: dict) -> dict:
    """
    Gera dados sinteticos diretamente a partir do config.

    Retorna
    -------
    dict com tensors prontos para o trainer
    """
    data_cfg = config["data"]
    gen = SyntheticDataGenerator(config)

    u_mesh, f_mesh, x_test, u_test, f_test, X, T = gen.get_test_dataset(
        N_test_x=data_cfg["N_test_x"],
        N_test_t=data_cfg["N_test_t"],
    )
    x_pde            = gen.get_pde_dataset(
        N_train_x=data_cfg["N_train_x"],
        N_train_t=data_cfg["N_train_t"],
    )
    x_bc,    y_bc    = gen.get_bc_dataset(N_bc=data_cfg["N_bc"])
    x_f_obs, f_obs   = gen.get_source_observations(N_obs=data_cfg["N_f_obs"])
    x_u_obs, u_obs   = gen.get_u_observations(
        N_obs_x=data_cfg.get("N_obs_ux", 50),
        N_obs_t=data_cfg.get("N_obs_ut", 50),
    )

    print(f"\n  PDE points   : {x_pde.shape[0]}")
    print(f"  BC points    : {x_bc.shape[0]}")
    print(f"  f obs points : {x_f_obs.shape[0]}")
    print(f"  u obs points : {x_u_obs.shape[0]}")
    print(f"  Test points  : {x_test.shape[0]}")

    return {
        "x_pde":   x_pde,
        "x_bc":    x_bc,
        "y_bc":    y_bc,
        "x_test":  x_test,
        "u_test":  u_test,
        "f_test":  f_test,
        "x_f_obs": x_f_obs,
        "f_obs":   f_obs,
        "x_u_obs": x_u_obs,
        "u_obs":   u_obs,
        "X":       X,
        "T":       T,
        "u_mesh":  u_mesh,
        "f_mesh":  f_mesh,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PINN para Identificacao de Termo Fonte f(x,t)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Gerar dados on-the-fly e treinar
  python pinn_main.py
  python pinn_main.py --config configs/exp2_standard.json
  python pinn_main.py --case example_3 --steps 10000 --lr 5e-4

  # Usar dataset pre-gerado (--dataset e --case sao mutuamente exclusivos)
  python pinn_main.py --dataset data_synthetic/exp1_baseline_20260218_120000.npz
  python pinn_main.py --dataset data_synthetic/exp5_noise05_20260218.npz --steps 5000
  python pinn_main.py --dataset data_synthetic/exp4_discontinuous_20260218.npz \\
                      --config configs/exp4_discontinuous.json
        """,
    )

    # Configuracao global
    parser.add_argument(
        "--config", type=str, default=None, metavar="JSON",
        help="Arquivo JSON de configuracao (ex: configs/exp2_standard.json)"
    )

    # Fonte de dados -- mutuamente exclusivos
    data_src = parser.add_mutually_exclusive_group()
    data_src.add_argument(
        "--dataset", type=str, default=None, metavar="NPZ",
        help=(
            "Dataset .npz pre-gerado para carregar. "
            "Quando fornecido, a geracao on-the-fly e ignorada e os metadados "
            "(case, alpha, noise) sao lidos diretamente do arquivo. "
            "Gere datasets com: python generate_all_experiments.py  ou  data_generator.ipynb"
        )
    )
    data_src.add_argument(
        "--case", type=str, default=None,
        choices=["example_1", "example_2", "example_3", "example_4", "example_5"],
        help="Experimento a gerar on-the-fly (sobrescreve experiment.case do config)"
    )

    # Overrides de hiperparametros
    parser.add_argument("--steps",  type=int,   default=None, help="Numero de steps de treinamento")
    parser.add_argument("--noise",  type=float, default=None, help="Nivel de ruido (0.0 a 1.0)")
    parser.add_argument("--lr",     type=float, default=None, help="Learning rate inicial")
    parser.add_argument("--output", type=str,   default=None, help="Pasta base de saida")

    args = parser.parse_args()

    # --- Montar config ---
    config = copy.deepcopy(DEFAULT_CONFIG)

    if args.config:
        with open(args.config, "r", encoding="utf-8") as fh:
            user_cfg = json.load(fh)
        config = merge_configs(config, user_cfg)

    # Overrides de hiperparametros via CLI (tem prioridade sobre JSON)
    if args.case:
        config["experiment"]["case"] = args.case
    if args.steps:
        config["training"]["n_steps"] = args.steps
    if args.noise is not None:
        config["experiment"]["noise_level"] = args.noise
    if args.lr:
        config["training"]["optimizer"]["lr"] = args.lr
    if args.output:
        config["output"]["base_dir"] = args.output

    # Sincronizar dominio e alpha
    config["model"]["domain"] = config["experiment"]["domain"]
    config["model"]["alpha"]  = config["experiment"]["alpha"]

    # --- Setup ---
    torch.set_default_dtype(torch.float)
    device = print_device_info()

    # --- Carregar ou gerar dados ---
    print("\n" + "=" * 70)
    if args.dataset:
        print("FONTE DE DADOS: arquivo .npz pre-gerado")
        print("=" * 70)
        data = load_data_from_npz(args.dataset, config)
        # Re-sincronizar apos update dos metadados do .npz
        config["model"]["alpha"]  = config["experiment"]["alpha"]
        config["model"]["domain"] = config["experiment"]["domain"]
        data_tag = os.path.splitext(os.path.basename(args.dataset))[0]
    else:
        print("FONTE DE DADOS: geracao on-the-fly")
        print("=" * 70)
        data = generate_data_on_the_fly(config)
        data_tag = config["experiment"]["case"]

    # --- Criar pasta de saida e salvar config ---
    out_dir = create_output_dir(config["output"]["base_dir"], data_tag)
    save_config(config, out_dir)
    print(f"\n  Saida: {out_dir}")

    # --- Modelo ---
    model = SourceTermPINN(config).to(device)

    # --- Visualizador e Trainer ---
    viz     = PINNVisualizer(out_dir)
    trainer = SourceTermTrainer(model, config, out_dir)
    history = trainer.train(data, visualizer=viz)

    # --- Avaliacao final ---
    print("\n" + "=" * 70)
    print("AVALIACAO FINAL")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        x_test_dev = data["x_test"].to(device)
        u_test_dev = data["u_test"].to(device)
        f_test_dev = data["f_test"].to(device)

        f_pred_flat = model.source_term(x_test_dev).cpu()
        u_pred_flat = model.forward(x_test_dev).cpu()

        # Inferir dimensoes da malha diretamente dos dados carregados
        Nx, Nt = data["X"].shape
        f_pred_mesh = f_pred_flat.reshape(Nx, Nt)
        u_pred_mesh = u_pred_flat.reshape(Nx, Nt)

        err_u = model.l2_error_u(x_test_dev, u_test_dev)
        err_f = model.l2_error_f(x_test_dev, f_test_dev)

    print(f"  Erro L2 relativo u(x,t): {err_u:.6f}")
    print(f"  Erro L2 relativo f(x,t): {err_f:.6f}  [METRICA PRINCIPAL]")
    print(f"  Erro max      em f(x,t): {(f_pred_mesh - data['f_mesh']).abs().max().item():.6f}")

    # --- Plots finais ---
    viz.plot_training_history(history)

    # Comparacoes finais -- Termo Fonte
    viz.plot_final_comparison_f(
        data["X"], data["T"], data["f_mesh"], f_pred_mesh, err_f=err_f,
    )
    viz.plot_final_analysis_f(
        data["X"], data["T"], data["f_mesh"], f_pred_mesh,
    )

    # Comparacoes finais -- Solucao PDE
    viz.plot_final_comparison_u(
        data["X"], data["T"], data["u_mesh"], u_pred_mesh, err_u=err_u,
    )
    viz.plot_final_analysis_u(
        data["X"], data["T"], data["u_mesh"], u_pred_mesh,
    )

    # --- GIFs ---
    gif_fps       = config["output"].get("gif_fps", 5)
    remove_frames = config["output"].get("remove_frames", False)

    gif_f = viz.generate_gif_f(fps=gif_fps, remove_frames=remove_frames)
    gif_u = viz.generate_gif_u(fps=gif_fps, remove_frames=remove_frames)
    if gif_f: print(f"\n  GIF f(x,t) : {gif_f}")
    if gif_u: print(f"  GIF u(x,t) : {gif_u}")

    trainer.close()

    print("\n" + "=" * 70)
    print(f"  Todos os resultados salvos em: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
