"""
Gerador de Dados Sinteticos para Todos os Experimentos

Gera e salva os datasets de todos os 5 experimentos em data_synthetic/.
Tambem cria um relatorio PDF resumindo os dados gerados.

Uso:
    python generate_all_experiments.py
    python generate_all_experiments.py --output_dir data_synthetic
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

from synthetic_data_generator import SyntheticDataGenerator
from analytical_solutions import EXPERIMENT_DESCRIPTIONS


# Configuracoes dos experimentos
EXPERIMENTS = [
    {
        "experiment": {
            "case": "example_1",
            "domain": [0, 1, 0, 1],
            "alpha": 1.0,
            "noise_level": 0.0,
        },
        "prefix": "exp1_baseline",
    },
    {
        "experiment": {
            "case": "example_2",
            "domain": [0, 1, 0, 1],
            "alpha": 1.0,
            "noise_level": 0.0,
        },
        "prefix": "exp2_variable",
    },
    {
        "experiment": {
            "case": "example_3",
            "domain": [0, 1, 0, 1],
            "alpha": 1.0,
            "noise_level": 0.0,
        },
        "prefix": "exp3_partial",
    },
    {
        "experiment": {
            "case": "example_4",
            "domain": [0, 1, 0, 1],
            "alpha": 1.0,
            "noise_level": 0.0,
        },
        "prefix": "exp4_discontinuous",
    },
    {
        "experiment": {
            "case": "example_5",
            "domain": [0, 1, 0, 1],
            "alpha": 1.0,
            "noise_level": 0.01,
        },
        "prefix": "exp5_noise01",
    },
    {
        "experiment": {
            "case": "example_5",
            "domain": [0, 1, 0, 1],
            "alpha": 1.0,
            "noise_level": 0.05,
        },
        "prefix": "exp5_noise05",
    },
    {
        "experiment": {
            "case": "example_5",
            "domain": [0, 1, 0, 1],
            "alpha": 1.0,
            "noise_level": 0.10,
        },
        "prefix": "exp5_noise10",
    },
]


def generate_preview_page(gen, config, pdf, N=80):
    """Adiciona uma pagina de preview ao PDF com u e f reais."""
    u_mesh, f_mesh, _, _, _, X, T = gen.get_test_dataset(N, N)

    X = X.numpy(); T = T.numpy()
    u = u_mesh.numpy(); f = f_mesh.numpy()

    case = config["experiment"]["case"]
    desc = EXPERIMENT_DESCRIPTIONS.get(case, {})
    noise = config["experiment"]["noise_level"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"{desc.get('name', case)}  (noise={noise*100:.0f}%)\n"
        f"f = {desc.get('f_formula', '')}",
        fontsize=11
    )

    vmax = max(abs(f.min()), abs(f.max())) + 1e-12
    lvls = np.linspace(-vmax, vmax, 21)

    c1 = axes[0].contourf(X, T, u, levels=20, cmap="viridis")
    axes[0].set_title(r"True $u(x,t)$"); axes[0].set_xlabel(r"$x$"); axes[0].set_ylabel(r"$t$")
    plt.colorbar(c1, ax=axes[0], format="%.3f")

    c2 = axes[1].contourf(X, T, f, levels=lvls, cmap="RdBu_r", extend="both")
    axes[1].set_title(r"True $f(x,t)$"); axes[1].set_xlabel(r"$x$"); axes[1].set_ylabel(r"$t$")
    plt.colorbar(c2, ax=axes[1], format="%.3f")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def generate_report(saved_files: list, output_dir: str, timestamp: str):
    """Gera relatorio de texto com estatisticas."""
    report_path = os.path.join(output_dir, f"report_{timestamp}.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("=" * 70 + "\n")
        fh.write("RELATORIO DE DADOS SINTETICOS\n")
        fh.write("=" * 70 + "\n")
        fh.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        fh.write(f"Total de arquivos gerados: {len(saved_files)}\n\n")
        for f in saved_files:
            size_mb = os.path.getsize(f) / (1024 * 1024)
            fh.write(f"  {os.path.basename(f)}  ({size_mb:.2f} MB)\n")
        fh.write("\n" + "=" * 70 + "\n")
    print(f"\nRelatorio salvo em: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Gera datasets sinteticos para todos os experimentos"
    )
    parser.add_argument("--output_dir", type=str, default="data_synthetic",
                        help="Pasta de saida dos dados")
    parser.add_argument("--N_test", type=int, default=200,
                        help="Resolucao da malha de teste")
    parser.add_argument("--N_train", type=int, default=150,
                        help="Resolucao dos pontos de treinamento")
    parser.add_argument("--N_bc", type=int, default=1000,
                        help="Numero de pontos de contorno")
    parser.add_argument("--N_f_obs", type=int, default=200,
                        help="Numero de observacoes esparsas de f")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("  GERACAO DE DADOS SINTETICOS -- IDENTIFICACAO DE TERMO FONTE")
    print("=" * 70)
    print(f"  Total de experimentos: {len(EXPERIMENTS)}")
    print(f"  Pasta de saida       : {args.output_dir}")
    print("=" * 70)

    saved_files = []
    pdf_path = os.path.join(args.output_dir, f"summary_all_{timestamp}.pdf")

    with PdfPages(pdf_path) as pdf:
        # Pagina de capa
        fig_cover = plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.6, "Dados Sinteticos\nIdentificacao de Termo Fonte f(x,t)",
                 ha="center", va="center", fontsize=18, fontweight="bold")
        plt.text(0.5, 0.35, f"Gerado em {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                 ha="center", va="center", fontsize=12, color="gray")
        plt.axis("off")
        pdf.savefig(fig_cover)
        plt.close(fig_cover)

        for i, exp_cfg in enumerate(EXPERIMENTS, 1):
            print(f"\n[{i}/{len(EXPERIMENTS)}] {exp_cfg['prefix']}")

            gen = SyntheticDataGenerator(exp_cfg)

            # Preview no PDF
            generate_preview_page(gen, exp_cfg, pdf)

            # Salvar dataset
            fpath = gen.save_dataset(
                output_dir=args.output_dir,
                prefix=exp_cfg["prefix"],
                N_test_x=args.N_test,
                N_test_t=args.N_test,
                N_train_x=args.N_train,
                N_train_t=args.N_train,
                N_bc=args.N_bc,
                N_f_obs=args.N_f_obs,
            )
            saved_files.append(fpath)

    print(f"\nPDF resumo salvo em: {pdf_path}")

    report = generate_report(saved_files, args.output_dir, timestamp)

    print("\n" + "=" * 70)
    print("  GERACAO CONCLUIDA!")
    print(f"  {len(saved_files)} arquivos gerados em '{args.output_dir}/'")
    print("=" * 70)


if __name__ == "__main__":
    main()
