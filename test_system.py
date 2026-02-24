"""
Testes de Validacao do Sistema PINN

Verifica que todos os modulos funcionam corretamente antes de rodar
experimentos longos.

Uso:
    python test_system.py
    python test_system.py -v   (verbose)
"""

import sys
import traceback
import numpy as np
import torch
import os


PASS = "[OK]"
FAIL = "[FAIL]"


def run_test(name: str, func, verbose: bool = False):
    try:
        func()
        print(f"  {PASS}  {name}")
        return True
    except Exception as e:
        print(f"  {FAIL}  {name}")
        if verbose:
            traceback.print_exc()
        else:
            print(f"        Erro: {e}")
        return False


# =============================================================================
# TESTES
# =============================================================================

def test_imports():
    import torch
    import numpy
    import matplotlib


def test_analytical_solutions():
    from analytical_solutions import u_real, f_real, EXPERIMENT_DESCRIPTIONS
    X = torch.linspace(0, 1, 20)
    T = torch.linspace(0, 1, 20)
    Xm, Tm = torch.meshgrid(X, T, indexing='ij')
    for case in ["example_1", "example_2", "example_3", "example_4", "example_5"]:
        u = u_real(Xm, Tm, case)
        f = f_real(Xm, Tm, case)
        assert u.shape == Xm.shape, f"Shape errado em u para {case}"
        assert f.shape == Xm.shape, f"Shape errado em f para {case}"
    assert len(EXPERIMENT_DESCRIPTIONS) == 5


def test_data_generator():
    from synthetic_data_generator import SyntheticDataGenerator
    config = {
        "experiment": {
            "case": "example_1",
            "domain": [0, 1, 0, 1],
            "alpha": 1.0,
            "noise_level": 0.0,
        }
    }
    gen = SyntheticDataGenerator(config)
    u_mesh, f_mesh, x_test, u_test, f_test, X, T = gen.get_test_dataset(50, 50)
    assert u_mesh.shape == (50, 50)
    assert f_mesh.shape == (50, 50)
    assert x_test.shape == (50 * 50, 2)

    x_pde = gen.get_pde_dataset(30, 30)
    assert x_pde.shape[1] == 2

    x_bc, y_bc = gen.get_bc_dataset(200)
    assert x_bc.shape[1] == 2
    assert y_bc.shape[1] == 1

    x_f, f_o = gen.get_source_observations(50)
    assert x_f.shape[1] == 2


def test_data_generator_noise():
    from synthetic_data_generator import SyntheticDataGenerator
    config = {
        "experiment": {
            "case": "example_5",
            "domain": [0, 1, 0, 1],
            "alpha": 1.0,
            "noise_level": 0.05,
        }
    }
    gen = SyntheticDataGenerator(config)
    x_bc, y_bc = gen.get_bc_dataset(100)
    assert y_bc.shape[1] == 1


def test_architecture():
    from pinn_architecture import SourceTermPINN, FullyConnectedNN, ActivationFactory
    # FullyConnectedNN
    net = FullyConnectedNN([2, 20, 20, 1])
    x = torch.randn(10, 2)
    y = net(x)
    assert y.shape == (10, 1)

    # ActivationFactory
    for name in ["tanh", "relu", "gelu", "sigmoid"]:
        act = ActivationFactory.get(name)
        z = act(torch.randn(5))
        assert z.shape == (5,)

    # SourceTermPINN
    config = {
        "model": {
            "domain": [0, 1, 0, 1],
            "alpha": 1.0,
            "net_u": {"layers": [2, 20, 20, 1], "activation": "tanh"},
            "net_f": {"layers": [2, 20, 1], "activation": "tanh"},
        }
    }
    model = SourceTermPINN(config)
    xt = torch.rand(15, 2)
    u = model.forward(xt)
    f = model.source_term(xt)
    assert u.shape == (15, 1)
    assert f.shape == (15, 1)

    # PDE residual
    res = model.pde_residual(xt)
    assert res.shape == (15, 1)


def test_loss_functions():
    from pinn_architecture import SourceTermPINN
    config = {
        "model": {
            "domain": [0, 1, 0, 1],
            "alpha": 1.0,
            "net_u": {"layers": [2, 20, 20, 1], "activation": "tanh"},
            "net_f": {"layers": [2, 20, 1], "activation": "tanh"},
        }
    }
    model = SourceTermPINN(config)
    x_bc  = torch.rand(20, 2)
    y_bc  = torch.rand(20, 1)
    x_pde = torch.rand(30, 2)
    x_f   = torch.rand(15, 2)
    f_obs = torch.rand(15, 1)

    total, comps = model.total_loss(x_bc, y_bc, x_pde,
                                   x_f_obs=x_f, f_obs=f_obs)
    assert isinstance(total.item(), float)
    assert "loss_bc" in comps
    assert "loss_pde" in comps
    assert "loss_f_obs" in comps


def test_error_metrics():
    from pinn_architecture import SourceTermPINN
    config = {
        "model": {
            "domain": [0, 1, 0, 1],
            "alpha": 1.0,
            "net_u": {"layers": [2, 20, 1], "activation": "tanh"},
            "net_f": {"layers": [2, 20, 1], "activation": "tanh"},
        }
    }
    model = SourceTermPINN(config)
    x = torch.rand(50, 2)
    u = torch.rand(50, 1)
    f = torch.rand(50, 1)
    eu = model.l2_error_u(x, u)
    ef = model.l2_error_f(x, f)
    assert isinstance(eu, float)
    assert isinstance(ef, float)


def test_trainer_quick():
    """Executa 5 steps de treinamento (smoke test)."""
    from pinn_architecture import SourceTermPINN
    from pinn_trainer import SourceTermTrainer
    import tempfile

    config = {
        "model": {
            "domain": [0, 1, 0, 1],
            "alpha": 1.0,
            "net_u": {"layers": [2, 20, 1], "activation": "tanh"},
            "net_f": {"layers": [2, 20, 1], "activation": "tanh"},
        },
        "training": {
            "n_steps": 5,
            "batch_size": 20,
            "log_interval": 5,
            "plot_interval": 10,
            "checkpoint_interval": 10,
            "loss_weights": {"bc": 1.0, "pde": 1.0, "reg": 0.0, "f_obs": 1.0},
            "optimizer": {"type": "adam", "lr": 1e-3},
            "scheduler": {"type": "none"},
        },
    }

    model = SourceTermPINN(config)
    data = {
        "x_pde":   torch.rand(50, 2),
        "x_bc":    torch.rand(20, 2),
        "y_bc":    torch.rand(20, 1),
        "x_test":  torch.rand(25, 2),
        "u_test":  torch.rand(25, 1),
        "f_test":  torch.rand(25, 1),
        "X":       torch.rand(5, 5),
        "T":       torch.rand(5, 5),
        "u_mesh":  torch.rand(5, 5),
        "f_mesh":  torch.rand(5, 5),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = SourceTermTrainer(model, config, tmpdir)
        history = trainer.train(data, visualizer=None)

        # Fechar handlers do logger antes do cleanup (necessario no Windows)
        for handler in trainer.logger.handlers[:]:
            handler.close()
            trainer.logger.removeHandler(handler)

    assert len(history["loss_total"]) == 5
    assert len(history["err_f"]) == 5


def test_gif_generation():
    """Verifica que o GIF ciclico e gerado corretamente."""
    import tempfile
    from pinn_visualization import PINNVisualizer

    with tempfile.TemporaryDirectory() as tmpdir:
        viz = PINNVisualizer(tmpdir)
        X = torch.rand(10, 10)
        T = torch.rand(10, 10)
        f_true = torch.rand(10, 10)
        f_pred = torch.rand(10, 10)
        u_true = torch.rand(10, 10)
        u_pred = torch.rand(10, 10)

        # Gerar 3 frames em ambas as subpastas
        for i in range(1, 4):
            viz.plot_frames(X, T, f_true, f_pred, u_true, u_pred,
                            step=i * 100, frame_idx=i,
                            err_f=0.1, err_u=0.05)

        gif_f = viz.generate_gif_f(fps=2, remove_frames=False)
        gif_u = viz.generate_gif_u(fps=2, remove_frames=False)

        assert gif_f is not None, "GIF de f nao foi criado"
        assert gif_u is not None, "GIF de u nao foi criado"
        assert os.path.exists(gif_f), f"Arquivo nao encontrado: {gif_f}"
        assert os.path.exists(gif_u), f"Arquivo nao encontrado: {gif_u}"
        assert os.path.getsize(gif_f) > 0, "GIF de f esta vazio"
        assert os.path.getsize(gif_u) > 0, "GIF de u esta vazio"


# =============================================================================
# RUNNER
# =============================================================================

def main():
    verbose = "-v" in sys.argv or "--verbose" in sys.argv

    tests = [
        ("Importacoes",            test_imports),
        ("Solucoes analiticas",    test_analytical_solutions),
        ("Gerador de dados",       test_data_generator),
        ("Gerador com ruido",      test_data_generator_noise),
        ("Arquitetura PINN",       test_architecture),
        ("Loss functions",         test_loss_functions),
        ("Metricas de erro",       test_error_metrics),
        ("Trainer (smoke test)",   test_trainer_quick),
        ("GIF ciclico",            test_gif_generation),
    ]

    print("\n" + "=" * 60)
    print("  TESTES DE VALIDACAO DO SISTEMA")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, func in tests:
        ok = run_test(name, func, verbose=verbose)
        if ok:
            passed += 1
        else:
            failed += 1

    print("=" * 60)
    print(f"  Resultado: {passed}/{len(tests)} testes passaram")
    if failed > 0:
        print(f"  {failed} teste(s) falharam!")
        print("  Use '-v' para ver detalhes dos erros.")
    else:
        print("  Sistema pronto para uso.")
    print("=" * 60 + "\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
