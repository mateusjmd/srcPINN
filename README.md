# srcPINN

## Identificação de Termo Fonte em Equações Parabólicas via PINN

O presente repositório tem por objetivo o versionamento desse projeto destinado à identificabilidade do termo fonte na Equação do Calor Unidimensional utilizando-se de uma Physics-informed Neural Network (PINN).

**Problema inverso:** dado observações de $u(x,t)$, identificar $f(x,t)$ em

$$u_t = \alpha\, u_{xx} + f(x,t), \quad (x,t)\in(0,1)\times(0,1]$$
$$u(0,t) = u(1,t) = 0, \qquad u(x,0) = 0$$

Uma PINN com **dupla rede** aprende simultaneamente $u_{\mathrm{NN}}(x,t)$ e $f_{\mathrm{NN}}(x,t)$, impondo a PDE como restrição diferencial via diferenciação automática.

---

## Índice

- [srcPINN](#srcpinn)
  - [Identificação de Termo Fonte em Equações Parabólicas via PINN](#identificação-de-termo-fonte-em-equações-parabólicas-via-pinn)
  - [Índice](#índice)
  - [Visão Geral](#visão-geral)
  - [Estrutura do Repositório](#estrutura-do-repositório)
  - [Instalação](#instalação)
  - [Início Rápido](#início-rápido)
    - [Rodar o experimento padrão (Experimento 1)](#rodar-o-experimento-padrão-experimento-1)
    - [Especificar configuração via JSON](#especificar-configuração-via-json)
    - [Sobrescrever parâmetros diretamente na linha de comando](#sobrescrever-parâmetros-diretamente-na-linha-de-comando)
    - [Adicionar ruído gaussiano](#adicionar-ruído-gaussiano)
    - [Usar dataset pré-gerado (`.npz`)](#usar-dataset-pré-gerado-npz)
    - [Referência dos argumentos CLI](#referência-dos-argumentos-cli)
  - [Experimentos](#experimentos)
  - [Arquitetura e Método](#arquitetura-e-método)
  - [Função de Perda](#função-de-perda)
  - [Configuração via JSON](#configuração-via-json)
  - [Saídas Geradas](#saídas-geradas)
  - [Testes](#testes)
  - [Ablation Study](#ablation-study)
  - [Referências](#referências)

---

## Visão Geral

O problema inverso de identificar $f(x,t)$ é **mal-posto** no sentido de Hadamard: pequenas perturbações nos dados podem causar grandes desvios na solução reconstruída. A abordagem combina três estratégias para torná-lo tratável:

| Estratégia | Implementação |
|---|---|
| **Dupla rede neural** | $\mathcal{N}_u$ e $\mathcal{N}_f$ independentes, com a PDE como elo via resíduo diferencial |
| **Ancoragem de escala** | $\mathcal{L}_{u\text{-obs}}$ com 3 600 pontos internos — quebra a degenerescência $(u,f)\mapsto(ku,kf)$ inerente às condições homogêneas |
| **Regularização de suavidade** | $\mathcal{L}\_{\rm reg}$ penaliza gradientes bruscos em $f_{\mathrm{NN}}$, análogo à regularização de Tikhonov |

Todas as soluções analíticas de referência foram derivadas rigorosamente (separação de variáveis ou *manufactured solutions*) e verificadas por diferenças finitas com resíduo $< 10^{-4}$.

---

## Estrutura do Repositório

```
srcPINN/
│
├── analytical_solutions.py       # Pares (u*, f*) exatos + verificação numérica
├── synthetic_data_generator.py   # Geração de datasets: PDE, BC/CI, f-obs, u-obs
├── generate_all_experiments.py   # Pré-gera e salva .npz de todos os 5 experimentos
├── data_generator.ipynb          # Notebook interativo para exploração dos dados
│
├── pinn_architecture.py          # Redes N_u e N_f, resíduo PDE via autograd, losses
├── pinn_trainer.py               # Loop de treinamento, logging, checkpointing, ablation
├── pinn_visualization.py         # Heatmaps, slices 1D, GIFs cíclicos de evolução
├── pinn_main.py                  # Ponto de entrada — config-driven via JSON ou CLI
│
├── configs/
│   ├── exp1_baseline.json        # Baseline: modo único, dados completos, sem ruído
│   ├── exp2_standard.json        # Dois modos de Fourier
│   ├── exp3_partial.json         # Oscilação periódica (manufactured solution)
│   ├── exp4_discontinuous.json   # Crescimento com saturação (manufactured solution)
│   └── exp5_noise05.json         # Robustez: ruído gaussiano de 5%
│
├── test_system.py                # Suite completa de testes de validação (9 grupos)
├── requirements.txt
└── README.md
```

---

## Instalação

**Pré-requisitos:** Python ≥ 3.9. GPU CUDA é detectada e usada automaticamente quando disponível.

```bash
git clone https://github.com/mateusjmd/srcPINN.git
cd srcPINN
pip install -r requirements.txt
```

Dependências:

```
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
imageio>=2.28.0
scipy>=1.10.0
```

Valide o ambiente antes de rodar qualquer experimento:

```bash
python test_system.py
```

Saída esperada em um ambiente funcional:

```
============================================================
  TESTES DE VALIDACAO DO SISTEMA
============================================================
  [OK]  Importacoes
  [OK]  Solucoes analiticas
  [OK]  Gerador de dados
  [OK]  Gerador com ruido
  [OK]  Arquitetura PINN
  [OK]  Loss functions
  [OK]  Metricas de erro
  [OK]  Trainer (smoke test)
  [OK]  GIF ciclico
============================================================
  Resultado: 9/9 testes passaram
  Sistema pronto para uso.
============================================================
```

---

## Início Rápido

### Rodar o experimento padrão (Experimento 1)

```bash
python pinn_main.py
```

### Especificar configuração via JSON

```bash
python pinn_main.py --config configs/exp2_standard.json
```

### Sobrescrever parâmetros diretamente na linha de comando

```bash
# Overrides de CLI têm prioridade sobre o JSON
python pinn_main.py --config configs/exp1_baseline.json \
                    --steps 12000 \
                    --lr 0.0005
```

### Adicionar ruído gaussiano

```bash
python pinn_main.py --case example_5 --noise 0.05
```

### Usar dataset pré-gerado (`.npz`)

```bash
# 1. Pré-gerar e salvar todos os datasets em data_synthetic/
python generate_all_experiments.py

# 2. Treinar com o arquivo salvo (ignora geração on-the-fly)
python pinn_main.py --dataset data_synthetic/exp1_baseline_20260218_120000.npz
```

> `--dataset` e `--case` são **mutuamente exclusivos**. Ao usar `--dataset`, os metadados de `alpha`, `case` e `noise_level` são lidos diretamente do arquivo `.npz`.

### Referência dos argumentos CLI

| Argumento | Tipo | Descrição |
|---|---|---|
| `--config` | `str` | Arquivo JSON de configuração |
| `--case` | `str` | Experimento on-the-fly: `example_1` … `example_5` |
| `--dataset` | `str` | Caminho para `.npz` pré-gerado (exclusivo com `--case`) |
| `--steps` | `int` | Número de steps de treinamento |
| `--noise` | `float` | Nível de ruído relativo (0.0 a 1.0) |
| `--lr` | `float` | Learning rate inicial |
| `--output` | `str` | Pasta base de saída |

---

## Experimentos

Todos os experimentos operam no domínio $Q_T = (0,1)\times(0,1]$ com $\alpha = 1$ e $u(x,0) = 0$.

| # | $f^*(x,t)$ | Característica principal | Solução |
|:---:|---|---|:---:|
| **1** | $\sin(\pi x)\,e^{-t}$ | Baseline — modo único, dados completos, sem ruído | Exata |
| **2** | $\bigl(\sin(\pi x)+0.5\sin(2\pi x)\bigr)e^{-2t}$ | Dois modos de Fourier; $n=2$ decai em $e^{-39.5t}$ | Exata |
| **3** | $\sin(\pi x)\bigl[0.1\pi\sin(2\pi t)+0.05\pi^2(1-\cos 2\pi t)\bigr]$ | Oscilação temporal periódica | Manufactured |
| **4** | $\sin(\pi x)\bigl[0.2e^{-2t}+0.1\pi^2(1-e^{-2t})\bigr]$ | Mudança de regime: difusão → saturação em $t^*\approx 0.8$ | Manufactured |
| **5** | $\sin(\pi x)\,e^{-t}$ + $\mathcal{N}(0,\sigma^2)$ | Robustez: $\sigma \in (0.01, 0.05, 0.1)$ | Exata |

**Soluções exatas** (Exp. 1–2): derivadas por separação de variáveis e variação de parâmetros. Para $f = A\,e^{-\mu t}\sin(n\pi x)$:

$$u^{*} (x,t) = \frac{A}{\lambda_n - \mu}\bigl(e^{-\mu t} - e^{-\lambda_n t}\bigr)\sin(n\pi x), \qquad \lambda_n = \alpha(n\pi)^2$$

**Soluções manufaturadas** (Exp. 3–4): escolhe-se $u^{\star}$ livremente e computa-se $f^{\star} = u^{\star}_t - \alpha u^{\star}\_{xx}$ analiticamente, garantindo que $(u^{\star}, f^{\star})$ satisfaça a PDE por construção. Todos os pares são verificados numericamente por diferenças finitas centradas de segunda ordem antes do treinamento.

---

## Arquitetura e Método

O modelo `SourceTermPINN` contém **duas redes feedforward independentes**:

```
                   normalização: (x,t) → [-1,1]²
                          │
          ┌───────────────┴───────────────┐
          │                               │
   N_u  [2→50→50→50→50→1]        N_f  [2→50→50→50→1]
          │  (tanh, Xavier)               │  (tanh, Xavier)
          │                               │
     u_NN(x,t)                       f_NN(x,t)
          │                               │
          └───────────────┬───────────────┘
                          │
              r = u_t − α·u_xx − f   →   L_PDE ≈ 0
```

As derivadas $u_t$, $u_x$ e $u_{xx}$ são calculadas por **diferenciação automática exata** (`torch.autograd.grad` com `create_graph=True`), sem erros de truncamento ou discretização espacial:

```python
xt = xt.clone().detach().requires_grad_(True)
u  = self.u_net(self._normalize(xt))
f  = self.f_net(self._normalize(xt))

grad_u   = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
u_x, u_t = grad_u[:, 0:1], grad_u[:, 1:2]
u_xx     = torch.autograd.grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]

residual = u_t - self.alpha * u_xx - f   # deve ser ≈ 0 em todo Q_T
```

---

## Função de Perda

$$\mathcal{L} = w_{\rm BC}\,\mathcal{L}_{\rm BC} + w_{\rm PDE}\,\mathcal{L}_{\rm PDE} + w_{\rm reg}\,\mathcal{L}_{\rm reg} + w_u\,\mathcal{L}_{u\text{-obs}} + w_f\,\mathcal{L}_{f\text{-obs}}$$

| Componente | Expressão | Papel |
|---|---|---|
| $\mathcal{L}_{\rm BC}$ | MSE em $u(0,t)=u(1,t)=u(x,0)=0$ | Satisfazer condições de contorno e inicial |
| $\mathcal{L}_{\rm PDE}$ | MSE do resíduo $u_t - \alpha u_{xx} - f$ em 22 500 pts | Impor a equação governante |
| $\mathcal{L}_{\rm reg}$ | MSE de $\|\nabla_{x,t} f_{\rm NN}\|^2$ | Suavidade de $f$ — regularização implícita |
| $\mathcal{L}_{u\text{-obs}}$ | MSE em grade $60\times60$ de $u^*$ | **Fixar escala e forma** de $u_{\rm NN}$ |
| $\mathcal{L}_{f\text{-obs}}$ | MSE em 200 amostras aleatórias de $f^*$ | Supervisão direta do campo a identificar |

> **Por que $\mathcal{L}_{u\text{-obs}}$ é indispensável?**
> Com condições de contorno e inicial homogêneas, o sistema é invariante por $(u,f)\mapsto(ku,kf)$ para qualquer $k\in\mathbb{R}$. Sem observações internas com magnitude absoluta conhecida, a PINN converge para uma escala arbitrária. Os 3 600 pontos em grade uniforme garantem ancoragem em toda fatia temporal.

**Pesos padrão** (ajustáveis por experimento no JSON):

```
w_BC = 100.0    w_PDE = 10.0    w_reg = 1e-4    w_u = 100.0    w_f = 50.0
```

**Protocolo de treinamento:**

- Otimizador: **Adam** ($\beta_1=0.9$, $\beta_2=0.999$, $\eta_0=10^{-3}$)
- Scheduler: decaimento por passo, fator $\gamma=0.9$ a cada 2 000 steps
- Steps padrão: **8 000**; batch de 1 000 pontos PDE por passo
- Métrica principal: erro $L^2$ relativo em $f$, calculado em malha de teste $200\times200$

$$E_f = \frac{\|f_{\rm NN} - f^\star\|_{L^2}}{\|f^\star\|_{L^2}}$$

---

## Configuração via JSON

Todos os parâmetros do experimento são controlados por um único arquivo JSON. A hierarquia de precedência é: **CLI > JSON > defaults internos**.

```json
{
  "experiment": {
    "case":        "example_1",
    "domain":      [0, 1, 0, 1],
    "alpha":       1.0,
    "noise_level": 0.0
  },
  "data": {
    "N_train_x": 150,
    "N_train_t": 150,
    "N_bc":      1000,
    "N_f_obs":   200,
    "N_obs_ux":  60,
    "N_obs_ut":  60,
    "N_test_x":  200,
    "N_test_t":  200
  },
  "model": {
    "net_u": { "layers": [2, 50, 50, 50, 50, 1], "activation": "tanh" },
    "net_f": { "layers": [2, 50, 50, 50, 1],     "activation": "tanh" }
  },
  "training": {
    "n_steps":             8000,
    "batch_size":          1000,
    "log_interval":        500,
    "plot_interval":       1000,
    "checkpoint_interval": 2000,
    "loss_weights": {
      "bc": 100.0, "pde": 10.0, "reg": 1e-4, "u_obs": 100.0, "f_obs": 50.0
    },
    "optimizer":  { "type": "adam", "lr": 0.001 },
    "scheduler":  { "type": "step", "step_size": 2000, "gamma": 0.9 }
  },
  "output": {
    "base_dir":      "outputs",
    "gif_fps":       5,
    "remove_frames": false
  }
}
```

**Ativações disponíveis:** `tanh`, `relu`, `gelu`, `elu`, `silu`, `softplus`, `sigmoid`, `sin`.

---

## Saídas Geradas

Cada execução cria automaticamente uma pasta datada em `outputs/`:

```
outputs/example_1_20260101_120000/
│
├── config.json                      # Configuração exata usada nesta execução
├── training_log.txt                 # Log completo passo a passo
├── training_history.json            # Histórico de losses e erros L2 (JSON)
├── 00_ground_truth.png              # Heatmaps de u* e f* verdadeiros
├── convergence_history.png          # Curvas de loss e erro L2 ao longo do treino
│
├── plots/
│   ├── source_term/
│   │   ├── frame_0001.png ...       # [True f | Identified f | |erro|] a cada plot_interval
│   │   ├── f_evolution.gif          # Animação cíclica da identificação de f(x,t)
│   │   ├── final_comparison.png     # Comparação final em heatmap
│   │   └── final_analysis.png       # Slices 1D em t=0.25, t=0.75, x=0.5 + scatter
│   │
│   └── pde_solution/
│       ├── frame_0001.png ...       # [True u | Predicted u | |erro|]
│       ├── u_evolution.gif          # Animação cíclica de u(x,t)
│       ├── final_comparison.png
│       └── final_analysis.png
│
└── checkpoints/
    ├── checkpoint_step002000.pt
    ├── checkpoint_step004000.pt
    ├── checkpoint_step006000.pt
    └── checkpoint_final.pt
```

Os GIFs usam **escala de cores fixada nos valores de $f^{\star}$** em todos os frames, permitindo comparação visual direta sem artefatos de reescalonamento entre passos.

---

## Testes

A suite `test_system.py` cobre 9 grupos de testes e deve passar completamente antes de rodar experimentos longos, especialmente em ambientes HPC:

```bash
python test_system.py        # saída resumida
python test_system.py -v     # verbose — exibe traceback completo em caso de falha
```

| Teste | O que verifica |
|---|---|
| Importações | Dependências instaladas corretamente |
| Soluções analíticas | Resíduo $< 10^{-4}$ para todos os 5 pares $(u^{\star}, f^{\star})$ |
| Gerador de dados | Shapes e ranges corretos de todos os conjuntos |
| Gerador com ruído | Ruído gaussiano escalado e aplicado corretamente |
| Arquitetura PINN | Forward pass, contagem de parâmetros, dimensões de saída |
| Loss functions | BC, PDE, reg, u-obs e f-obs retornam escalares positivos |
| Métricas de erro | Erro $L^2$ relativo zero para predição perfeita |
| Trainer (smoke test) | 50 steps de treinamento sem NaN, loss decrescente |
| GIF cíclico | Arquivos `.gif` para $f$ e $u$ criados e não-vazios |

O script retorna `exit code 0` se todos os testes passam, e `exit code 1` caso contrário — compatível com pipelines CI/CD.

---

## Ablation Study

Para comparar sistematicamente variações de hiperparâmetros, use o helper `run_ablation_study` de `pinn_trainer`:

```python
from pinn_trainer import run_ablation_study

ablations = [
    {"name": "w_pde_1",   "training.loss_weights.pde": 1.0},
    {"name": "w_pde_10",  "training.loss_weights.pde": 10.0},
    {"name": "w_pde_100", "training.loss_weights.pde": 100.0},
]

results = run_ablation_study(
    base_config = config,
    ablations   = ablations,
    data        = data,
    output_base = "outputs/ablation_pde_weight"
)
```

Os overrides usam **notação aninhada por ponto** (`"training.loss_weights.pde"`), permitindo alterar qualquer campo do config sem criar arquivos JSON separados. Cada variante é salva em subpasta própria com log e checkpoints independentes.

Ao final, o helper imprime automaticamente uma tabela comparativa:

```
============================================================
  RESUMO ABLACAO
============================================================
  Nome                              Err_u       Err_f
------------------------------------------------------------
  w_pde_1                        0.082341    0.134872
  w_pde_10                       0.031204    0.047631
  w_pde_100                      0.028910    0.055203
============================================================
```

---

## Referências

Para a formulação matemática completa e as referências utilizadas, acesse: [Formulação Matemática](https://github.com/mateusjmd/srcPINN/blob/main/Formula%C3%A7%C3%A3o_Matem%C3%A1tica.pdf)