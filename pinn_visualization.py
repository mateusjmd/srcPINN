"""
Modulo de Visualizacao para a PINN de Identificacao de Termo Fonte

Estrutura de saida:
    output_dir/
        plots/
            source_term/
                frame_0001.png  ...  (frames para o GIF de f)
                final_comparison.png
                final_analysis.png
                f_evolution.gif
            pde_solution/
                frame_0001.png  ...  (frames para o GIF de u)
                final_comparison.png
                final_analysis.png
                u_evolution.gif
        00_ground_truth.png
        convergence_history.png

NOTA: Apenas caracteres ASCII para evitar erros de encoding.
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.size"] = 11
import matplotlib.pyplot as plt
import torch


# =============================================================================
# CLASSE DE VISUALIZACAO
# =============================================================================

class PINNVisualizer:
    """
    Gerencia todos os plots e animacoes da PINN.

    Subpastas criadas automaticamente:
        <output_dir>/plots/source_term/   -- frames e GIF de f(x,t)
        <output_dir>/plots/pde_solution/  -- frames e GIF de u(x,t)

    Parametros
    ----------
    output_dir : str  pasta raiz do experimento
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

        # Subpastas de plots
        self.plots_dir  = os.path.join(output_dir, "plots")
        self.dir_f      = os.path.join(self.plots_dir, "source_term")
        self.dir_u      = os.path.join(self.plots_dir, "pde_solution")

        for d in [output_dir, self.plots_dir, self.dir_f, self.dir_u]:
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    @staticmethod
    def _to_np(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    def _path(self, name: str) -> str:
        """Caminho na raiz do output (para ground truth e convergencia)."""
        return os.path.join(self.output_dir, name)

    def _path_f(self, name: str) -> str:
        return os.path.join(self.dir_f, name)

    def _path_u(self, name: str) -> str:
        return os.path.join(self.dir_u, name)

    # ------------------------------------------------------------------
    # Ground truth (salvo na raiz, unico para ambos)
    # ------------------------------------------------------------------

    def plot_ground_truth(self, X, T, u_mesh, f_mesh):
        """Plota u(x,t) e f(x,t) verdadeiros lado a lado."""
        X = self._to_np(X); T = self._to_np(T)
        u = self._to_np(u_mesh); f = self._to_np(f_mesh)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        _contour(axes[0], X, T, u, cmap="viridis",
                 title=r"True $u(x,t)$", xlabel=r"$x$", ylabel=r"$t$")
        _contour_div(axes[1], X, T, f,
                     title=r"True $f(x,t)$", xlabel=r"$x$", ylabel=r"$t$")
        plt.tight_layout()
        plt.savefig(self._path("00_ground_truth.png"),
                    dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Frames intermediarios -- Termo Fonte f(x,t)
    # ------------------------------------------------------------------

    def plot_frame_f(self, X, T, f_true, f_pred, step: int,
                     frame_idx: int, err_f: float = None):
        """
        Salva um frame de comparacao para f(x,t):
            [True f | Predicted f | Abs. Error]
        Arquivo: plots/source_term/frame_XXXX.png
        """
        X      = self._to_np(X); T      = self._to_np(T)
        f_true = self._to_np(f_true);   f_pred = self._to_np(f_pred)
        error  = np.abs(f_pred - f_true)

        # Limites fixos definidos APENAS por f_true -- o painel True nao
        # muda entre frames e o predito e avaliado na mesma escala
        vmax_true    = max(abs(float(f_true.min())), abs(float(f_true.max()))) + 1e-12
        levels_fixed = np.linspace(-vmax_true, vmax_true, 21)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=100,
                                 tight_layout=False)
        fig.subplots_adjust(left=0.05, right=0.97,
                            top=0.88, bottom=0.12, wspace=0.35)

        # (a) True -- colorbar fixo, identico em todos os frames
        cf1 = axes[0].contourf(X, T, f_true, levels=levels_fixed,
                               cmap="RdBu_r", extend="neither")
        axes[0].set_title(r"(a) True $f(x,t)$", fontsize=13)
        _label_axes(axes[0]); plt.colorbar(cf1, ax=axes[0], format="%.3f")

        # (b) Predicted -- mesma escala do True; extend="both" expoe outliers
        cf2 = axes[1].contourf(X, T, f_pred, levels=levels_fixed,
                               cmap="RdBu_r", extend="both")
        title_b = r"(b) Identified $f(x,t)$ -- step %d" % step
        if err_f is not None:
            title_b += "\n$L_2$ err: %.4f" % err_f
        axes[1].set_title(title_b, fontsize=12)
        _label_axes(axes[1]); plt.colorbar(cf2, ax=axes[1], format="%.3f")

        # (c) Erro absoluto -- escala propria (varia entre frames, esperado)
        cf3 = axes[2].contourf(X, T, error, levels=20,
                               cmap="Reds", extend="max")
        axes[2].set_title(
            r"(c) Abs. Error $|f_\mathrm{pred} - f_\mathrm{true}|$"
            + ("\nmax: %.4f" % error.max()), fontsize=12)
        _label_axes(axes[2]); plt.colorbar(cf3, ax=axes[2], format="%.4f")

        fname = "frame_%04d.png" % frame_idx
        plt.savefig(self._path_f(fname),
                    dpi=100, bbox_inches=None, facecolor="white")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Frames intermediarios -- Solucao PDE u(x,t)
    # ------------------------------------------------------------------

    def plot_frame_u(self, X, T, u_true, u_pred, step: int,
                     frame_idx: int, err_u: float = None):
        """
        Salva um frame de comparacao para u(x,t):
            [True u | Predicted u | Abs. Error]
        Arquivo: plots/pde_solution/frame_XXXX.png

        IMPORTANTE: os limites do colormap de (a) e (b) sao fixados
        exclusivamente por u_true, de modo que o painel "True" nao
        mude visualmente entre frames e a convergencia do predito
        seja legivel na mesma escala.
        """
        X      = self._to_np(X); T      = self._to_np(T)
        u_true = self._to_np(u_true);   u_pred = self._to_np(u_pred)
        error  = np.abs(u_pred - u_true)

        # Limites fixos definidos APENAS por u_true -- nao mudam entre frames
        vmin_true = float(u_true.min())
        vmax_true = float(u_true.max())
        levels_fixed = np.linspace(vmin_true, vmax_true, 21)

        # Margem de extensao para o predito (permite valores fora do range)
        margin = 0.1 * (vmax_true - vmin_true + 1e-12)
        vmin_ext = vmin_true - margin
        vmax_ext = vmax_true + margin

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=100,
                                 tight_layout=False)
        fig.subplots_adjust(left=0.05, right=0.97,
                            top=0.88, bottom=0.12, wspace=0.35)

        # (a) True -- colorbar fixo, nao muda entre frames
        cf1 = axes[0].contourf(X, T, u_true, levels=levels_fixed,
                               cmap="viridis", extend="neither")
        axes[0].set_title(r"(a) True $u(x,t)$", fontsize=13)
        _label_axes(axes[0])
        plt.colorbar(cf1, ax=axes[0], format="%.3f")

        # (b) Predicted -- mesma escala do True para comparacao direta
        #     extend="both" revela regioes fora do range esperado
        cf2 = axes[1].contourf(X, T, u_pred, levels=levels_fixed,
                               cmap="viridis", extend="both")
        title_b = r"(b) Predicted $u(x,t)$ -- step %d" % step
        if err_u is not None:
            title_b += "\n$L_2$ err: %.4f" % err_u
        axes[1].set_title(title_b, fontsize=12)
        _label_axes(axes[1])
        plt.colorbar(cf2, ax=axes[1], format="%.3f")

        # (c) Erro absoluto -- escala propria (varia entre frames, ok)
        cf3 = axes[2].contourf(X, T, error, levels=20,
                               cmap="Reds", extend="max")
        axes[2].set_title(
            r"(c) Abs. Error $|u_\mathrm{pred} - u_\mathrm{true}|$"
            + ("\nmax: %.4f" % error.max()), fontsize=12)
        _label_axes(axes[2])
        plt.colorbar(cf3, ax=axes[2], format="%.4f")

        fname = "frame_%04d.png" % frame_idx
        plt.savefig(self._path_u(fname),
                    dpi=100, bbox_inches=None, facecolor="white")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Wrapper conveniente: salva ambos os frames no mesmo passo
    # ------------------------------------------------------------------

    def plot_frames(self, X, T, f_true, f_pred, u_true, u_pred,
                    step: int, frame_idx: int,
                    err_f: float = None, err_u: float = None):
        """Salva frame de f e frame de u para o mesmo step."""
        self.plot_frame_f(X, T, f_true, f_pred,
                          step=step, frame_idx=frame_idx, err_f=err_f)
        self.plot_frame_u(X, T, u_true, u_pred,
                          step=step, frame_idx=frame_idx, err_u=err_u)

    # ------------------------------------------------------------------
    # Convergencia (salvo na raiz)
    # ------------------------------------------------------------------

    def plot_training_history(self, history: dict):
        """Salva graficos de loss e erro relativo ao longo dos steps."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        axes[0].semilogy(history["loss_total"], lw=2,
                         color="steelblue", label="Total Loss")
        if "loss_bc" in history:
            axes[0].semilogy(history["loss_bc"], lw=1.2,
                             color="coral", ls="--", label="BC Loss")
        if "loss_pde" in history:
            axes[0].semilogy(history["loss_pde"], lw=1.2,
                             color="green", ls="-.", label="PDE Loss")
        axes[0].set_xlabel("Step", fontsize=12)
        axes[0].set_ylabel("Loss (log)", fontsize=12)
        axes[0].set_title("(a) Training Loss", fontsize=13)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3, ls="--")

        # Erros L2
        if "err_u" in history:
            axes[1].semilogy(history["err_u"], lw=2,
                             color="steelblue", label=r"$L_2$ err  $u(x,t)$")
        if "err_f" in history:
            axes[1].semilogy(history["err_f"], lw=2,
                             color="coral", ls="--",
                             label=r"$L_2$ err  $f(x,t)$  [target]")
        if history.get("err_u") and history.get("err_f"):
            eu = history["err_u"][-1]; ef = history["err_f"][-1]
            axes[1].text(0.98, 0.97,
                         "Final $e_u$: %.4f\nFinal $e_f$: %.4f" % (eu, ef),
                         transform=axes[1].transAxes, fontsize=10,
                         va="top", ha="right",
                         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))
        axes[1].set_xlabel("Step", fontsize=12)
        axes[1].set_ylabel(r"Relative $L_2$ Error (log)", fontsize=12)
        axes[1].set_title("(b) Convergence History", fontsize=13)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, ls="--")

        plt.tight_layout()
        plt.savefig(self._path("convergence_history.png"),
                    dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Analise final detalhada -- Termo Fonte f(x,t)
    # ------------------------------------------------------------------

    def plot_final_analysis_f(self, X, T, f_true, f_pred):
        """
        Painel 2x2 para f(x,t):
            (a) slice t=0.25   (b) slice t=0.75
            (c) slice x=0.5    (d) scatter
        Salvo em: plots/source_term/final_analysis.png
        """
        X = self._to_np(X); T = self._to_np(T)
        f_true = self._to_np(f_true); f_pred = self._to_np(f_pred)
        Nx, Nt = X.shape

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        for frac, ax, lbl in [(0.25, axes[0,0], "(a)"), (0.75, axes[0,1], "(b)")]:
            ti = int(frac * (Nt - 1))
            ax.plot(X[:,0], f_true[:,ti], "b-",  lw=2, label="True")
            ax.plot(X[:,0], f_pred[:,ti], "r--", lw=2, label="Identified")
            ax.set_xlabel(r"$x$", fontsize=12)
            ax.set_ylabel(r"$f(x,\, t=%.2f)$" % frac, fontsize=12)
            ax.set_title(r"%s Source at $t=%.2f$" % (lbl, frac), fontsize=12)
            ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

        xi = int(0.5 * (Nx - 1))
        axes[1,0].plot(T[0,:], f_true[xi,:], "b-",  lw=2, label="True")
        axes[1,0].plot(T[0,:], f_pred[xi,:], "r--", lw=2, label="Identified")
        axes[1,0].set_xlabel(r"$t$", fontsize=12)
        axes[1,0].set_ylabel(r"$f(x=0.5,\, t)$", fontsize=12)
        axes[1,0].set_title(r"(c) Source at $x=0.5$", fontsize=12)
        axes[1,0].legend(fontsize=10); axes[1,0].grid(True, alpha=0.3)

        axes[1,1].scatter(f_true.flatten(), f_pred.flatten(),
                          alpha=0.3, s=2, color="coral")
        lo = min(f_true.min(), f_pred.min()) * 1.05
        hi = max(f_true.max(), f_pred.max()) * 1.05
        axes[1,1].plot([lo,hi],[lo,hi], "k--", lw=2, label="Perfect")
        axes[1,1].set_xlabel(r"True $f$", fontsize=12)
        axes[1,1].set_ylabel(r"Predicted $f$", fontsize=12)
        axes[1,1].set_title(
            r"(d) Correlation $f_\mathrm{pred}$ vs $f_\mathrm{true}$",
            fontsize=12)
        axes[1,1].legend(fontsize=10); axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_aspect("equal")

        plt.tight_layout()
        plt.savefig(self._path_f("final_analysis.png"),
                    dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Analise final detalhada -- Solucao PDE u(x,t)
    # ------------------------------------------------------------------

    def plot_final_analysis_u(self, X, T, u_true, u_pred):
        """
        Painel 2x2 para u(x,t):
            (a) slice t=0.25   (b) slice t=0.75
            (c) slice x=0.5    (d) scatter
        Salvo em: plots/pde_solution/final_analysis.png
        """
        X = self._to_np(X); T = self._to_np(T)
        u_true = self._to_np(u_true); u_pred = self._to_np(u_pred)
        Nx, Nt = X.shape

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        for frac, ax, lbl in [(0.25, axes[0,0], "(a)"), (0.75, axes[0,1], "(b)")]:
            ti = int(frac * (Nt - 1))
            ax.plot(X[:,0], u_true[:,ti], "b-",  lw=2, label="True")
            ax.plot(X[:,0], u_pred[:,ti], "r--", lw=2, label="Predicted")
            ax.set_xlabel(r"$x$", fontsize=12)
            ax.set_ylabel(r"$u(x,\, t=%.2f)$" % frac, fontsize=12)
            ax.set_title(r"%s Solution at $t=%.2f$" % (lbl, frac), fontsize=12)
            ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

        xi = int(0.5 * (Nx - 1))
        axes[1,0].plot(T[0,:], u_true[xi,:], "b-",  lw=2, label="True")
        axes[1,0].plot(T[0,:], u_pred[xi,:], "r--", lw=2, label="Predicted")
        axes[1,0].set_xlabel(r"$t$", fontsize=12)
        axes[1,0].set_ylabel(r"$u(x=0.5,\, t)$", fontsize=12)
        axes[1,0].set_title(r"(c) Solution at $x=0.5$", fontsize=12)
        axes[1,0].legend(fontsize=10); axes[1,0].grid(True, alpha=0.3)

        axes[1,1].scatter(u_true.flatten(), u_pred.flatten(),
                          alpha=0.3, s=2, color="steelblue")
        lo = min(u_true.min(), u_pred.min()) * 1.05
        hi = max(u_true.max(), u_pred.max()) * 1.05
        axes[1,1].plot([lo,hi],[lo,hi], "k--", lw=2, label="Perfect")
        axes[1,1].set_xlabel(r"True $u$", fontsize=12)
        axes[1,1].set_ylabel(r"Predicted $u$", fontsize=12)
        axes[1,1].set_title(
            r"(d) Correlation $u_\mathrm{pred}$ vs $u_\mathrm{true}$",
            fontsize=12)
        axes[1,1].legend(fontsize=10); axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_aspect("equal")

        plt.tight_layout()
        plt.savefig(self._path_u("final_analysis.png"),
                    dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Comparacoes finais (alta qualidade, sem entrar no GIF)
    # ------------------------------------------------------------------
    def plot_final_comparison_f(self, X, T, f_true, f_pred, err_f=None):
        """Comparacao final de f(x,t) em alta resolucao."""
        X = self._to_np(X); T = self._to_np(T)
        f_true = self._to_np(f_true); f_pred = self._to_np(f_pred)
        error  = np.abs(f_pred - f_true)

        # Limites fixos por f_true apenas
        vmax_true = max(abs(float(f_true.min())), abs(float(f_true.max()))) + 1e-12
        levels    = np.linspace(-vmax_true, vmax_true, 21)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        cf1 = axes[0].contourf(X, T, f_true, levels=levels, cmap="RdBu_r", extend="neither")
        axes[0].set_title(r"(a) True $f(x,t)$", fontsize=13)
        _label_axes(axes[0]); plt.colorbar(cf1, ax=axes[0], format="%.3f")

        cf2 = axes[1].contourf(X, T, f_pred, levels=levels, cmap="RdBu_r", extend="both")
        lbl = r"(b) Identified $f(x,t)$"
        if err_f is not None: lbl += "\n$L_2$ err: %.4f" % err_f
        axes[1].set_title(lbl, fontsize=13)
        _label_axes(axes[1]); plt.colorbar(cf2, ax=axes[1], format="%.3f")

        cf3 = axes[2].contourf(X, T, error, levels=20, cmap="Reds", extend="max")
        axes[2].set_title(r"(c) Abs. Error $|f_{\mathrm{pred}} - f_{\mathrm{true}}|$"
                          + ("\nmax: %.4f" % error.max()), fontsize=13)
        _label_axes(axes[2]); plt.colorbar(cf3, ax=axes[2], format="%.4f")

        plt.tight_layout()
        plt.savefig(self._path_f("final_comparison.png"),
                    dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def plot_final_comparison_u(self, X, T, u_true, u_pred, err_u=None):
        """Comparacao final de u(x,t) em alta resolucao."""
        X = self._to_np(X); T = self._to_np(T)
        u_true = self._to_np(u_true); u_pred = self._to_np(u_pred)
        error  = np.abs(u_pred - u_true)

        # Limites fixos por u_true apenas
        vmin = float(u_true.min())
        vmax = float(u_true.max())
        levels = np.linspace(vmin, vmax, 21)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        cf1 = axes[0].contourf(X, T, u_true, levels=levels, cmap="viridis", extend="neither")
        axes[0].set_title(r"(a) True $u(x,t)$", fontsize=13)
        _label_axes(axes[0]); plt.colorbar(cf1, ax=axes[0], format="%.3f")

        cf2 = axes[1].contourf(X, T, u_pred, levels=levels, cmap="viridis", extend="both")
        lbl = r"(b) Predicted $u(x,t)$"
        if err_u is not None: lbl += "\n$L_2$ err: %.4f" % err_u
        axes[1].set_title(lbl, fontsize=13)
        _label_axes(axes[1]); plt.colorbar(cf2, ax=axes[1], format="%.3f")

        cf3 = axes[2].contourf(X, T, error, levels=20, cmap="Reds", extend="max")
        axes[2].set_title(r"(c) Abs. Error $|u_{\mathrm{pred}} - u_{\mathrm{true}}|$"
                          + ("\nmax: %.4f" % error.max()), fontsize=13)
        _label_axes(axes[2]); plt.colorbar(cf3, ax=axes[2], format="%.4f")

        plt.tight_layout()
        plt.savefig(self._path_u("final_comparison.png"),
                    dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ------------------------------------------------------------------
    # GIF ciclico (ping-pong, loop infinito) -- generico por subpasta
    # ------------------------------------------------------------------

    def _generate_gif_from_dir(self, frames_dir: str, output_name: str,
                               fps: int = 5, remove_frames: bool = False):
        """
        Gera GIF ciclico com ping-pong a partir dos frame_XXXX.png
        encontrados em `frames_dir`.

        Usa PIL diretamente para evitar o problema de shapes diferentes
        que ocorre com imageio no Windows quando bbox_inches='tight'.
        """
        try:
            from PIL import Image as PILImage
        except ImportError:
            print("Pillow nao encontrado. Instale: pip install Pillow")
            return None

        pattern = re.compile(r"^frame_(\d{4})\.png$")
        frame_files = sorted(
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if pattern.match(f)
        )

        if not frame_files:
            print(f"Nenhum frame encontrado em: {frames_dir}")
            return None

        # Ler e converter para RGB
        pil_raw = []
        for fp in frame_files:
            try:
                pil_raw.append(PILImage.open(fp).convert("RGB"))
            except Exception as e:
                print(f"Erro ao ler {fp}: {e}")

        if not pil_raw:
            return None

        # Normalizar tamanho (crop centralizado no menor denominador comum)
        w_min = min(im.width  for im in pil_raw)
        h_min = min(im.height for im in pil_raw)
        pil_frames = []
        for im in pil_raw:
            if im.width != w_min or im.height != h_min:
                left = (im.width  - w_min) // 2
                top  = (im.height - h_min) // 2
                im = im.crop((left, top, left + w_min, top + h_min))
            pil_frames.append(
                im.quantize(colors=256, method=PILImage.Quantize.MEDIANCUT)
            )

        # Ping-pong para transicao suave e sem salto abrupto no loop
        ping_pong = pil_frames + (pil_frames[-2:0:-1] if len(pil_frames) > 1 else [])

        gif_path = os.path.join(frames_dir, output_name)
        duration_ms = max(1, int(1000 / fps))

        try:
            ping_pong[0].save(
                gif_path,
                save_all=True,
                append_images=ping_pong[1:],
                loop=0,             # 0 = repeticao infinita
                duration=duration_ms,
                optimize=False,
            )
            n = len(pil_frames)
            print(f"GIF gerado: {gif_path}  ({n} frames + ping-pong, {fps} fps)")
        except Exception as e:
            print(f"Erro ao salvar GIF: {e}")
            return None

        if remove_frames:
            for fp in frame_files:
                try: os.remove(fp)
                except Exception: pass

        return gif_path

    def generate_gif_f(self, fps: int = 5, remove_frames: bool = False):
        """GIF de evolucao de f(x,t). Salvo em plots/source_term/f_evolution.gif"""
        return self._generate_gif_from_dir(
            self.dir_f, "f_evolution.gif", fps=fps, remove_frames=remove_frames
        )

    def generate_gif_u(self, fps: int = 5, remove_frames: bool = False):
        """GIF de evolucao de u(x,t). Salvo em plots/pde_solution/u_evolution.gif"""
        return self._generate_gif_from_dir(
            self.dir_u, "u_evolution.gif", fps=fps, remove_frames=remove_frames
        )


# =============================================================================
# FUNCOES AUXILIARES INTERNAS
# =============================================================================

def _label_axes(ax):
    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel(r"$t$", fontsize=13)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)


def _contour(ax, X, T, Z, cmap="viridis", title="", xlabel="", ylabel=""):
    cf = ax.contourf(X, T, Z, levels=20, cmap=cmap)
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlabel(xlabel, fontsize=13); ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.2, linestyle="--")
    plt.colorbar(cf, ax=ax, format="%.3f")
    return cf


def _contour_div(ax, X, T, Z, title="", xlabel="", ylabel=""):
    vmax = max(abs(Z.min()), abs(Z.max())) + 1e-12
    levels = np.linspace(-vmax, vmax, 21)
    cf = ax.contourf(X, T, Z, levels=levels, cmap="RdBu_r", extend="both")
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlabel(xlabel, fontsize=13); ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.2, linestyle="--")
    plt.colorbar(cf, ax=ax, format="%.3f")
    return cf
