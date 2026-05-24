# Chladni - plaque libre carrée, modes (m,n)
# 1-9 modes, G = bascule gravité/micro-g, R redisperse, espace pause.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

L           = 1.0
N_GRAINS    = 12000   # bon compromis nettete / fluidite
N_BG        = 300
drift_rate  = 0.0025  # plus grand = convergence plus rapide mais grains qui oscillent
noise_scale = 0.003

rng = np.random.default_rng()

# (m, n, sign, nom). sign='A' antisym, 'S' sym. Pour l'instant je n'utilise que A.
PRESETS = {
    "1": (2, 1, "A", "croix"),
    "2": (3, 1, "A", "lobes diagonaux"),
    "3": (3, 2, "A", "pentagone"),
    "4": (4, 1, "A", "etoile 4 branches"),
    "5": (4, 3, "A", "rosace"),
    "6": (5, 2, "A", "papillon"),
    "7": (5, 4, "A", "fleur"),
    "8": (6, 1, "A", "couronne"),
    "9": (7, 4, "A", "complexe"),
}
state = {"m": 3, "n": 2, "sign": "A", "name": "pentagone",
         "gravity": True, "running": True}

def u_field(x, y, m, n, sign):
    cx_m = np.cos(m * np.pi * x / L); cy_m = np.cos(m * np.pi * y / L)
    cx_n = np.cos(n * np.pi * x / L); cy_n = np.cos(n * np.pi * y / L)
    if sign == "A":
        return cx_m * cy_n - cx_n * cy_m
    else:
        return cx_m * cy_n + cx_n * cy_m

def grad_V(x, y, m, n, sign):
    # gradient analytique de V = u^2 : dV/dx = 2 u du/dx
    # (j'avais d'abord pris np.gradient sur une grille, beaucoup plus lent)
    sx_m = np.sin(m * np.pi * x / L); cx_m = np.cos(m * np.pi * x / L)
    sy_m = np.sin(m * np.pi * y / L); cy_m = np.cos(m * np.pi * y / L)
    sx_n = np.sin(n * np.pi * x / L); cx_n = np.cos(n * np.pi * x / L)
    sy_n = np.sin(n * np.pi * y / L); cy_n = np.cos(n * np.pi * y / L)
    if sign == "A":
        u   = cx_m * cy_n - cx_n * cy_m
        ux  = -(m*np.pi/L) * sx_m * cy_n + (n*np.pi/L) * sx_n * cy_m
        uy  = -(n*np.pi/L) * cx_m * sy_n + (m*np.pi/L) * cx_n * sy_m
    else:
        # cas sym - jamais utilisé pour l'instant (pas dans les presets)
        u   = cx_m * cy_n + cx_n * cy_m
        ux  = -(m*np.pi/L) * sx_m * cy_n - (n*np.pi/L) * sx_n * cy_m
        uy  = -(n*np.pi/L) * cx_m * sy_n - (m*np.pi/L) * cx_n * sy_m
    return 2 * u * ux, 2 * u * uy

_xg = np.linspace(0, L, N_BG)
_yg = np.linspace(0, L, N_BG)
_Xg, _Yg = np.meshgrid(_xg, _yg, indexing="ij")
def background_field(m, n, sign):
    u = u_field(_Xg, _Yg, m, n, sign)
    return np.abs(u) / (np.abs(u).max() + 1e-12)

grains = rng.uniform(0.01, L - 0.01, (N_GRAINS, 2))
def redisperse():
    grains[:] = rng.uniform(0.01, L - 0.01, (N_GRAINS, 2))

def step():
    # Langevin: drift +/- grad(u^2) + bruit gaussien
    gx, gy = grad_V(grains[:, 0], grains[:, 1],
                    state["m"], state["n"], state["sign"])
    sign_drift = -1.0 if state["gravity"] else +1.0   # gravité -> noeuds, micro-g -> ventres
    grains[:, 0] += sign_drift * drift_rate * gx + rng.normal(0, noise_scale, N_GRAINS)
    grains[:, 1] += sign_drift * drift_rate * gy + rng.normal(0, noise_scale, N_GRAINS)
    # FIXME: au mode (7,4) quelques grains restent collés au bord, à creuser
    np.clip(grains[:, 0], 0.005, L - 0.005, out=grains[:, 0])
    np.clip(grains[:, 1], 0.005, L - 0.005, out=grains[:, 1])

plt.rcParams["axes.facecolor"]   = "#08080c"
plt.rcParams["figure.facecolor"] = "#08080c"
plt.rcParams["text.color"]       = "white"

fig, ax = plt.subplots(figsize=(8.5, 9))
plt.subplots_adjust(left=0.05, right=0.97, top=0.91, bottom=0.05)

cmap_bg = LinearSegmentedColormap.from_list(
    "plaque",
    [(0.00, "#08080c"),
     (0.30, "#1a2030"),
     (0.70, "#36486c"),
     (1.00, "#6b8ac8")])
bg = ax.imshow(background_field(state["m"], state["n"], state["sign"]).T,
               origin="lower", extent=(0, L, 0, L),
               cmap=cmap_bg, vmin=0, vmax=1, alpha=0.85,
               interpolation="bilinear", zorder=1)

sand = ax.scatter(grains[:, 0], grains[:, 1], s=1.4, c="#fff6d8",
                  alpha=0.9, edgecolors="none", zorder=3)

ax.set_aspect("equal")
ax.set_xlim(0, L); ax.set_ylim(0, L)
ax.set_xticks([]); ax.set_yticks([])
title = ax.set_title("", color="white", fontsize=11)

def on_key(event):
    if event.key in PRESETS:
        m, n, sign, name = PRESETS[event.key]
        state["m"], state["n"], state["sign"], state["name"] = m, n, sign, name
        bg.set_data(background_field(m, n, sign).T)
    elif event.key in ("g", "G"):
        state["gravity"] = not state["gravity"]
    elif event.key == " ":
        state["running"] = not state["running"]
    elif event.key in ("r", "R"):
        redisperse()
    elif event.key in ("q", "Q"):
        plt.close(fig)

fig.canvas.mpl_connect("key_press_event", on_key)

substeps = 3
def update(_frame):
    if state["running"]:
        for _ in range(substeps):
            step()
    sand.set_offsets(grains)
    grav_label = "TERRE - sable -> noeuds" if state["gravity"] \
                 else "MICRO-G - sable -> ventres"
    title.set_text(
        f"Chladni plaque libre   mode ({state['m']},{state['n']}) - {state['name']}\n"
        f"{grav_label}\n"
        f"1-9 modes | G gravite | R disperser | SPACE pause | Q quitter"
    )
    return sand, bg, title

ani = FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False)
plt.show()
