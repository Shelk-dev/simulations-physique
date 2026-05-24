# Gray-Scott reaction-diffusion
# dU/dt = Du lap(U) - UV^2 + F(1-U)
# dV/dt = Dv lap(V) + UV^2 - (F+K)V
# 1-9 = presets, space pause, R reset, Q quit

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N  = 220
dx = 1.0
dt = 1.0
Du, Dv = 0.16, 0.08    # condition Du > Dv pour l'instabilite de Turing

# (F, K, nom) - trouvés a la main en jouant avec les params
PRESETS = {
    "1": (0.0367, 0.0649, "spots mobiles"),
    "2": (0.0250, 0.0550, "solitons"),
    "3": (0.0400, 0.0600, "labyrinthe"),
    "4": (0.0180, 0.0510, "ondes"),
    "5": (0.0545, 0.0620, "rayures de zebre"),
    "6": (0.0367, 0.0600, "taches de leopard"),
    "7": (0.0620, 0.0620, "mitose (chaos)"),
    "8": (0.0140, 0.0540, "taches mobiles"),
    "9": (0.0390, 0.0580, "trous"),
}

state = {"F": PRESETS["3"][0], "K": PRESETS["3"][1],
         "name": PRESETS["3"][2], "running": True}

rng = np.random.default_rng()

U = np.ones((N, N))
V = np.zeros((N, N))

def reset_all():
    U[:] = 1.0
    V[:] = 0.0
    # carre central + bruit pour casser la symetrie (sinon rien ne demarre)
    c = N // 2
    s = 18
    U[c-s:c+s, c-s:c+s] = 0.50 + 0.02 * rng.standard_normal((2*s, 2*s))
    V[c-s:c+s, c-s:c+s] = 0.25 + 0.02 * rng.standard_normal((2*s, 2*s))
    # essayer aussi avec plusieurs taches aleatoires ?
    # for _ in range(5):
    #     i, j = rng.integers(20, N-20, 2)
    #     V[i-5:i+5, j-5:j+5] = 0.5

reset_all()

def laplacien(Z):
    # 5 points, bords periodiques
    return (np.roll(Z,  1, 0) + np.roll(Z, -1, 0) +
            np.roll(Z,  1, 1) + np.roll(Z, -1, 1) - 4.0 * Z) / dx**2

def step():
    Lu = laplacien(U)
    Lv = laplacien(V)
    uvv = U * V * V
    U[:] += dt * (Du * Lu - uvv + state["F"] * (1.0 - U))
    V[:] += dt * (Dv * Lv + uvv - (state["F"] + state["K"]) * V)

plt.rcParams["axes.facecolor"]   = "#08080c"
plt.rcParams["figure.facecolor"] = "#08080c"
plt.rcParams["text.color"]       = "white"

fig, ax = plt.subplots(figsize=(8.5, 9))
plt.subplots_adjust(left=0.04, right=0.97, top=0.91, bottom=0.04)

img = ax.imshow(V.T, origin="lower", cmap="inferno",
                vmin=0, vmax=0.5, interpolation="bilinear")
ax.set_xticks([]); ax.set_yticks([])
title = ax.set_title("", color="white", fontsize=11)

def on_key(event):
    if event.key in PRESETS:
        F, K, name = PRESETS[event.key]
        state["F"], state["K"], state["name"] = F, K, name
        reset_all()
    elif event.key == " ":
        state["running"] = not state["running"]
    elif event.key in ("r", "R"):
        reset_all()
    elif event.key in ("q", "Q"):
        plt.close(fig)

fig.canvas.mpl_connect("key_press_event", on_key)

substeps = 30  # iterations par frame (plus eleve = animation plus rapide)

def update(_frame):
    if state["running"]:
        for _ in range(substeps):
            step()
    img.set_data(V.T)
    title.set_text(
        f"Gray-Scott   F = {state['F']:.4f}    K = {state['K']:.4f}    {state['name']}\n"
        f"1-9 regimes  |  R reset  |  SPACE pause  |  Q quitter"
    )
    return img, title

ani = FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False)
plt.show()
