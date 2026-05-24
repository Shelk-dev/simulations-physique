# jeu de la vie de Conway, avec dessin interactif de la config initiale
# clic pour dessiner, entrée lance, espace pause, R reset, 1-9 presets, Q quit

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import convolve2d

N = 100   # taille grille (au dessus de ~300 le fps tombe)

def placer_config(config):
    # met la config au centre
    global grid, mode_dessin, running
    grid[:] = 0
    h, w = config.shape
    y0 = (N - h) // 2
    x0 = (N - w) // 2
    grid[y0:y0+h, x0:x0+w] = config
    # si on charge un preset pendant que ça tourne, on repasse en mode dessin
    if not mode_dessin:
        mode_dessin = True
        running = False
    update_display()

glider = np.array([[0, 1, 0],
                   [0, 0, 1],
                   [1, 1, 1]])

double_glider = np.array([[0, 1, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [1, 1, 1, 0, 0],
                          [0, 0, 0, 0, 1],
                          [0, 0, 0, 1, 1]])

blinker = np.array([[1, 1, 1]])

bloc = np.array([[1, 1],
                 [1, 1]])

barre = np.array([[1, 1, 1, 1, 1]])

lwss = np.array([[0, 1, 0, 1],
                 [1, 0, 0, 0],
                 [1, 0, 0, 1],
                 [1, 1, 1, 0]])

# canon de Gosper
gosper = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0],
    [0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
])
PRESETS = {
    "1": glider,
    "2": double_glider,
    "3": blinker,
    "4": bloc,
    "5": barre,
    "6": lwss,
    "7": gosper,
}

grid = np.zeros((N, N), dtype=int)
mode_dessin = True
running = False
pause = False

# kernel des voisins (Moore)
KERNEL = np.array([[1,1,1],
                   [1,0,1],
                   [1,1,1]])

def update_display():
    img.set_data(grid)
    if mode_dessin:
        title.set_text("Dessinez (clic) - Entree pour lancer - 1-9 presets")
    else:
        title.set_text("Espace pause | R reset | 1-9 presets | Q quitter")
    plt.draw()

def step():
    # comptage des voisins par convolution (beaucoup plus rapide que 2 boucles for)
    # puis on applique les règles vectoriellement
    global grid
    voisins = convolve2d(grid, KERNEL, mode='same', boundary='fill', fillvalue=0)
    new_grid = np.zeros_like(grid)
    new_grid[(grid == 1) & ((voisins == 2) | (voisins == 3))] = 1
    new_grid[(grid == 0) & (voisins == 3)] = 1
    grid = new_grid

def reset_to_drawing():
    global grid, mode_dessin, running, pause
    grid[:] = 0
    mode_dessin = True
    running = False
    pause = False
    update_display()

def start_simulation():
    global mode_dessin, running, pause
    if mode_dessin:
        mode_dessin = False
        running = True
        pause = False
        update_display()

def on_click(event):
    if not mode_dessin:
        return
    if event.inaxes != ax:
        return
    x = int(event.xdata)
    y = int(event.ydata)
    if 0 <= x < N and 0 <= y < N:
        grid[y, x] = 1 - grid[y, x]
        update_display()

def on_key(event):
    global running, mode_dessin, pause, grid
    key = event.key

    if key in ("l", "L", "enter"):
        start_simulation()
        return

    if mode_dessin:
        if key in PRESETS:
            placer_config(PRESETS[key])
        elif key in ("r", "R"):
            reset_to_drawing()
        elif key in ("q", "Q"):
            plt.close(fig)
        return

    if key == " ":
        if running:
            running = False
            pause = True
        else:
            running = True
            pause = False
        update_display()
    elif key in ("r", "R"):
        reset_to_drawing()
    elif key in PRESETS:
        placer_config(PRESETS[key])
    elif key in ("q", "Q"):
        plt.close(fig)

def update(_frame):
    if not mode_dessin and running:
        step()
        update_display()
    return img, title

plt.rcParams["axes.facecolor"] = "#08080c"
plt.rcParams["figure.facecolor"] = "#08080c"
plt.rcParams["text.color"] = "white"

fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02)

img = ax.imshow(grid, cmap="plasma", interpolation="nearest", vmin=0, vmax=1)
ax.set_xticks([]); ax.set_yticks([])
title = ax.set_title("Dessinez (clic) - Entree pour lancer - 1-9 presets",
                     color="white", fontsize=12)

fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)

ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
plt.show()