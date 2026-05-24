import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 400
L = 1.0
r = 0.010   # rayon des billes
m = 1.0
v0 = 1.0
dt = 0.001
substeps = 3
# TODO: ajouter un slider pour faire varier T en live

rng = np.random.default_rng()

def init_state():
    # placement sur une grille reguliere + petit jitter (sinon des billes
    # initialement tangentes explosent au premier pas)
    grid_n = int(np.ceil(np.sqrt(N)))
    spacing = (L - 2*r) / grid_n
    ix = np.arange(N) // grid_n
    iy = np.arange(N) %  grid_n
    p = np.column_stack([r + (ix + 0.5) * spacing,
                         r + (iy + 0.5) * spacing])
    p += rng.uniform(-spacing*0.1, spacing*0.1, (N, 2))
    # vitesses monodisperses |v|=v0, direction aleatoire : etat initial
    # tres loin de Maxwell-Boltzmann, ce qui permet de bien voir la relaxation
    theta = rng.uniform(0, 2*np.pi, N)
    v = np.column_stack([v0 * np.cos(theta), v0 * np.sin(theta)])
    return p, v

pos, vel = init_state()
momentum_wall = 0.0
collision_time_window = 0.0

def step():
    global pos, vel, momentum_wall, collision_time_window
    pos += vel * dt
    collision_time_window += dt

    # rebond sur les 4 parois - mesure aussi l'impulsion echangee (pour calculer P)
    for k in (0, 1):
        too_low = pos[:, k] < r
        too_high = pos[:, k] > L - r
        momentum_wall += 2 * m * np.sum(np.abs(vel[too_low, k]))
        momentum_wall += 2 * m * np.sum(np.abs(vel[too_high, k]))
        vel[too_low, k] = np.abs(vel[too_low, k])
        vel[too_high, k] = -np.abs(vel[too_high, k])
        pos[too_low, k] = r
        pos[too_high, k] = L - r

    # collisions billes-billes : matrice des distances O(N^2) puis masque
    # des paires en contact. Cell-list serait plus malin pour N >> 1000,
    # ici N=400 le naif vectorise reste plus rapide en numpy.
    diff = pos[None, :, :] - pos[:, None, :]
    dist2 = (diff**2).sum(axis=-1)
    np.fill_diagonal(dist2, np.inf)
    mask = (dist2 < (2*r)**2) & (np.triu(np.ones((N, N), dtype=bool), 1))
    i_idx, j_idx = np.where(mask)
    for i, j in zip(i_idx, j_idx):
        dxy = pos[j] - pos[i]
        d = np.sqrt((dxy**2).sum())
        if d < 1e-12:
            continue
        n = dxy / d
        v_rel = (vel[j] - vel[i]) @ n
        if v_rel < 0:   # ils se rapprochent => collision elastique
            vel[i] += v_rel * n
            vel[j] -= v_rel * n
            # on les separe pour eviter qu'ils se chevauchent au pas suivant
            overlap = 2*r - d
            pos[i] -= 0.5 * overlap * n
            pos[j] += 0.5 * overlap * n

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7),
                               gridspec_kw=dict(width_ratios=[1, 1.1]))
plt.subplots_adjust(left=0.05, right=0.97, top=0.91, bottom=0.09, wspace=0.15)

ax1.set_xlim(0, L); ax1.set_ylim(0, L); ax1.set_aspect("equal")
ax1.set_xticks([]); ax1.set_yticks([])
ax1.set_facecolor("#101018")
for spine in ax1.spines.values():
    spine.set_color("white"); spine.set_linewidth(1.5)
ax1.set_title(f"Boite : N = {N} billes en collisions elastiques", color="black")

speeds = np.sqrt((vel**2).sum(axis=1))
scatter = ax1.scatter(pos[:, 0], pos[:, 1], s=(2 * r * 700)**2,
                      c=speeds, cmap="inferno",
                      vmin=0, vmax=2.8*v0,
                      edgecolors="white", linewidths=0.3, zorder=2)

ax2.set_xlim(0, 3*v0); ax2.set_ylim(0, 1.8)
ax2.set_xlabel("vitesse |v|")
ax2.set_ylabel("densite de probabilite")
ax2.set_title("Distribution des vitesses (histo)\n"
              "vs Maxwell-Boltzmann 2D theorique",
              color="black")
ax2.grid(alpha=0.3)

n_bins = 40
v_edges = np.linspace(0, 3*v0, n_bins+1)
v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])
bin_w = v_edges[1] - v_edges[0]
bars = ax2.bar(v_centers, np.zeros(n_bins), width=bin_w*0.95,
               color="#3a85cc", edgecolor="white", alpha=0.85,
               label=f"simulation ({N} billes)")
mb_line, = ax2.plot([], [], "-", color="#cc4f33", lw=2.2,
                    label="Maxwell-Boltzmann 2D")
ax2.legend(loc="upper right")

state = {"running": True}

def on_key(event):
    global pos, vel, momentum_wall, collision_time_window
    if event.key == " ":
        state["running"] = not state["running"]
    elif event.key in ("r", "R"):
        pos[:], vel[:] = init_state()
        momentum_wall = 0.0
        collision_time_window = 0.0
    elif event.key in ("q", "Q"):
        plt.close(fig)

fig.canvas.mpl_connect("key_press_event", on_key)

sim_time = 0.0

def update(_frame):
    global momentum_wall, collision_time_window, sim_time
    if state["running"]:
        for _ in range(substeps):
            step()
        sim_time += substeps * dt

    speeds = np.sqrt((vel**2).sum(axis=1))
    scatter.set_offsets(pos)
    scatter.set_array(speeds)

    counts, _ = np.histogram(speeds, bins=v_edges, density=True)
    for bar, c in zip(bars, counts):
        bar.set_height(c)

    E_tot = 0.5 * m * np.sum(speeds**2)
    kT = E_tot / N
    v_arr = np.linspace(0.001, 3*v0, 300)
    f_mb = (m * v_arr / kT) * np.exp(-m * v_arr**2 / (2 * kT))
    mb_line.set_data(v_arr, f_mb)

    return scatter, mb_line, *bars

ani = FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False)
plt.show()