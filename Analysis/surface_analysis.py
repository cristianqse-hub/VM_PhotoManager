import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

# ============================================================
# Configuración
# ============================================================
TXT_FILE = "datasets_lines.txt"   # ruta al fichero
DX = 1.877                        # separación entre puntos originales
DENSE_STEP = 0.1                  # paso fino para comparar splines
K_MIN, K_MAX = 4, 10              # rango de puntos del spline reducido

# ============================================================
# Interpolador spline cúbico
# ============================================================
try:
    from scipy.interpolate import CubicSpline

    def cubic_interp(x, y, x_new):
        cs = CubicSpline(x, y)
        return cs(x_new)

except Exception:
    from scipy.interpolate import interp1d

    def cubic_interp(x, y, x_new):
        f = interp1d(x, y, kind="cubic", fill_value="extrapolate")
        return f(x_new)

# ============================================================
# Lectura de datasets
# Cada línea del fichero = un dataset
# ============================================================
def load_datasets(txt_file):
    datasets = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            values = np.array([float(v) for v in s.split()], dtype=float)
            datasets.append(values)
    return datasets

datasets = load_datasets(TXT_FILE)
if not datasets:
    raise ValueError("No se encontraron datasets en el fichero.")

n_sets = len(datasets)
n_pts = len(datasets[0])

for i, z in enumerate(datasets):
    if len(z) != n_pts:
        raise ValueError(
            f"El dataset {i} no tiene el mismo número de puntos que los demás."
        )

x = np.arange(n_pts) * DX
x_dense = np.arange(x[0], x[-1] + DENSE_STEP, DENSE_STEP)

# ============================================================
# Utilidades
# ============================================================
def reduced_indices(n_total, k):
    idx = np.linspace(0, n_total - 1, k)
    idx = np.round(idx).astype(int)
    idx = np.unique(idx)

    while len(idx) < k:
        missing = k - len(idx)
        extra = np.setdiff1d(np.arange(n_total), idx)
        idx = np.sort(np.concatenate([idx, extra[:missing]]))
    return idx


def compute_full_and_reduced(z, k):
    y_full = cubic_interp(x, z, x_dense)

    idx = reduced_indices(len(z), k)
    x_sub = x[idx]
    z_sub = z[idx]
    y_red = cubic_interp(x_sub, z_sub, x_dense)

    max_diff = np.max(np.abs(y_full - y_red))
    return y_full, x_sub, z_sub, y_red, max_diff


def compute_stats_all_sets(k_values):
    means = []
    stds = []
    all_diffs = {}

    for k in k_values:
        diffs = []
        for z in datasets:
            _, _, _, _, max_diff = compute_full_and_reduced(z, k)
            diffs.append(max_diff)

        diffs = np.array(diffs)
        all_diffs[k] = diffs
        means.append(np.mean(diffs))
        stds.append(np.std(diffs))

    return np.array(means), np.array(stds), all_diffs


k_values = np.arange(K_MIN, K_MAX + 1)
mean_diffs, std_diffs, diffs_per_k = compute_stats_all_sets(k_values)

# ============================================================
# Figura y layout
# ============================================================
fig = plt.figure(figsize=(16, 8))

# ejes principales
ax_left = fig.add_axes([0.20, 0.12, 0.43, 0.80])
ax_right = fig.add_axes([0.68, 0.12, 0.28, 0.80])

# sliders verticales a la izquierda
ax_slider_dataset = fig.add_axes([0.04, 0.26, 0.025, 0.56])
ax_slider_k = fig.add_axes([0.09, 0.26, 0.025, 0.56])

# textbox alpha
ax_text_alpha = fig.add_axes([0.03, 0.13, 0.12, 0.05])

# textbox escala fuentes
ax_text_scale = fig.add_axes([0.03, 0.05, 0.12, 0.05])

# ============================================================
# Widgets
# ============================================================
slider_dataset = Slider(
    ax=ax_slider_dataset,
    label="Dataset",
    valmin=1,
    valmax=n_sets,
    valinit=1,
    valstep=1,
    orientation="vertical"
)

slider_k = Slider(
    ax=ax_slider_k,
    label="N pts",
    valmin=K_MIN,
    valmax=K_MAX,
    valinit=6,
    valstep=1,
    orientation="vertical"
)

text_alpha = TextBox(
    ax=ax_text_alpha,
    label="alpha",
    initial="0.20"
)

text_scale = TextBox(
    ax=ax_text_scale,
    label="font scale",
    initial="1.0"
)

# ============================================================
# Colores base en escala de grises
# de gris intermedio a negro
# ============================================================
gray_levels = np.linspace(0.55, 0.05, n_sets)
base_colors = [(g, g, g) for g in gray_levels]

# ============================================================
# Dibujo base subplot izquierdo
# ============================================================
base_lines = []
base_markers = []

initial_alpha = 0.20

for i, z in enumerate(datasets):
    y_dense = cubic_interp(x, z, x_dense)

    line, = ax_left.plot(
        x_dense, y_dense,
        color=base_colors[i],
        lw=1.5,
        alpha=initial_alpha,
        zorder=1
    )
    pts, = ax_left.plot(
        x, z,
        linestyle="None",
        marker="o",
        markersize=3.0,
        color=base_colors[i],
        alpha=initial_alpha,
        zorder=2
    )
    base_lines.append(line)
    base_markers.append(pts)

# overlays dinámicos en izquierda
selected_full_line, = ax_left.plot(
    [], [], color="blue", lw=2.5, alpha=0.95, zorder=5, label="Full spline"
)
selected_full_pts, = ax_left.plot(
    [], [], "o", color="blue", markersize=4, alpha=0.95, zorder=6
)

selected_red_line, = ax_left.plot(
    [], [], color="red", lw=2.2, alpha=0.95, zorder=7, label="Reduced spline"
)
selected_red_pts, = ax_left.plot(
    [], [], "o", color="red", markersize=6, alpha=0.95, zorder=8, label="Reduced points"
)

ax_left.set_title("Datasets + spline completo y spline reducido")
ax_left.set_xlabel("X")
ax_left.set_ylabel("Z")
ax_left.grid(True, alpha=0.25)
legend_left = ax_left.legend(loc="best")

# ============================================================
# Dibujo base subplot derecho
# media ± std + marcador dataset actual
# ============================================================
ax_right.errorbar(
    k_values,
    mean_diffs,
    yerr=std_diffs,
    marker="o",
    capsize=4,
    lw=1.8,
    label="Mean ± std"
)

selected_marker, = ax_right.plot(
    [], [], "o", color="red", markersize=10, zorder=10, label="Selected dataset"
)

info_text = ax_right.text(
    0.03,
    0.97,
    "",
    transform=ax_right.transAxes,
    va="top",
    ha="left",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.7")
)

ax_right.set_title("Max difference vs number of interpolated points")
ax_right.set_xlabel("Number of interpolated points")
ax_right.set_ylabel("Max difference")
ax_right.set_xticks(k_values)
ax_right.grid(True, alpha=0.3)
legend_right = ax_right.legend(loc="best")

# ============================================================
# Límites iniciales
# ============================================================
all_z = np.concatenate(datasets)
z_span = all_z.max() - all_z.min()
margin = 0.05 * z_span if z_span > 0 else 1.0

ax_left.set_xlim(x[0], x[-1])
ax_left.set_ylim(all_z.min() - margin, all_z.max() + margin)

ymax_right = np.max(mean_diffs + std_diffs) * 1.15
ax_right.set_xlim(K_MIN - 0.5, K_MAX + 0.5)
ax_right.set_ylim(0, ymax_right)

# ============================================================
# Funciones auxiliares widgets
# ============================================================
def get_alpha():
    try:
        a = float(text_alpha.text)
        return np.clip(a, 0.0, 1.0)
    except ValueError:
        return 0.20


def get_scale():
    try:
        s = float(text_scale.text)
        return max(0.2, s)
    except ValueError:
        return 1.0


def apply_font_scale(scale):
    title_fs = 14 * scale
    label_fs = 12 * scale
    tick_fs = 10 * scale
    legend_fs = 10 * scale
    widget_fs = 10 * scale
    info_fs = 10 * scale

    # títulos
    ax_left.title.set_fontsize(title_fs)
    ax_right.title.set_fontsize(title_fs)

    # labels ejes
    ax_left.xaxis.label.set_fontsize(label_fs)
    ax_left.yaxis.label.set_fontsize(label_fs)
    ax_right.xaxis.label.set_fontsize(label_fs)
    ax_right.yaxis.label.set_fontsize(label_fs)

    # ticks
    for label in ax_left.get_xticklabels() + ax_left.get_yticklabels():
        label.set_fontsize(tick_fs)

    for label in ax_right.get_xticklabels() + ax_right.get_yticklabels():
        label.set_fontsize(tick_fs)

    # leyendas
    if legend_left is not None:
        for txt in legend_left.get_texts():
            txt.set_fontsize(legend_fs)

    if legend_right is not None:
        for txt in legend_right.get_texts():
            txt.set_fontsize(legend_fs)

    # sliders
    slider_dataset.label.set_fontsize(widget_fs)
    slider_k.label.set_fontsize(widget_fs)

    # el texto numérico de los sliders
    if hasattr(slider_dataset, "valtext") and slider_dataset.valtext is not None:
        slider_dataset.valtext.set_fontsize(widget_fs)

    if hasattr(slider_k, "valtext") and slider_k.valtext is not None:
        slider_k.valtext.set_fontsize(widget_fs)

    # textbox labels
    text_alpha.label.set_fontsize(widget_fs)
    text_scale.label.set_fontsize(widget_fs)

    # texto dentro de textbox
    if hasattr(text_alpha, "text_disp") and text_alpha.text_disp is not None:
        text_alpha.text_disp.set_fontsize(widget_fs)

    if hasattr(text_scale, "text_disp") and text_scale.text_disp is not None:
        text_scale.text_disp.set_fontsize(widget_fs)

    # texto informativo
    info_text.set_fontsize(info_fs)

# ============================================================
# Update
# ============================================================
def update(_=None):
    alpha = get_alpha()
    scale = get_scale()
    ds_idx = int(slider_dataset.val) - 1
    k = int(slider_k.val)

    apply_font_scale(scale)

    # actualizar alpha de trazas base
    for line, pts in zip(base_lines, base_markers):
        line.set_alpha(alpha)
        pts.set_alpha(alpha)

    # dataset seleccionado
    z = datasets[ds_idx]
    y_full, x_sub, z_sub, y_red, max_diff = compute_full_and_reduced(z, k)

    # izquierda: spline completo seleccionado
    selected_full_line.set_data(x_dense, y_full)
    selected_full_pts.set_data(x, z)

    # izquierda: spline reducido seleccionado
    selected_red_line.set_data(x_dense, y_red)
    selected_red_pts.set_data(x_sub, z_sub)

    # derecha: marcador del dataset actual
    selected_marker.set_data([k], [max_diff])

    info_text.set_text(
        f"dataset = {ds_idx + 1}\n"
        f"N puntos = {k}\n"
        f"max diff = {max_diff:.6f}"
    )

    fig.canvas.draw_idle()

# ============================================================
# Callbacks
# ============================================================
slider_dataset.on_changed(update)
slider_k.on_changed(update)

def on_alpha_submit(_text):
    update()

def on_scale_submit(_text):
    update()

text_alpha.on_submit(on_alpha_submit)
text_scale.on_submit(on_scale_submit)

# inicial
update()

plt.show()