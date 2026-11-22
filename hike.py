"""Plotting helpers for Hike_with_JAX.ipynb.

The terrain f(x, y) lives in the notebook. These functions only draw it:
    plot_map(f, points)      -> topographic (contour) map
    plot_terrain(f, points)  -> rotatable 3D landscape
    plot_map_terrain(f, points) -> both, side by side
`points` can be an (N, 2) array or a dict {name: (x, y)}; `paths` is a list of
(N, 2) arrays (or dicts {name: array}) for routes the hiker walks.
With show=True (default) the figure is drawn and nothing is returned;
use show=False to get the Plotly figure back and modify it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------
def _grid(f, lim, n):
    xs = jnp.linspace(-lim, lim, n)
    ys = jnp.linspace(-lim, lim, n)
    X, Y = jnp.meshgrid(xs, ys)               # rows follow y, columns follow x
    Z = jax.vmap(jax.vmap(f))(X, Y)
    return np.asarray(xs), np.asarray(ys), np.asarray(Z)


def _terrain_colors(zmin, zmax, shore=(-0.15, 0.15)):
    """Water below the shore band (light at the edge, dark when deep), sand around 0, land above.

    `shore` is the height range painted as sand, so flat ground near 0 looks like beach.
    """
    frac = lambda h: float(np.clip((h - zmin) / (zmax - zmin), 0.0, 1.0))
    s_lo, s_hi = frac(shore[0]), frac(shore[1])
    land = [[s_hi, "#d8c99b"],                                  # sand up to the top of the band
            [frac(shore[1] + 0.3), "#4a8c3a"],                  # meadows
            [max(0.65, frac(shore[1] + 0.6)), "#8b6b4a"],       # rock
            [max(0.85, frac(shore[1] + 0.9)), "#a8a29a"],       # scree
            [1.00, "#ffffff"]]                                  # snow
    if s_lo <= 0.0:                                             # no water in range
        return [[0.0, "#d8c99b"]] + land
    return [[0.0, "#08306b"],                                   # deepest water
            [s_lo / 2, "#2171b5"],                              # mid-depth
            [s_lo, "#c6dbef"],                                  # light blue at the water's edge
            [min(s_lo + 1e-3, s_hi), "#d8c99b"]] + land          # sand starts


def _as_named(points):
    """Return (names or None, array of shape (N, 2))."""
    if points is None:
        return None, np.zeros((0, 2))
    if isinstance(points, dict):
        return list(points), np.array(list(points.values()), dtype=float).reshape(-1, 2)
    return None, np.asarray(points, dtype=float).reshape(-1, 2)


def _as_paths(paths):
    if paths is None:
        return []
    if isinstance(paths, dict):
        return [(k, np.asarray(v, dtype=float)) for k, v in paths.items()]
    return [(None, np.asarray(p, dtype=float)) for p in paths]


def _heights(f, P):
    return np.asarray(f(jnp.asarray(P[:, 0]), jnp.asarray(P[:, 1]))) if len(P) else np.zeros(0)


# --------------------------------------------------------------------------
# Trace builders (shared by all plot functions)
# --------------------------------------------------------------------------
def _map_traces(f, xs, ys, Z, zmin, zmax, points, paths, point_color, colorbar=True):
    step, lo = 0.3, -0.15                                    # one contour band is exactly [-0.15, 0.15]
    start = lo - step * np.ceil((lo - zmin) / step)
    traces = [go.Contour(x=xs, y=ys, z=Z, colorscale=_terrain_colors(zmin, zmax),
                         zmin=zmin, zmax=zmax, contours=dict(start=start, end=zmax, size=step),
                         showscale=colorbar, colorbar=dict(title="height", tick0=0, dtick=1))]
    for name, P in _as_paths(paths):
        traces.append(go.Scatter(x=P[:, 0], y=P[:, 1], mode="lines+markers", name=name,
                                 marker=dict(size=4), showlegend=name is not None))
    names, P = _as_named(points)
    if len(P):
        traces.append(go.Scatter(x=P[:, 0], y=P[:, 1],
                                 mode="markers+text" if names else "markers",
                                 text=names, textposition="top right", showlegend=False,
                                 marker=dict(size=9, color=point_color,
                                             line=dict(color="white", width=1))))
    return traces


def _terrain_traces(f, xs, ys, Z, zmin, zmax, points, paths, point_color, lift, colorbar=True):
    traces = [go.Surface(x=xs, y=ys, z=Z, colorscale=_terrain_colors(zmin, zmax),
                         cmin=zmin, cmax=zmax, showscale=colorbar,
                         colorbar=dict(title="height"))]
    for name, P in _as_paths(paths):
        traces.append(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=_heights(f, P) + lift / 3,
                                   mode="lines+markers", name=name,
                                   line=dict(width=5), marker=dict(size=2),
                                   showlegend=name is not None))
    names, P = _as_named(points)
    if len(P):
        traces.append(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=_heights(f, P) + lift,
                                   mode="markers+text" if names else "markers",
                                   text=names, textposition="top center", showlegend=False,
                                   marker=dict(size=4, color=point_color)))
    return traces


def _scene(z_aspect):
    return dict(xaxis_title="x", yaxis_title="y", zaxis_title="height",
                aspectratio=dict(x=1, y=1, z=z_aspect))


# --------------------------------------------------------------------------
# Public
# --------------------------------------------------------------------------
def plot_map(f, points=None, paths=None, lim=3.0, n=200, title="Topographic map",
             point_color="red", show=True):
    """Contour map of f(x, y) with optional marked points and paths."""
    xs, ys, Z = _grid(f, lim, n)
    zmin, zmax = Z.min() - 0.1, Z.max() + 0.1
    fig = go.Figure(_map_traces(f, xs, ys, Z, zmin, zmax, points, paths, point_color))
    fig.update_layout(title=title, width=700, height=650, xaxis_title="x", yaxis_title="y",
                      yaxis_scaleanchor="x")
    if show:
        fig.show()            # draw it; return nothing so Jupyter doesn't draw it twice
        return None
    return fig


def plot_terrain(f, points=None, paths=None, lim=3.0, n=150, title="The landscape",
                 z_aspect=0.6, point_color="red", lift=0.15, show=True):
    """Rotatable 3D surface of f(x, y) with optional marked points and paths."""
    xs, ys, Z = _grid(f, lim, n)
    zmin, zmax = Z.min() - 0.1, Z.max() + 0.1
    fig = go.Figure(_terrain_traces(f, xs, ys, Z, zmin, zmax, points, paths, point_color, lift))
    fig.update_layout(title=title, width=850, height=650, scene=_scene(z_aspect))
    if show:
        fig.show()            # draw it; return nothing so Jupyter doesn't draw it twice
        return None
    return fig


def plot_map_terrain(f, points=None, paths=None, lim=3.0, n=150,
                     titles=("Topographic map", "The landscape"), z_aspect=0.6,
                     point_color="red", lift=0.15, width=None, height=620, show=True):
    """Map (left) and rotatable 3D landscape (right) side by side, one shared colour bar.

    width=None stretches the figure to the notebook width.
    """
    xs, ys, Z = _grid(f, lim, n)
    zmin, zmax = Z.min() - 0.1, Z.max() + 0.1
    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55], horizontal_spacing=0.08,
                        specs=[[{"type": "xy"}, {"type": "scene"}]], subplot_titles=titles)
    for t in _map_traces(f, xs, ys, Z, zmin, zmax, points, paths, point_color, colorbar=True):
        fig.add_trace(t, row=1, col=1)
    for t in _terrain_traces(f, xs, ys, Z, zmin, zmax, points, paths, point_color, lift,
                             colorbar=False):
        fig.add_trace(t, row=1, col=2)
    fig.data[0].colorbar.update(x=0.43, len=0.9, title="height")   # bar between the panels
    fig.update_xaxes(title_text="x", range=[-lim, lim], row=1, col=1)
    fig.update_yaxes(title_text="y", range=[-lim, lim], scaleanchor="x", row=1, col=1)
    fig.update_layout(width=width, height=height, scene=_scene(z_aspect),
                      margin=dict(l=40, r=20, t=60, b=40))
    if show:
        fig.show()            # draw it; return nothing so Jupyter doesn't draw it twice
        return None
    return fig
