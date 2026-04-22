import numpy as np
import plotly.graph_objects as go
from statistics import NormalDist
from sympy import Abs, E, Float, Integer, Rational, Symbol, cos, diff, exp, limit, log, pi, sin, sqrt, tan
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)
from sympy.utilities.lambdify import lambdify

from .mathlatex import normalize_latex_expression
from .registry import ExpressionSpec, ParameterSpec, SimulationSpec, register_spec

_ALLOWED_SYMBOLS = {
    "sin": sin,
    "cos": cos,
    "tan": tan,
    "exp": exp,
    "log": log,
    "sqrt": sqrt,
    "abs": Abs,
    "pi": pi,
    "E": E,
}

_FUNCTION_COLOR = "#e2e8f0"
_FUNCTION_WIDTH = 2.6
_AUX_FILL_COLOR = "#22c55e"
_AUX_FILL_ALPHA = 0.20
_AUX_EDGE_COLOR = "#15803d"
_AUX_EDGE_WIDTH = 1.1
_NODE_COLOR = "#f97316"
_GRID_COLOR = "#334155"
_PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "responsive": True,
}

_DEFAULT_SEED_PARAM = ParameterSpec(
    "seed",
    "Semilla",
    "int",
    42,
    0,
    2_147_483_647,
    "Semilla para reproducibilidad (metodos estocasticos).",
)


class SimulationInputError(ValueError):
    pass


def _build_callable(expression: str, variables: tuple[str, ...]):
    parsed, local_scope = _build_symbolic_expression(expression, variables)

    allowed_free_symbols = {local_scope[name] for name in variables}
    if not parsed.free_symbols.issubset(allowed_free_symbols):
        raise SimulationInputError(
            f"La expresion '{expression}' contiene simbolos no permitidos."
        )

    ordered_symbols = tuple(local_scope[name] for name in variables)
    return lambdify(ordered_symbols, parsed, modules=["numpy"])


def _build_symbolic_expression(expression: str, variables: tuple[str, ...]):
    local_scope = {name: Symbol(name) for name in variables}
    local_scope.update(_ALLOWED_SYMBOLS)

    normalized_expression = normalize_latex_expression(expression)

    try:
        parsed = parse_expr(
            normalized_expression,
            local_dict=local_scope,
            global_dict={
                "__builtins__": {},
                "Integer": Integer,
                "Rational": Rational,
                "Float": Float,
            },
            transformations=standard_transformations
            + (convert_xor, implicit_multiplication_application),
            evaluate=True,
        )
    except Exception as exc:
        raise SimulationInputError(f"No se pudo interpretar la expresion '{expression}'.") from exc

    return parsed, local_scope


def _safe_limit_value(parsed, symbol, x_value: float) -> float:
    try:
        limit_expr = limit(parsed, symbol, float(x_value))
        limit_value = float(limit_expr.evalf())
    except Exception:
        return np.nan
    return limit_value if np.isfinite(limit_value) else np.nan


def _safe_evaluate_univariate(expression: str, x_values):
    parsed, local_scope = _build_symbolic_expression(expression, ("x",))
    symbol = local_scope["x"]
    callable_fn = lambdify(symbol, parsed, modules=["numpy"])

    x_array = np.asarray(x_values, dtype=float)
    scalar_input = x_array.shape == ()
    if scalar_input:
        x_array = x_array.reshape(1)

    try:
        with np.errstate(all="ignore"):
            raw_values = callable_fn(x_array)
    except Exception:
        with np.errstate(all="ignore"):
            raw_values = np.array([callable_fn(float(value)) for value in x_array.flat], dtype=float).reshape(x_array.shape)

    values = np.asarray(raw_values, dtype=float)
    if values.shape != x_array.shape:
        values = np.broadcast_to(values, x_array.shape).astype(float).copy()

    mask = ~np.isfinite(values)
    if np.any(mask):
        values = values.copy()
        for index in np.argwhere(mask):
            index_tuple = tuple(index)
            values[index_tuple] = _safe_limit_value(parsed, symbol, float(x_array[index_tuple]))

    if scalar_input:
        return float(values[0])
    return values


def _ensure_positive_int(value: int, field_name: str) -> int:
    value = int(value)
    if value <= 0:
        raise SimulationInputError(f"{field_name} debe ser mayor a 0.")
    return value


def _ensure_interval(a: float, b: float) -> tuple[float, float]:
    a = float(a)
    b = float(b)
    if b <= a:
        raise SimulationInputError("Se requiere b > a.")
    return a, b


def _ensure_xi_in_interval(xi: float, a: float, b: float) -> float:
    xi = float(xi)
    if not (a <= xi <= b):
        raise SimulationInputError("xi debe estar dentro del intervalo [a, b].")
    return xi


def _ensure_positive_step(value: float, field_name: str = "h") -> float:
    step = float(value)
    if step <= 0:
        raise SimulationInputError(f"{field_name} debe ser mayor a 0.")
    return step


def _ensure_confidence_level(value: float) -> float:
    level = float(value)
    if level <= 0.0 or level >= 1.0:
        raise SimulationInputError("confidence_level debe estar entre 0 y 1 (exclusivo).")
    return level


def _monte_carlo_stats(values: np.ndarray, confidence_level: float, scale: float) -> dict:
    if values.size == 0:
        raise SimulationInputError("No hay muestras para calcular estadisticos de Monte Carlo.")

    if not np.all(np.isfinite(values)):
        raise SimulationInputError("Se detectaron valores no finitos en las muestras de Monte Carlo.")

    n = int(values.size)
    mean = float(np.mean(values))
    variance = float(np.var(values, ddof=1)) if n > 1 else 0.0
    std_dev = float(np.sqrt(max(variance, 0.0)))
    std_error = float(std_dev / np.sqrt(n))
    z_score = float(NormalDist().inv_cdf(0.5 * (1.0 + confidence_level)))
    ci_half_width = z_score * std_error

    integral_estimate = float(scale * mean)
    integral_std_error = float(abs(scale) * std_error)
    integral_ci_low = float(integral_estimate - z_score * integral_std_error)
    integral_ci_high = float(integral_estimate + z_score * integral_std_error)

    return {
        "n": n,
        "mean": mean,
        "variance": variance,
        "std_dev": std_dev,
        "std_error": std_error,
        "z_score": z_score,
        "integral_estimate": integral_estimate,
        "integral_ci_low": integral_ci_low,
        "integral_ci_high": integral_ci_high,
    }


def _normalize_precision(cleaned_data: dict) -> int:
    raw_value = cleaned_data.get("precision", 6)
    try:
        precision = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise SimulationInputError("precision debe ser un entero entre 0 y 12.") from exc

    if precision < 0 or precision > 12:
        raise SimulationInputError("precision debe estar entre 0 y 12.")
    return precision


def _round_numeric(value, precision: int):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric):
            return round(numeric, precision)
        return numeric
    return value


def _apply_precision(value, precision: int):
    if isinstance(value, dict):
        return {key: _apply_precision(item, precision) for key, item in value.items()}
    if isinstance(value, list):
        return [_apply_precision(item, precision) for item in value]
    if isinstance(value, tuple):
        return tuple(_apply_precision(item, precision) for item in value)
    return _round_numeric(value, precision)


def _parse_points_csv(value: str, field_name: str) -> np.ndarray:
    try:
        parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SimulationInputError(f"{field_name} debe ser una lista CSV de numeros.") from exc

    if len(parsed) < 2:
        raise SimulationInputError(f"{field_name} requiere al menos dos valores.")
    return np.array(parsed, dtype=float)


def _to_plotly_values(values) -> list:
    if isinstance(values, np.ndarray):
        return values.tolist()
    return list(values)


def _figure_to_html(fig: go.Figure) -> str:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111a2e",
        plot_bgcolor="#111a2e",
        font={"color": "#e5e7eb"},
        margin={"l": 55, "r": 20, "t": 55, "b": 50},
        dragmode="pan",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config=_PLOTLY_CONFIG)


def _line_plot(series: list[dict], title: str, x_label: str, y_label: str) -> str:
    fig = go.Figure()
    for item in series:
        if item.get("style") == "scatter":
            fig.add_trace(
                go.Scatter(
                    x=_to_plotly_values(item["x"]),
                    y=_to_plotly_values(item["y"]),
                    mode="markers",
                    name=item["name"],
                    marker={"size": 6},
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=_to_plotly_values(item["x"]),
                    y=_to_plotly_values(item["y"]),
                    mode="lines",
                    name=item["name"],
                    line={"width": 2.5},
                )
            )

    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    fig.update_xaxes(showgrid=True, gridcolor=_GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID_COLOR)
    return _figure_to_html(fig)


def _aux_plot_distribution_with_standard_normal(values: np.ndarray, title: str, x_label: str) -> str:
    sample_values = np.asarray(values, dtype=float)
    finite_values = sample_values[np.isfinite(sample_values)]
    if finite_values.size == 0:
        raise SimulationInputError("No hay valores finitos para construir la distribucion.")

    mean = float(np.mean(finite_values))
    std_dev = float(np.std(finite_values, ddof=1)) if finite_values.size > 1 else 0.0
    if std_dev > 0.0:
        z_values = (finite_values - mean) / std_dev
    else:
        z_values = np.zeros_like(finite_values, dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=_to_plotly_values(z_values),
            histnorm="probability density",
            nbinsx=40,
            name="histograma estandarizado",
            marker={"color": "rgba(56, 189, 248, 0.55)", "line": {"color": "#0ea5e9", "width": 1}},
            opacity=0.85,
        )
    )

    normal_x = np.linspace(-4.0, 4.0, 400)
    normal_pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (normal_x ** 2))
    fig.add_trace(
        go.Scatter(
            x=_to_plotly_values(normal_x),
            y=_to_plotly_values(normal_pdf),
            mode="lines",
            name="normal estandar",
            line={"color": "#f97316", "width": 2.6},
        )
    )

    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title="densidad")
    fig.update_xaxes(showgrid=True, gridcolor=_GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID_COLOR)
    return _figure_to_html(fig)


def _aux_plot_trapezoids(f, a: float, b: float, x_nodes: np.ndarray, title: str, expression: str | None = None) -> str:
    dense_x = np.linspace(a, b, 600)
    dense_y = _safe_evaluate_univariate(expression, dense_x) if expression else np.asarray(f(dense_x), dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_to_plotly_values(dense_x),
            y=_to_plotly_values(dense_y),
            mode="lines",
            name="f(x)",
            line={"color": _FUNCTION_COLOR, "width": _FUNCTION_WIDTH},
        )
    )

    y_nodes = _safe_evaluate_univariate(expression, x_nodes) if expression else np.asarray(f(x_nodes), dtype=float)
    for idx, (left, right, y_left, y_right) in enumerate(zip(x_nodes[:-1], x_nodes[1:], y_nodes[:-1], y_nodes[1:])):
        fig.add_trace(
            go.Scatter(
                x=[left, left, right, right, left],
                y=[0.0, y_left, y_right, 0.0, 0.0],
                fill="toself",
                mode="lines",
                line={"color": _AUX_EDGE_COLOR, "width": _AUX_EDGE_WIDTH},
                fillcolor="rgba(34, 197, 94, 0.20)",
                name="trapecios" if idx == 0 else None,
                showlegend=idx == 0,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=_to_plotly_values(x_nodes),
            y=_to_plotly_values(y_nodes),
            mode="markers",
            name="nodos",
            marker={"size": 7, "color": _NODE_COLOR},
        )
    )
    fig.update_layout(title=title, xaxis_title="x", yaxis_title="f(x)")
    fig.update_xaxes(showgrid=True, gridcolor=_GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID_COLOR)
    return _figure_to_html(fig)


def _aux_plot_rectangles_midpoint(f, a: float, b: float, n: int, title: str, expression: str | None = None) -> str:
    h = (b - a) / n
    midpoints = a + (np.arange(n) + 0.5) * h
    heights = _safe_evaluate_univariate(expression, midpoints) if expression else np.asarray(f(midpoints), dtype=float)

    dense_x = np.linspace(a, b, 600)
    dense_y = _safe_evaluate_univariate(expression, dense_x) if expression else np.asarray(f(dense_x), dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_to_plotly_values(dense_x),
            y=_to_plotly_values(dense_y),
            mode="lines",
            name="f(x)",
            line={"color": _FUNCTION_COLOR, "width": _FUNCTION_WIDTH},
        )
    )
    fig.add_trace(
        go.Bar(
            x=_to_plotly_values(midpoints),
            y=_to_plotly_values(heights),
            name="subrectangulos",
            marker={"color": "rgba(34, 197, 94, 0.20)", "line": {"color": _AUX_EDGE_COLOR, "width": _AUX_EDGE_WIDTH}},
            width=h,
            opacity=1.0,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_to_plotly_values(midpoints),
            y=_to_plotly_values(heights),
            mode="markers",
            name="puntos medios",
            marker={"size": 7, "color": _NODE_COLOR},
        )
    )
    fig.update_layout(title=title, xaxis_title="x", yaxis_title="f(x)", barmode="overlay")
    fig.update_xaxes(showgrid=True, gridcolor=_GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID_COLOR)
    return _figure_to_html(fig)


def _aux_plot_simpson_panels(
    f,
    a: float,
    b: float,
    x_nodes: np.ndarray,
    panel_size: int,
    title: str,
    expression: str | None = None,
) -> tuple[str, int]:
    dense_x = np.linspace(a, b, 700)
    dense_y = _safe_evaluate_univariate(expression, dense_x) if expression else np.asarray(f(dense_x), dtype=float)
    y_nodes = _safe_evaluate_univariate(expression, x_nodes) if expression else np.asarray(f(x_nodes), dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_to_plotly_values(dense_x),
            y=_to_plotly_values(dense_y),
            mode="lines",
            name="f(x)",
            line={"color": _FUNCTION_COLOR, "width": _FUNCTION_WIDTH},
        )
    )

    total_panels = (len(x_nodes) - 1) // panel_size
    max_panels = min(total_panels, 24)
    palette = ["#16a34a", "#0ea5e9", "#f59e0b", "#a855f7"]

    for panel_idx in range(max_panels):
        start = panel_idx * panel_size
        end = start + panel_size + 1
        panel_x = x_nodes[start:end]
        panel_y = y_nodes[start:end]
        coeffs = np.polyfit(panel_x, panel_y, deg=panel_size)
        poly = np.poly1d(coeffs)
        panel_dense_x = np.linspace(panel_x[0], panel_x[-1], 120)
        panel_dense_y = poly(panel_dense_x)
        color = palette[panel_idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=_to_plotly_values(panel_dense_x),
                y=_to_plotly_values(panel_dense_y),
                fill="tozeroy",
                mode="lines",
                line={"color": color, "width": 1.7},
                opacity=0.25,
                name="paneles" if panel_idx == 0 else None,
                showlegend=panel_idx == 0,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=_to_plotly_values(x_nodes),
            y=_to_plotly_values(y_nodes),
            mode="markers",
            name="nodos",
            marker={"size": 7, "color": _NODE_COLOR},
        )
    )
    fig.update_layout(title=title, xaxis_title="x", yaxis_title="f(x)")
    fig.update_xaxes(showgrid=True, gridcolor=_GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID_COLOR)
    return _figure_to_html(fig), total_panels


def _result_payload(
    title: str,
    description: str,
    summary: dict,
    plot_series: list[dict],
    plot_title: str,
    x_label: str,
    y_label: str,
    table_headers: list[str],
    table_rows: list[list],
    auxiliary_plots: list[dict] | None = None,
    plot_image: str | None = None,
) -> dict:
    return {
        "title": title,
        "description": description,
        "summary": summary,
        "plot": plot_image or _line_plot(plot_series, plot_title, x_label, y_label),
        "auxiliary_plots": auxiliary_plots or [],
        "table": {
            "headers": table_headers,
            "rows": table_rows[:25],
        },
    }


def _suggest_plot_interval(points, margin_ratio: float = 0.2, min_span: float = 1.0) -> tuple[float, float]:
    values = np.asarray(points, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return -1.0, 1.0

    x_min = float(np.min(finite_values))
    x_max = float(np.max(finite_values))
    span = max(x_max - x_min, min_span)
    margin = span * margin_ratio
    return x_min - margin, x_max + margin


def _aux_plot_original_function(
    f,
    a: float,
    b: float,
    title: str,
    function_label: str = "f(x)",
    x_points=None,
    xi: float | None = None,
    expression: str | None = None,
) -> str:
    a, b = _ensure_interval(a, b)
    x_grid = np.linspace(a, b, 450)
    y_grid = _safe_evaluate_univariate(expression, x_grid) if expression else np.asarray(f(x_grid), dtype=float)
    y_grid = np.where(np.isfinite(y_grid), y_grid, np.nan)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_to_plotly_values(x_grid),
            y=_to_plotly_values(y_grid),
            mode="lines",
            name=function_label,
            line={"color": _FUNCTION_COLOR, "width": _FUNCTION_WIDTH},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[a, b],
            y=[0.0, 0.0],
            mode="lines",
            name="y = 0",
            line={"color": "#64748b", "width": 1.2, "dash": "dash"},
        )
    )

    if x_points is not None:
        x_arr = np.asarray(x_points, dtype=float)
        if x_arr.size > 0:
            y_arr = _safe_evaluate_univariate(expression, x_arr) if expression else np.asarray(f(x_arr), dtype=float)
            mask = np.isfinite(x_arr) & np.isfinite(y_arr)
            if np.any(mask):
                fig.add_trace(
                    go.Scatter(
                        x=_to_plotly_values(x_arr[mask]),
                        y=_to_plotly_values(y_arr[mask]),
                        mode="markers",
                        name="iteraciones",
                        marker={"size": 7, "color": _NODE_COLOR},
                    )
                )

    if xi is not None and np.isfinite(float(xi)):
        xi_value = float(xi)
        yi_value = float(f(xi_value))
        if np.isfinite(yi_value):
            fig.add_trace(
                go.Scatter(
                    x=[xi_value],
                    y=[yi_value],
                    mode="markers",
                    name="xi",
                    marker={"size": 10, "color": "#f43f5e"},
                )
            )
            fig.add_shape(
                type="line",
                x0=xi_value,
                x1=xi_value,
                y0=0.0,
                y1=yi_value,
                line={"color": "#f43f5e", "width": 1.2, "dash": "dot"},
            )

    fig.update_layout(title=title, xaxis_title="x", yaxis_title="y")
    fig.update_xaxes(showgrid=True, gridcolor=_GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID_COLOR)
    return _figure_to_html(fig)


def _truncation_error_from_xi(
    expression: str,
    xi: float,
    derivative_order: int,
    coefficient: float,
    variables: tuple[str, ...] = ("x",),
) -> tuple[float, float]:
    if derivative_order < 1:
        raise SimulationInputError("El orden de la derivada debe ser mayor a 0.")

    parsed, local_scope = _build_symbolic_expression(expression, variables)
    symbol = local_scope[variables[0]]
    derivative_expr = parsed
    for _ in range(derivative_order):
        derivative_expr = diff(derivative_expr, symbol)

    xi_value = float(xi)
    derivative_value = float(derivative_expr.subs(symbol, xi_value))
    if not np.isfinite(derivative_value):
        raise SimulationInputError("No se pudo evaluar la derivada en xi.")

    return coefficient * derivative_value, derivative_value


def _integration_truncation_report(
    expression: str,
    xi: float | None,
    a: float,
    b: float,
    derivative_order: int,
    coefficient: float,
    derivative_label: str,
) -> dict:
    xi_value = 0.5 * (a + b) if xi is None else _ensure_xi_in_interval(xi, a, b)
    truncation_error, derivative_value = _truncation_error_from_xi(
        expression,
        xi_value,
        derivative_order,
        coefficient,
    )
    return {
        "xi": xi_value,
        derivative_label: derivative_value,
        "error de truncamiento": truncation_error,
    }

def _aux_plot_fixed_point(g, a: float, b: float, iterations: list, title: str) -> str:
    """Grafica g(x), y=x y las iteraciones como pasos."""
    x_grid = np.linspace(a, b, 400)
    y_g = np.asarray(g(x_grid), dtype=float)
    y_line = x_grid  # y = x
    
    fig = go.Figure()
    
    # Gráfica de g(x)
    fig.add_trace(
        go.Scatter(
            x=_to_plotly_values(x_grid),
            y=_to_plotly_values(y_g),
            mode="lines",
            name="g(x)",
            line={"color": _FUNCTION_COLOR, "width": _FUNCTION_WIDTH},
        )
    )
    
    # Línea y = x
    fig.add_trace(
        go.Scatter(
            x=_to_plotly_values(x_grid),
            y=_to_plotly_values(y_line),
            mode="lines",
            name="y = x",
            line={"color": "#64748b", "width": 1.5, "dash": "dash"},
        )
    )
    
    # Dibujar iteraciones como pasos (escalera)
    if len(iterations) > 0:
        x_current = float(iterations[0][1])  # x0 (x_n inicial)
        
        step_x = [x_current]
        step_y = [0.0]
        
        max_iterations_plot = min(len(iterations), 20)  # Limitar a 20 pasos visuales
        
        for idx in range(max_iterations_plot):
            x_next = float(iterations[idx][2])  # x_n+1
            
            # Línea vertical de (x_n, x_n) a (x_n, g(x_n))
            step_x.extend([x_current, x_current])
            step_y.extend([x_current, x_next])
            
            # Línea horizontal de (x_n, g(x_n)) a (g(x_n), g(x_n))
            step_x.extend([x_current, x_next])
            step_y.extend([x_next, x_next])
            
            x_current = x_next
        
        fig.add_trace(
            go.Scatter(
                x=_to_plotly_values(step_x),
                y=_to_plotly_values(step_y),
                mode="lines",
                name="iteraciones",
                line={"color": _NODE_COLOR, "width": 1.8},
            )
        )
        
        # Marcar punto inicial
        fig.add_trace(
            go.Scatter(
                x=[float(iterations[0][1])],
                y=[0.0],
                mode="markers",
                name="x0",
                marker={"size": 8, "color": "#22c55e"},
            )
        )
    
    fig.update_layout(title=title, xaxis_title="x", yaxis_title="y")
    fig.update_xaxes(showgrid=True, gridcolor=_GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID_COLOR)
    return _figure_to_html(fig)


def _run_fixed_point(data: dict) -> dict:
    g = _build_callable(data["gx"], variables=("x",))
    tol = float(data["tol"])
    max_iter = _ensure_positive_int(data["max_iter"], "max_iter")
    x_current = float(data["x0"])
    
    # Nuevo: intervalo para visualización
    a = float(data.get("a", x_current - 1.0))
    b = float(data.get("b", x_current + 1.0))
    a, b = _ensure_interval(a, b)

    iterations = []
    for idx in range(1, max_iter + 1):
        x_next = float(g(x_current))
        error = abs(x_next - x_current)
        iterations.append([idx, x_current, x_next, error])
        if error < tol:
            break
        x_current = x_next

    if not iterations:
        raise SimulationInputError("No se pudieron generar iteraciones.")

    it = np.array([row[0] for row in iterations], dtype=float)
    xn = np.array([row[2] for row in iterations], dtype=float)
    
    # Gráfico de convergencia (iteración vs x_n)
    convergence_plot = _line_plot(
        [{"name": "x_n", "x": it, "y": xn, "style": "line"}],
        "Convergencia de punto fijo",
        "iteracion",
        "x_n"
    )
    
    # Gráfico de g(x), y=x e iteraciones
    phase_plot = _aux_plot_fixed_point(g, a, b, iterations, "Metodo de punto fijo: g(x) vs y=x")

    return _result_payload(
        title="Resultado: Metodo de punto fijo",
        description=f"Iteracion x(n+1) = g(x(n)), g(x) = {data['gx']}",
        summary={
            "iteraciones": len(iterations),
            "aproximacion": float(xn[-1]),
            "error final": float(iterations[-1][3]),
            "intervalo": f"[{a}, {b}]",
        },
        plot_series=[{"name": "x_n", "x": it, "y": xn, "style": "line"}],
        plot_title="Convergencia de punto fijo",
        x_label="iteracion",
        y_label="x_n",
        table_headers=["n", "x_n", "x_n+1", "error"],
        table_rows=iterations,
        auxiliary_plots=[
            {"title": "Fase del punto fijo", "plot": phase_plot},
            {"title": "Convergencia", "plot": convergence_plot},
        ],
    )


def _run_bisection(data: dict) -> dict:
    f = _build_callable(data["fx"], variables=("x",))
    a, b = _ensure_interval(data["a"], data["b"])
    plot_a, plot_b = a, b
    tol = float(data["tol"])
    max_iter = _ensure_positive_int(data["max_iter"], "max_iter")

    fa = float(f(a))
    fb = float(f(b))
    if fa * fb > 0:
        raise SimulationInputError("f(a) y f(b) deben tener signos opuestos.")

    rows = []
    midpoints = []
    fmid_values = []

    for idx in range(1, max_iter + 1):
        c = 0.5 * (a + b)
        fc = float(f(c))
        rows.append([idx, a, b, c, fc, abs(b - a)])
        midpoints.append(c)
        fmid_values.append(fc)

        if abs(fc) < tol or abs(b - a) < tol:
            break

        if fa * fc < 0:
            b = c
        else:
            a = c
            fa = fc

    function_plot = _aux_plot_original_function(
        f,
        plot_a,
        plot_b,
        "Funcion original en el intervalo [a, b]",
        function_label="f(x)",
        x_points=midpoints,
        expression=data["fx"],
    )

    return _result_payload(
        title="Resultado: Metodo de biseccion",
        description=f"Raiz de f(x) = {data['fx']}",
        summary={
            "iteraciones": len(rows),
            "aproximacion": float(midpoints[-1]),
            "|f(c)| final": float(abs(fmid_values[-1])),
        },
        plot_series=[
            {
                "name": "f(c_n)",
                "x": np.arange(1, len(fmid_values) + 1),
                "y": np.abs(np.array(fmid_values)),
                "style": "line",
            }
        ],
        plot_title="Error por iteracion en biseccion",
        x_label="iteracion",
        y_label="|f(c_n)|",
        table_headers=["n", "a", "b", "c", "f(c)", "|b-a|"],
        table_rows=rows,
        auxiliary_plots=[{"title": "Funcion original f(x)", "plot": function_plot}],
    )


def _run_newton_raphson(data: dict) -> dict:
    f = _build_callable(data["fx"], variables=("x",))
    df = _build_callable(data["dfx"], variables=("x",))
    tol = float(data["tol"])
    max_iter = _ensure_positive_int(data["max_iter"], "max_iter")

    x_current = float(data["x0"])
    rows = []

    for idx in range(1, max_iter + 1):
        fx_value = float(f(x_current))
        dfx_value = float(df(x_current))
        if abs(dfx_value) < 1e-14:
            raise SimulationInputError("La derivada se anulo durante la iteracion.")

        x_next = x_current - fx_value / dfx_value
        error = abs(x_next - x_current)
        rows.append([idx, x_current, fx_value, dfx_value, x_next, error])

        if error < tol:
            break
        x_current = x_next

    x_values = np.array([row[4] for row in rows], dtype=float)

    x_for_plot = [rows[0][1]] + [row[4] for row in rows]
    interval_a, interval_b = _suggest_plot_interval(x_for_plot)
    function_plot = _aux_plot_original_function(
        f,
        interval_a,
        interval_b,
        "Funcion original f(x) y aproximaciones",
        function_label="f(x)",
        x_points=x_for_plot,
        expression=data["fx"],
    )

    return _result_payload(
        title="Resultado: Metodo de Newton-Raphson",
        description=f"Raiz de f(x) = {data['fx']}",
        summary={
            "iteraciones": len(rows),
            "aproximacion": float(x_values[-1]),
            "error final": float(rows[-1][5]),
        },
        plot_series=[
            {
                "name": "x_n",
                "x": np.arange(1, len(rows) + 1),
                "y": x_values,
                "style": "line",
            }
        ],
        plot_title="Convergencia de Newton-Raphson",
        x_label="iteracion",
        y_label="x_n",
        table_headers=["n", "x_n", "f(x_n)", "f'(x_n)", "x_n+1", "error"],
        table_rows=rows,
        auxiliary_plots=[{"title": "Funcion original f(x)", "plot": function_plot}],
    )


def _run_aitken_delta2(data: dict) -> dict:
    g = _build_callable(data["gx"], variables=("x",))
    tol = float(data["tol"])
    max_iter = _ensure_positive_int(data["max_iter"], "max_iter")

    x0 = float(data["x0"])
    rows = []

    for idx in range(1, max_iter + 1):
        x1 = float(g(x0))
        x2 = float(g(x1))
        denom = x2 - 2.0 * x1 + x0
        if abs(denom) < 1e-14:
            raise SimulationInputError("No se puede aplicar Aitken (denominador cercano a cero).")

        x_hat = x0 - ((x1 - x0) ** 2) / denom
        error = abs(x_hat - x0)
        rows.append([idx, x0, x1, x2, x_hat, error])

        if error < tol:
            break
        x0 = x_hat

    aitken_values = np.array([row[4] for row in rows], dtype=float)

    x_candidates = [row[idx] for row in rows for idx in (1, 2, 3, 4)]
    interval_a, interval_b = _suggest_plot_interval(x_candidates)
    iterations_for_phase = [
        [int(row[0]), float(row[1]), float(row[2]), abs(float(row[2]) - float(row[1]))]
        for row in rows
    ]
    function_plot = _aux_plot_fixed_point(
        g,
        interval_a,
        interval_b,
        iterations_for_phase,
        "Funcion original g(x) y y = x",
    )

    return _result_payload(
        title="Resultado: Metodo delta cuadrado de Aitken",
        description=f"Aceleracion para la iteracion de g(x) = {data['gx']}",
        summary={
            "iteraciones": len(rows),
            "aproximacion": float(aitken_values[-1]),
            "error final": float(rows[-1][5]),
        },
        plot_series=[
            {
                "name": "x_hat_n",
                "x": np.arange(1, len(rows) + 1),
                "y": aitken_values,
                "style": "line",
            }
        ],
        plot_title="Convergencia de Aitken",
        x_label="iteracion",
        y_label="x_hat_n",
        table_headers=["n", "x0", "x1", "x2", "x_hat", "error"],
        table_rows=rows,
        auxiliary_plots=[{"title": "Funcion original g(x)", "plot": function_plot}],
    )


def _lagrange_evaluate(x_nodes: np.ndarray, y_nodes: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    y_eval = np.zeros_like(x_eval, dtype=float)
    n = len(x_nodes)
    for i in range(n):
        li = np.ones_like(x_eval, dtype=float)
        for j in range(n):
            if i == j:
                continue
            li *= (x_eval - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        y_eval += y_nodes[i] * li
    return y_eval


def _run_lagrange_interpolation(data: dict) -> dict:
    x_nodes = _parse_points_csv(data["x_points"], "x_points")
    y_nodes = _parse_points_csv(data["y_points"], "y_points")
    if len(x_nodes) != len(y_nodes):
        raise SimulationInputError("x_points e y_points deben tener el mismo largo.")

    x_eval = float(data["x_eval"])
    y_eval = float(_lagrange_evaluate(x_nodes, y_nodes, np.array([x_eval]))[0])

    x_min, x_max = float(np.min(x_nodes)), float(np.max(x_nodes))
    grid = np.linspace(x_min, x_max, 400)
    y_grid = _lagrange_evaluate(x_nodes, y_nodes, grid)

    return _result_payload(
        title="Resultado: Interpolacion de Lagrange",
        description="Polinomio interpolante construido con nodos dados.",
        summary={
            "cantidad de nodos": int(len(x_nodes)),
            "x evaluado": x_eval,
            "P(x_eval)": y_eval,
        },
        plot_series=[
            {"name": "P(x)", "x": grid, "y": y_grid, "style": "line"},
            {"name": "nodos", "x": x_nodes, "y": y_nodes, "style": "scatter"},
            {
                "name": "x_eval",
                "x": np.array([x_eval]),
                "y": np.array([y_eval]),
                "style": "scatter",
            },
        ],
        plot_title="Interpolacion de Lagrange",
        x_label="x",
        y_label="y",
        table_headers=["x_i", "y_i"],
        table_rows=[[float(x_nodes[i]), float(y_nodes[i])] for i in range(len(x_nodes))],
    )


def _newton_divided_coeffs(x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
    n = len(x_nodes)
    coeffs = y_nodes.astype(float).copy()
    for j in range(1, n):
        coeffs[j:n] = (coeffs[j:n] - coeffs[j - 1 : n - 1]) / (x_nodes[j:n] - x_nodes[0 : n - j])
    return coeffs


def _newton_poly_eval(x_nodes: np.ndarray, coeffs: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    values = np.full_like(x_eval, coeffs[-1], dtype=float)
    for idx in range(len(coeffs) - 2, -1, -1):
        values = values * (x_eval - x_nodes[idx]) + coeffs[idx]
    return values


def _run_newton_divided_interpolation(data: dict) -> dict:
    x_nodes = _parse_points_csv(data["x_points"], "x_points")
    y_nodes = _parse_points_csv(data["y_points"], "y_points")
    if len(x_nodes) != len(y_nodes):
        raise SimulationInputError("x_points e y_points deben tener el mismo largo.")

    coeffs = _newton_divided_coeffs(x_nodes, y_nodes)
    x_eval = float(data["x_eval"])
    y_eval = float(_newton_poly_eval(x_nodes, coeffs, np.array([x_eval]))[0])

    grid = np.linspace(float(np.min(x_nodes)), float(np.max(x_nodes)), 400)
    y_grid = _newton_poly_eval(x_nodes, coeffs, grid)

    return _result_payload(
        title="Resultado: Diferencias divididas de Newton",
        description="Polinomio interpolante en forma de Newton.",
        summary={
            "cantidad de nodos": int(len(x_nodes)),
            "x evaluado": x_eval,
            "P(x_eval)": y_eval,
        },
        plot_series=[
            {"name": "P(x)", "x": grid, "y": y_grid, "style": "line"},
            {"name": "nodos", "x": x_nodes, "y": y_nodes, "style": "scatter"},
        ],
        plot_title="Interpolacion de Newton",
        x_label="x",
        y_label="y",
        table_headers=["coeficiente", "valor"],
        table_rows=[[f"a_{i}", float(coeffs[i])] for i in range(len(coeffs))],
    )


def _run_trapezoid_simple(data: dict) -> dict:
    f = _build_callable(data["fx"], variables=("x",))
    a, b = _ensure_interval(data["a"], data["b"])
    fa = float(_safe_evaluate_univariate(data["fx"], a))
    fb = float(_safe_evaluate_univariate(data["fx"], b))
    integral = (b - a) * (fa + fb) / 2.0
    xi_report = _integration_truncation_report(
        data["fx"],
        data.get("xi", 0.5 * (a + b)),
        a,
        b,
        2,
        -((b - a) ** 3) / 12.0,
        "f''(xi)",
    )

    grid = np.linspace(a, b, 250)
    y_grid = _safe_evaluate_univariate(data["fx"], grid)
    combined_plot = _aux_plot_trapezoids(f, a, b, np.array([a, b]), "Trapecio simple", expression=data["fx"])

    return _result_payload(
        title="Resultado: Regla del trapecio simple",
        description=f"Integral aproximada de f(x) = {data['fx']}",
        summary={"integral aproximada": float(integral), "intervalo": f"[{a}, {b}]", **xi_report},
        plot_series=[
            {"name": "f(x)", "x": grid, "y": y_grid, "style": "line"},
            {"name": "extremos", "x": np.array([a, b]), "y": np.array([fa, fb]), "style": "scatter"},
        ],
        plot_title="Trapecio simple",
        x_label="x",
        y_label="f(x)",
        table_headers=["x", "f(x)"],
        table_rows=[[a, fa], [b, fb]],
        plot_image=combined_plot,
    )


def _run_trapezoid_composite(data: dict) -> dict:
    f = _build_callable(data["fx"], variables=("x",))
    a, b = _ensure_interval(data["a"], data["b"])
    n = _ensure_positive_int(data["n"], "n")
    h = (b - a) / n
    xi_report = _integration_truncation_report(
        data["fx"],
        data.get("xi", 0.5 * (a + b)),
        a,
        b,
        2,
        -((b - a) * (h ** 2)) / 12.0,
        "f''(xi)",
    )

    x = np.linspace(a, b, n + 1)
    y = _safe_evaluate_univariate(data["fx"], x)
    integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    combined_plot = _aux_plot_trapezoids(f, a, b, x, "Trapecio compuesto", expression=data["fx"])

    return _result_payload(
        title="Resultado: Regla del trapecio compuesta",
        description=f"Integral aproximada de f(x) = {data['fx']}",
        summary={"integral aproximada": float(integral), "subintervalos": int(n), **xi_report},
        plot_series=[
            {"name": "f(x)", "x": x, "y": y, "style": "line"},
            {"name": "nodos", "x": x, "y": y, "style": "scatter"},
        ],
        plot_title="Trapecio compuesto",
        x_label="x",
        y_label="f(x)",
        table_headers=["i", "x_i", "f(x_i)"],
        table_rows=[[i, float(x[i]), float(y[i])] for i in range(len(x))],
        plot_image=combined_plot,
    )


def _run_simpson_13_simple(data: dict) -> dict:
    f = _build_callable(data["fx"], variables=("x",))
    a, b = _ensure_interval(data["a"], data["b"])
    m = 0.5 * (a + b)

    fa = float(_safe_evaluate_univariate(data["fx"], a))
    fm = float(_safe_evaluate_univariate(data["fx"], m))
    fb = float(_safe_evaluate_univariate(data["fx"], b))
    integral = (b - a) * (fa + 4 * fm + fb) / 6.0
    xi_report = _integration_truncation_report(
        data["fx"],
        data.get("xi", 0.5 * (a + b)),
        a,
        b,
        4,
        -((b - a) ** 5) / 2880.0,
        "f''''(xi)",
    )

    grid = np.linspace(a, b, 250)
    y_grid = _safe_evaluate_univariate(data["fx"], grid)
    combined_plot, _ = _aux_plot_simpson_panels(
        f,
        a,
        b,
        np.array([a, m, b]),
        panel_size=2,
        title="Simpson 1/3 simple",
        expression=data["fx"],
    )

    return _result_payload(
        title="Resultado: Simpson 1/3 simple",
        description=f"Integral aproximada de f(x) = {data['fx']}",
        summary={"integral aproximada": float(integral), "intervalo": f"[{a}, {b}]", **xi_report},
        plot_series=[
            {"name": "f(x)", "x": grid, "y": y_grid, "style": "line"},
            {"name": "nodos", "x": np.array([a, m, b]), "y": np.array([fa, fm, fb]), "style": "scatter"},
        ],
        plot_title="Simpson 1/3 simple",
        x_label="x",
        y_label="f(x)",
        table_headers=["x", "f(x)"],
        table_rows=[[a, fa], [m, fm], [b, fb]],
        plot_image=combined_plot,
    )


def _run_simpson_13_composite(data: dict) -> dict:
    f = _build_callable(data["fx"], variables=("x",))
    a, b = _ensure_interval(data["a"], data["b"])
    n = _ensure_positive_int(data["n"], "n")
    if n % 2 != 0:
        raise SimulationInputError("Para Simpson 1/3 compuesta, n debe ser par.")

    h = (b - a) / n
    xi_report = _integration_truncation_report(
        data["fx"],
        data.get("xi", 0.5 * (a + b)),
        a,
        b,
        4,
        -((b - a) * (h ** 4)) / 180.0,
        "f''''(xi)",
    )

    x = np.linspace(a, b, n + 1)
    y = _safe_evaluate_univariate(data["fx"], x)
    integral = (h / 3.0) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))
    combined_plot, _ = _aux_plot_simpson_panels(
        f,
        a,
        b,
        x,
        panel_size=2,
        title="Simpson 1/3 compuesta",
        expression=data["fx"],
    )

    return _result_payload(
        title="Resultado: Simpson 1/3 compuesta",
        description=f"Integral aproximada de f(x) = {data['fx']}",
        summary={"integral aproximada": float(integral), "subintervalos": int(n), **xi_report},
        plot_series=[
            {"name": "f(x)", "x": x, "y": y, "style": "line"},
            {"name": "nodos", "x": x, "y": y, "style": "scatter"},
        ],
        plot_title="Simpson 1/3 compuesta",
        x_label="x",
        y_label="f(x)",
        table_headers=["i", "x_i", "f(x_i)"],
        table_rows=[[i, float(x[i]), float(y[i])] for i in range(len(x))],
        plot_image=combined_plot,
    )


def _run_simpson_38_simple(data: dict) -> dict:
    f = _build_callable(data["fx"], variables=("x",))
    a, b = _ensure_interval(data["a"], data["b"])
    h = (b - a) / 3.0
    x0 = a
    x1 = a + h
    x2 = a + 2 * h
    x3 = b

    y0 = float(_safe_evaluate_univariate(data["fx"], x0))
    y1 = float(_safe_evaluate_univariate(data["fx"], x1))
    y2 = float(_safe_evaluate_univariate(data["fx"], x2))
    y3 = float(_safe_evaluate_univariate(data["fx"], x3))
    integral = (3 * h / 8.0) * (y0 + 3 * y1 + 3 * y2 + y3)
    xi_report = _integration_truncation_report(
        data["fx"],
        data.get("xi", 0.5 * (a + b)),
        a,
        b,
        4,
        -((b - a) ** 5) / 6480.0,
        "f''''(xi)",
    )

    grid = np.linspace(a, b, 250)
    y_grid = _safe_evaluate_univariate(data["fx"], grid)
    combined_plot, _ = _aux_plot_simpson_panels(
        f,
        a,
        b,
        np.array([x0, x1, x2, x3]),
        panel_size=3,
        title="Simpson 3/8 simple",
        expression=data["fx"],
    )

    return _result_payload(
        title="Resultado: Simpson 3/8 simple",
        description=f"Integral aproximada de f(x) = {data['fx']}",
        summary={"integral aproximada": float(integral), "intervalo": f"[{a}, {b}]", **xi_report},
        plot_series=[
            {"name": "f(x)", "x": grid, "y": y_grid, "style": "line"},
            {
                "name": "nodos",
                "x": np.array([x0, x1, x2, x3]),
                "y": np.array([y0, y1, y2, y3]),
                "style": "scatter",
            },
        ],
        plot_title="Simpson 3/8 simple",
        x_label="x",
        y_label="f(x)",
        table_headers=["x", "f(x)"],
        table_rows=[[x0, y0], [x1, y1], [x2, y2], [x3, y3]],
        plot_image=combined_plot,
    )


def _run_simpson_38_composite(data: dict) -> dict:
    f = _build_callable(data["fx"], variables=("x",))
    a, b = _ensure_interval(data["a"], data["b"])
    n = _ensure_positive_int(data["n"], "n")
    if n % 3 != 0:
        raise SimulationInputError("Para Simpson 3/8 compuesta, n debe ser multiplo de 3.")

    h = (b - a) / n
    xi_report = _integration_truncation_report(
        data["fx"],
        data.get("xi", 0.5 * (a + b)),
        a,
        b,
        4,
        -((b - a) * (h ** 4)) / 80.0,
        "f''''(xi)",
    )

    x = np.linspace(a, b, n + 1)
    y = _safe_evaluate_univariate(data["fx"], x)

    mask = np.arange(1, n)
    sum_mult_3 = np.sum(y[(mask % 3) == 0])
    sum_not_mult_3 = np.sum(y[(mask % 3) != 0])
    integral = (3 * h / 8.0) * (y[0] + y[-1] + 2 * sum_mult_3 + 3 * sum_not_mult_3)
    combined_plot, _ = _aux_plot_simpson_panels(
        f,
        a,
        b,
        x,
        panel_size=3,
        title="Simpson 3/8 compuesta",
        expression=data["fx"],
    )

    return _result_payload(
        title="Resultado: Simpson 3/8 compuesta",
        description=f"Integral aproximada de f(x) = {data['fx']}",
        summary={"integral aproximada": float(integral), "subintervalos": int(n), **xi_report},
        plot_series=[
            {"name": "f(x)", "x": x, "y": y, "style": "line"},
            {"name": "nodos", "x": x, "y": y, "style": "scatter"},
        ],
        plot_title="Simpson 3/8 compuesta",
        x_label="x",
        y_label="f(x)",
        table_headers=["i", "x_i", "f(x_i)"],
        table_rows=[[i, float(x[i]), float(y[i])] for i in range(len(x))],
        plot_image=combined_plot,
    )


def _run_rectangle_rule(data: dict) -> dict:
    f = _build_callable(data["fx"], variables=("x",))
    a, b = _ensure_interval(data["a"], data["b"])
    n = _ensure_positive_int(data["n"], "n")

    h = (b - a) / n
    xi_report = _integration_truncation_report(
        data["fx"],
        data.get("xi", 0.5 * (a + b)),
        a,
        b,
        2,
        -((b - a) * (h ** 2)) / 24.0,
        "f''(xi)",
    )

    midpoints = a + (np.arange(n) + 0.5) * h
    f_mid = _safe_evaluate_univariate(data["fx"], midpoints)
    integral = h * np.sum(f_mid)

    dense_x = np.linspace(a, b, 400)
    dense_y = _safe_evaluate_univariate(data["fx"], dense_x)
    combined_plot = _aux_plot_rectangles_midpoint(
        f,
        a,
        b,
        n,
        "Regla del rectangulo (punto medio)",
        expression=data["fx"],
    )

    return _result_payload(
        title="Resultado: Regla del rectangulo (punto medio)",
        description=f"Integral aproximada de f(x) = {data['fx']}",
        summary={"integral aproximada": float(integral), "subintervalos": int(n), **xi_report},
        plot_series=[
            {"name": "f(x)", "x": dense_x, "y": dense_y, "style": "line"},
            {"name": "puntos medios", "x": midpoints, "y": f_mid, "style": "scatter"},
        ],
        plot_title="Regla del rectangulo",
        x_label="x",
        y_label="f(x)",
        table_headers=["i", "x_medio", "f(x_medio)"],
        table_rows=[[i + 1, float(midpoints[i]), float(f_mid[i])] for i in range(n)],
        plot_image=combined_plot,
    )


def _run_monte_carlo_simple(data: dict) -> dict:
    f = _build_callable(data["fx"], variables=("x",))
    a, b = _ensure_interval(data["a"], data["b"])
    samples = _ensure_positive_int(data["samples"], "samples")
    seed = int(data["seed"])
    confidence_level = _ensure_confidence_level(data.get("confidence_level", 0.95))

    rng = np.random.default_rng(seed)
    x_samples = rng.uniform(a, b, samples)
    y_samples = np.asarray(f(x_samples), dtype=float)
    stats = _monte_carlo_stats(y_samples, confidence_level, scale=(b - a))
    integral = stats["integral_estimate"]

    dense_x = np.linspace(a, b, 300)
    dense_y = _safe_evaluate_univariate(data["fx"], dense_x)
    value_distribution_plot = _aux_plot_distribution_with_standard_normal(
        y_samples,
        "Distribucion de valores y normal estandar",
        "z",
    )

    return _result_payload(
        title="Resultado: Integracion Monte Carlo simple",
        description=f"Integral aproximada de f(x) = {data['fx']}",
        summary={
            "integral aproximada": float(integral),
            "muestras": int(samples),
            "semilla": seed,
            "media muestral": stats["mean"],
            "varianza": stats["variance"],
            "desviacion estandar": stats["std_dev"],
            "error estandar": stats["std_error"],
            "nivel de confianza": confidence_level,
            "intervalo de confianza": f"[{stats['integral_ci_low']}, {stats['integral_ci_high']}]",
        },
        plot_series=[
            {"name": "f(x)", "x": dense_x, "y": dense_y, "style": "line"},
            {"name": "muestras", "x": x_samples[:1200], "y": y_samples[:1200], "style": "scatter"},
        ],
        plot_title="Monte Carlo simple",
        x_label="x",
        y_label="f(x)",
        table_headers=["x_i", "f(x_i)"],
        table_rows=[[float(x_samples[i]), float(y_samples[i])] for i in range(min(40, samples))],
        auxiliary_plots=[
            {
                "title": "Histograma de valores + normal estandar",
                "plot": value_distribution_plot,
            }
        ],
    )


def _run_monte_carlo_double(data: dict) -> dict:
    f = _build_callable(data["fxy"], variables=("x", "y"))
    ax, bx = _ensure_interval(data["ax"], data["bx"])
    ay, by = _ensure_interval(data["ay"], data["by"])
    samples = _ensure_positive_int(data["samples"], "samples")
    seed = int(data["seed"])
    confidence_level = _ensure_confidence_level(data.get("confidence_level", 0.95))

    rng = np.random.default_rng(seed)
    x_samples = rng.uniform(ax, bx, samples)
    y_samples = rng.uniform(ay, by, samples)
    f_samples = np.asarray(f(x_samples, y_samples), dtype=float)

    area = (bx - ax) * (by - ay)
    stats = _monte_carlo_stats(f_samples, confidence_level, scale=area)
    integral = stats["integral_estimate"]
    value_distribution_plot = _aux_plot_distribution_with_standard_normal(
        f_samples,
        "Distribucion de valores y normal estandar",
        "z",
    )

    return _result_payload(
        title="Resultado: Integracion Monte Carlo doble",
        description=f"Integral doble aproximada de f(x, y) = {data['fxy']}",
        summary={
            "integral aproximada": float(integral),
            "muestras": int(samples),
            "semilla": seed,
            "media muestral": stats["mean"],
            "varianza": stats["variance"],
            "desviacion estandar": stats["std_dev"],
            "error estandar": stats["std_error"],
            "nivel de confianza": confidence_level,
            "intervalo de confianza": f"[{stats['integral_ci_low']}, {stats['integral_ci_high']}]",
        },
        plot_series=[
            {
                "name": "muestras (x, y)",
                "x": x_samples[:1500],
                "y": y_samples[:1500],
                "style": "scatter",
            }
        ],
        plot_title="Muestras para Monte Carlo doble",
        x_label="x",
        y_label="y",
        table_headers=["x_i", "y_i", "f(x_i, y_i)"],
        table_rows=[
            [float(x_samples[i]), float(y_samples[i]), float(f_samples[i])]
            for i in range(min(40, samples))
        ],
        auxiliary_plots=[
            {
                "title": "Histograma de valores + normal estandar",
                "plot": value_distribution_plot,
            }
        ],
    )


def _run_euler(data: dict) -> dict:
    f = _build_callable(data["fxy"], variables=("x", "y"))
    x_current = float(data["x0"])
    y_current = float(data["y0"])
    h = _ensure_positive_step(data["h"])
    n = _ensure_positive_int(data["n"], "n")

    rows = []
    x_values = [x_current]
    y_values = [y_current]

    for idx in range(1, n + 1):
        slope = float(f(x_current, y_current))
        y_next = y_current + h * slope
        slope_next = float(f(x_current + h, y_next))
        y_heun = y_current + (h / 2.0) * (slope + slope_next)
        error_est = abs(y_heun - y_next)
        x_next = x_current + h

        if not np.isfinite(y_next):
            raise SimulationInputError("Se detecto inestabilidad numerica durante Euler.")
        if not np.isfinite(error_est):
            raise SimulationInputError("No se pudo estimar el error en Euler.")

        rows.append([idx, x_current, y_current, y_next, slope, error_est])
        x_values.append(x_next)
        y_values.append(y_next)
        x_current, y_current = x_next, y_next

    return _result_payload(
        title="Resultado: Metodo de Euler",
        description=f"EDO y' = f(x, y), f(x, y) = {data['fxy']}",
        summary={
            "paso h": h,
            "iteraciones": int(n),
            "x final": float(x_values[-1]),
            "y aproximado final": float(y_values[-1]),
            "error estimado final": float(rows[-1][5]) if rows else 0.0,
        },
        plot_series=[
            {"name": "y_n", "x": np.array(x_values), "y": np.array(y_values), "style": "line"}
        ],
        plot_title="Aproximacion por Metodo de Euler",
        x_label="x",
        y_label="y",
        table_headers=["n", "x_n", "y_n", "y_n+1", "f(x_n,y_n)", "error estimado"],
        table_rows=rows,
    )


def _run_heun(data: dict) -> dict:
    f = _build_callable(data["fxy"], variables=("x", "y"))
    x_current = float(data["x0"])
    y_current = float(data["y0"])
    h = _ensure_positive_step(data["h"])
    n = _ensure_positive_int(data["n"], "n")

    rows = []
    x_values = [x_current]
    y_values = [y_current]

    for idx in range(1, n + 1):
        k1 = float(f(x_current, y_current))
        y_predictor = y_current + h * k1
        k2 = float(f(x_current + h, y_predictor))
        y_next = y_current + (h / 2.0) * (k1 + k2)
        x_next = x_current + h

        if not np.isfinite(y_next):
            raise SimulationInputError("Se detecto inestabilidad numerica durante Heun.")

        rows.append([idx, x_current, y_current, k1, y_predictor, k2, y_next])
        x_values.append(x_next)
        y_values.append(y_next)
        x_current, y_current = x_next, y_next

    return _result_payload(
        title="Resultado: Metodo de Heun (Euler mejorado)",
        description=f"EDO y' = f(x, y), f(x, y) = {data['fxy']}",
        summary={
            "paso h": h,
            "iteraciones": int(n),
            "x final": float(x_values[-1]),
            "y aproximado final": float(y_values[-1]),
        },
        plot_series=[
            {"name": "y_n", "x": np.array(x_values), "y": np.array(y_values), "style": "line"}
        ],
        plot_title="Aproximacion por Metodo de Heun",
        x_label="x",
        y_label="y",
        table_headers=["n", "x_n", "y_n", "k1", "y_pred", "k2", "y_n+1"],
        table_rows=rows,
    )


def _run_rk4(data: dict) -> dict:
    f = _build_callable(data["fxy"], variables=("x", "y"))
    x_current = float(data["x0"])
    y_current = float(data["y0"])
    h = _ensure_positive_step(data["h"])
    n = _ensure_positive_int(data["n"], "n")


    rows = []
    x_values = [x_current]
    y_values = [y_current]

    for idx in range(1, n + 1):
        k1 = float(f(x_current, y_current))
        k2 = float(f(x_current + h / 2.0, y_current + h * k1 / 2.0))
        k3 = float(f(x_current + h / 2.0, y_current + h * k2 / 2.0))
        k4 = float(f(x_current + h, y_current + h * k3))

        y_next = y_current + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x_next = x_current + h

        if not np.isfinite(y_next):
            raise SimulationInputError("Se detecto inestabilidad numerica durante RK4.")

        rows.append([idx, x_current, y_current, k1, k2, k3, k4, y_next])
        x_values.append(x_next)
        y_values.append(y_next)
        x_current, y_current = x_next, y_next

    return _result_payload(
        title="Resultado: Metodo de Runge-Kutta de 4to orden",
        description=f"EDO y' = f(x, y), f(x, y) = {data['fxy']}",
        summary={
            "paso h": h,
            "iteraciones": int(n),
            "x final": float(x_values[-1]),
            "y aproximado final": float(y_values[-1]),
        },
        plot_series=[
            {"name": "y_n", "x": np.array(x_values), "y": np.array(y_values), "style": "line"}
        ],
        plot_title="Aproximacion por Runge-Kutta 4",
        x_label="x",
        y_label="y",
        table_headers=["n", "x_n", "y_n", "k1", "k2", "k3", "k4", "y_n+1"],
        table_rows=rows,
    )


def register_default_simulations() -> None:
    if len(list_simulations()) > 0:
        return

    register_spec(
        SimulationSpec(
            slug="punto-fijo",
            title="Metodo de punto fijo",
            description="Aproxima una raiz usando x(n+1)=g(x_n).",
            expressions=(
                ExpressionSpec("gx", "Funcion g(x)", ("x",), "cos(x)", "Variable permitida: x."),
            ),
            parameters=(
                ParameterSpec("x0", "Valor inicial x0", "float", 0.5),
                ParameterSpec("a", "Extremo inferior (visualizacion)", "float", -0.5),
                ParameterSpec("b", "Extremo superior (visualizacion)", "float", 1.5),
                ParameterSpec("tol", "Tolerancia", "float", 1e-6, 1e-12, 1.0),
                ParameterSpec("max_iter", "Max iteraciones", "int", 100, 1, 10000),
            ),
            runner=_run_fixed_point,
        )
    )

    register_spec(
        SimulationSpec(
            slug="biseccion",
            title="Metodo de biseccion",
            description="Busca una raiz en [a,b] con cambio de signo.",
            expressions=(
                ExpressionSpec("fx", "Funcion f(x)", ("x",), "x**3 - x - 2", "Variable permitida: x."),
            ),
            parameters=(
                ParameterSpec("a", "Extremo a", "float", 1.0),
                ParameterSpec("b", "Extremo b", "float", 2.0),
                ParameterSpec("tol", "Tolerancia", "float", 1e-6, 1e-12, 1.0),
                ParameterSpec("max_iter", "Max iteraciones", "int", 100, 1, 10000),
            ),
            runner=_run_bisection,
        )
    )

    register_spec(
        SimulationSpec(
            slug="newton-raphson",
            title="Metodo de Newton-Raphson",
            description="Busca una raiz usando derivada de la funcion.",
            expressions=(
                ExpressionSpec("fx", "Funcion f(x)", ("x",), "x**3 - x - 2", "Variable permitida: x."),
                ExpressionSpec("dfx", "Derivada f'(x)", ("x",), "3*x**2 - 1", "Variable permitida: x."),
            ),
            parameters=(
                ParameterSpec("x0", "Valor inicial x0", "float", 1.5),
                ParameterSpec("tol", "Tolerancia", "float", 1e-6, 1e-12, 1.0),
                ParameterSpec("max_iter", "Max iteraciones", "int", 100, 1, 10000),
            ),
            runner=_run_newton_raphson,
        )
    )

    register_spec(
        SimulationSpec(
            slug="aitken-delta-cuadrado",
            title="Metodo delta cuadrado de Aitken",
            description="Acelera convergencia de una iteracion de punto fijo.",
            expressions=(
                ExpressionSpec("gx", "Funcion g(x)", ("x",), "cos(x)", "Variable permitida: x."),
            ),
            parameters=(
                ParameterSpec("x0", "Valor inicial x0", "float", 0.5),
                ParameterSpec("tol", "Tolerancia", "float", 1e-6, 1e-12, 1.0),
                ParameterSpec("max_iter", "Max iteraciones", "int", 50, 1, 10000),
            ),
            runner=_run_aitken_delta2,
        )
    )

    register_spec(
        SimulationSpec(
            slug="interpolacion-lagrange",
            title="Interpolacion de Lagrange",
            description="Construye P(x) desde nodos (x_i, y_i).",
            expressions=(
                ExpressionSpec(
                    "x_points",
                    "Nodos x (CSV)",
                    tuple(),
                    "0,1,2,3",
                    "Ingrese valores separados por coma.",
                ),
                ExpressionSpec(
                    "y_points",
                    "Nodos y (CSV)",
                    tuple(),
                    "1,3,2,5",
                    "Ingrese valores separados por coma.",
                ),
            ),
            parameters=(
                ParameterSpec("x_eval", "x a evaluar", "float", 1.5),
            ),
            runner=_run_lagrange_interpolation,
        )
    )

    register_spec(
        SimulationSpec(
            slug="newton-diferencias-divididas",
            title="Diferencias divididas de Newton",
            description="Interpolacion con polinomio en forma de Newton.",
            expressions=(
                ExpressionSpec("x_points", "Nodos x (CSV)", tuple(), "0,1,2,3", "Valores separados por coma."),
                ExpressionSpec("y_points", "Nodos y (CSV)", tuple(), "1,3,2,5", "Valores separados por coma."),
            ),
            parameters=(
                ParameterSpec("x_eval", "x a evaluar", "float", 1.5),
            ),
            runner=_run_newton_divided_interpolation,
        )
    )

    register_spec(
        SimulationSpec(
            slug="trapecio-simple",
            title="Regla del trapecio simple",
            description="Aproxima integral con un solo trapecio.",
            expressions=(
                ExpressionSpec("fx", "Funcion f(x)", ("x",), "sin(x)", "Variable permitida: x."),
            ),
            parameters=(
                ParameterSpec("a", "Limite inferior a", "float", 0.0),
                ParameterSpec("b", "Limite superior b", "float", 3.14159),
                ParameterSpec("xi", "Punto xi (en [a, b])", "float", 1.5707963267948966),
            ),
            runner=_run_trapezoid_simple,
        )
    )

    register_spec(
        SimulationSpec(
            slug="trapecio-compuesta",
            title="Regla del trapecio compuesta",
            description="Aproxima integral con varios subintervalos.",
            expressions=(
                ExpressionSpec("fx", "Funcion f(x)", ("x",), "sin(x)", "Variable permitida: x."),
            ),
            parameters=(
                ParameterSpec("a", "Limite inferior a", "float", 0.0),
                ParameterSpec("b", "Limite superior b", "float", 3.14159),
                ParameterSpec("n", "Cantidad de subintervalos", "int", 12, 1, 100000),
                ParameterSpec("xi", "Punto xi (en [a, b])", "float", 1.5707963267948966),
            ),
            runner=_run_trapezoid_composite,
        )
    )

    register_spec(
        SimulationSpec(
            slug="simpson-13-simple",
            title="Regla de Simpson 1/3 simple",
            description="Aproxima integral con una aplicacion de Simpson 1/3.",
            expressions=(
                ExpressionSpec("fx", "Funcion f(x)", ("x",), "sin(x)", "Variable permitida: x."),
            ),
            parameters=(
                ParameterSpec("a", "Limite inferior a", "float", 0.0),
                ParameterSpec("b", "Limite superior b", "float", 3.14159),
                ParameterSpec("xi", "Punto xi (en [a, b])", "float", 1.5707963267948966),
            ),
            runner=_run_simpson_13_simple,
        )
    )

    register_spec(
        SimulationSpec(
            slug="simpson-13-compuesta",
            title="Regla de Simpson 1/3 compuesta",
            description="Aproxima integral con Simpson 1/3 en n subintervalos.",
            expressions=(
                ExpressionSpec("fx", "Funcion f(x)", ("x",), "sin(x)", "Variable permitida: x."),
            ),
            parameters=(
                ParameterSpec("a", "Limite inferior a", "float", 0.0),
                ParameterSpec("b", "Limite superior b", "float", 3.14159),
                ParameterSpec("n", "Cantidad de subintervalos (par)", "int", 12, 2, 100000),
                ParameterSpec("xi", "Punto xi (en [a, b])", "float", 1.5707963267948966),
            ),
            runner=_run_simpson_13_composite,
        )
    )

    register_spec(
        SimulationSpec(
            slug="simpson-38-simple",
            title="Regla de Simpson 3/8 simple",
            description="Aproxima integral con una aplicacion de Simpson 3/8.",
            expressions=(
                ExpressionSpec("fx", "Funcion f(x)", ("x",), "sin(x)", "Variable permitida: x."),
            ),
            parameters=(
                ParameterSpec("a", "Limite inferior a", "float", 0.0),
                ParameterSpec("b", "Limite superior b", "float", 3.14159),
                ParameterSpec("xi", "Punto xi (en [a, b])", "float", 1.5707963267948966),
            ),
            runner=_run_simpson_38_simple,
        )
    )

    register_spec(
        SimulationSpec(
            slug="simpson-38-compuesta",
            title="Regla de Simpson 3/8 compuesta",
            description="Aproxima integral con Simpson 3/8 en n subintervalos.",
            expressions=(
                ExpressionSpec("fx", "Funcion f(x)", ("x",), "sin(x)", "Variable permitida: x."),
            ),
            parameters=(
                ParameterSpec("a", "Limite inferior a", "float", 0.0),
                ParameterSpec("b", "Limite superior b", "float", 3.14159),
                ParameterSpec("n", "Cantidad de subintervalos (multiplo de 3)", "int", 12, 3, 100000),
                ParameterSpec("xi", "Punto xi (en [a, b])", "float", 1.5707963267948966),
            ),
            runner=_run_simpson_38_composite,
        )
    )

    register_spec(
        SimulationSpec(
            slug="regla-rectangulo",
            title="Regla del rectangulo",
            description="Aproxima integral con la variante de punto medio.",
            expressions=(
                ExpressionSpec("fx", "Funcion f(x)", ("x",), "sin(x)", "Variable permitida: x."),
            ),
            parameters=(
                ParameterSpec("a", "Limite inferior a", "float", 0.0),
                ParameterSpec("b", "Limite superior b", "float", 3.14159),
                ParameterSpec("n", "Cantidad de subintervalos", "int", 10, 1, 100000),
                ParameterSpec("xi", "Punto xi (en [a, b])", "float", 1.5707963267948966),
            ),
            runner=_run_rectangle_rule,
        )
    )

    register_spec(
        SimulationSpec(
            slug="monte-carlo-simple",
            title="Integracion de Monte Carlo simple",
            description="Aproxima una integral 1D mediante muestreo aleatorio.",
            expressions=(
                ExpressionSpec("fx", "Funcion f(x)", ("x",), "sin(x)", "Variable permitida: x."),
            ),
            parameters=(
                ParameterSpec("a", "Limite inferior a", "float", 0.0),
                ParameterSpec("b", "Limite superior b", "float", 3.14159),
                ParameterSpec("samples", "Cantidad de muestras", "int", 5000, 10, 1_000_000),
                ParameterSpec(
                    "confidence_level",
                    "Nivel de confianza (0-1)",
                    "float",
                    0.95,
                    0.5,
                    0.999,
                    "Ejemplos: 0.90, 0.95, 0.99",
                ),
                _DEFAULT_SEED_PARAM,
            ),
            runner=_run_monte_carlo_simple,
        )
    )

    register_spec(
        SimulationSpec(
            slug="monte-carlo-doble",
            title="Integracion de Monte Carlo doble",
            description="Aproxima una integral doble por muestreo aleatorio.",
            expressions=(
                ExpressionSpec("fxy", "Funcion f(x, y)", ("x", "y"), "x**2 + y**2", "Variables permitidas: x, y."),
            ),
            parameters=(
                ParameterSpec("ax", "Limite inferior x", "float", 0.0),
                ParameterSpec("bx", "Limite superior x", "float", 1.0),
                ParameterSpec("ay", "Limite inferior y", "float", 0.0),
                ParameterSpec("by", "Limite superior y", "float", 1.0),
                ParameterSpec("samples", "Cantidad de muestras", "int", 7000, 10, 1_000_000),
                ParameterSpec(
                    "confidence_level",
                    "Nivel de confianza (0-1)",
                    "float",
                    0.95,
                    0.5,
                    0.999,
                    "Ejemplos: 0.90, 0.95, 0.99",
                ),
                _DEFAULT_SEED_PARAM,
            ),
            runner=_run_monte_carlo_double,
        )
    )

    register_spec(
        SimulationSpec(
            slug="euler",
            title="Metodo de Euler",
            description="Resuelve y' = f(x, y) mediante Euler explicito.",
            expressions=(
                ExpressionSpec(
                    "fxy",
                    "Funcion f(x, y)",
                    ("x", "y"),
                    "x + y",
                    "Variables permitidas: x, y.",
                ),
            ),
            parameters=(
                ParameterSpec("x0", "Valor inicial x0", "float", 0.0),
                ParameterSpec("y0", "Valor inicial y0", "float", 1.0),
                ParameterSpec("h", "Paso h", "float", 0.1, 1e-6, 10.0),
                ParameterSpec("n", "Cantidad de pasos", "int", 20, 1, 100000),
            ),
            runner=_run_euler,
        )
    )

    register_spec(
        SimulationSpec(
            slug="heun-euler-mejorado",
            title="Metodo de Heun (Euler mejorado)",
            description="Resuelve y' = f(x, y) usando predictor-corrector de Heun.",
            expressions=(
                ExpressionSpec(
                    "fxy",
                    "Funcion f(x, y)",
                    ("x", "y"),
                    "x + y",
                    "Variables permitidas: x, y.",
                ),
            ),
            parameters=(
                ParameterSpec("x0", "Valor inicial x0", "float", 0.0),
                ParameterSpec("y0", "Valor inicial y0", "float", 1.0),
                ParameterSpec("h", "Paso h", "float", 0.1, 1e-6, 10.0),
                ParameterSpec("n", "Cantidad de pasos", "int", 20, 1, 100000),
            ),
            runner=_run_heun,
        )
    )

    register_spec(
        SimulationSpec(
            slug="runge-kutta-4",
            title="Metodo de Runge-Kutta 4to orden",
            description="Resuelve y' = f(x, y) con Runge-Kutta de cuarto orden.",
            expressions=(
                ExpressionSpec(
                    "fxy",
                    "Funcion f(x, y)",
                    ("x", "y"),
                    "x + y",
                    "Variables permitidas: x, y.",
                ),
            ),
            parameters=(
                ParameterSpec("x0", "Valor inicial x0", "float", 0.0),
                ParameterSpec("y0", "Valor inicial y0", "float", 1.0),
                ParameterSpec("h", "Paso h", "float", 0.1, 1e-6, 10.0),
                ParameterSpec("n", "Cantidad de pasos", "int", 20, 1, 100000),
            ),
            runner=_run_rk4,
        )
    )


def execute_simulation(slug: str, cleaned_data: dict) -> dict:
    spec = get_simulation(slug)
    if spec.runner is None:
        raise SimulationInputError("La simulacion seleccionada no tiene un motor configurado.")

    precision = _normalize_precision(cleaned_data)
    normalized_input = _apply_precision(cleaned_data, precision)
    normalized_input["precision"] = precision

    try:
        result = spec.runner(normalized_input)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError, FloatingPointError) as exc:
        raise SimulationInputError(str(exc)) from exc

    rounded_result = _apply_precision(result, precision)

    return {
        "slug": spec.slug,
        "title": spec.title,
        "input": normalized_input,
        "result": rounded_result,
    }


def list_simulations() -> list[SimulationSpec]:
    from .registry import list_specs

    return list_specs()


def get_simulation(slug: str) -> SimulationSpec:
    from .registry import get_spec

    return get_spec(slug)

