import numpy as np
import plotly.graph_objects as go
from sympy import Abs, E, Float, Integer, Rational, Symbol, cos, exp, log, pi, sin, sqrt, tan
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

    allowed_free_symbols = {local_scope[name] for name in variables}
    if not parsed.free_symbols.issubset(allowed_free_symbols):
        raise SimulationInputError(
            f"La expresion '{expression}' contiene simbolos no permitidos."
        )

    ordered_symbols = tuple(local_scope[name] for name in variables)
    return lambdify(ordered_symbols, parsed, modules=["numpy"])


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


def _ensure_positive_step(value: float, field_name: str = "h") -> float:
    step = float(value)
    if step <= 0:
        raise SimulationInputError(f"{field_name} debe ser mayor a 0.")
    return step


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


def _line_plot(
    series: list[dict],
    title: str,
    x_label: str,
    y_label: str,
    equal_aspect: bool = False,
) -> str:
    fig = go.Figure()
    for item in series:
        if item.get("style") == "scatter":
            fig.add_trace(
                go.Scatter(
                    x=_to_plotly_values(item["x"]),
                    y=_to_plotly_values(item["y"]),
                    mode="markers",
                    name=item["name"],
                    marker=item.get("marker", {"size": 6}),
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=_to_plotly_values(item["x"]),
                    y=_to_plotly_values(item["y"]),
                    mode="lines",
                    name=item["name"],
                    line=item.get("line", {"width": 2.5}),
                )
            )

    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    fig.update_xaxes(showgrid=True, gridcolor=_GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID_COLOR)
    if equal_aspect:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return _figure_to_html(fig)


def _aux_plot_trapezoids(f, a: float, b: float, x_nodes: np.ndarray, title: str) -> str:
    dense_x = np.linspace(a, b, 600)
    dense_y = np.asarray(f(dense_x), dtype=float)

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

    y_nodes = np.asarray(f(x_nodes), dtype=float)
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


def _aux_plot_rectangles_midpoint(f, a: float, b: float, n: int, title: str) -> str:
    h = (b - a) / n
    midpoints = a + (np.arange(n) + 0.5) * h
    heights = np.asarray(f(midpoints), dtype=float)

    dense_x = np.linspace(a, b, 600)
    dense_y = np.asarray(f(dense_x), dtype=float)

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


def _aux_plot_bisection_function(f, a: float, b: float, rows: list[list], title: str) -> str:
    dense_x = np.linspace(a, b, 700)
    dense_y = np.asarray(f(dense_x), dtype=float)

    left_edges = np.array([row[1] for row in rows], dtype=float)
    right_edges = np.array([row[2] for row in rows], dtype=float)
    midpoints = np.array([row[3] for row in rows], dtype=float)

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

    interval_y = np.min(dense_y) - 0.08 * max(1.0, float(np.ptp(dense_y)))
    if not np.isfinite(interval_y):
        interval_y = -0.5

    for idx, (left, right, midpoint) in enumerate(zip(left_edges, right_edges, midpoints), start=1):
        fig.add_trace(
            go.Scatter(
                x=[left, right],
                y=[interval_y, interval_y],
                mode="lines",
                line={"color": "rgba(34, 197, 94, 0.30)", "width": 2.2},
                name="intervalos" if idx == 1 else None,
                showlegend=idx == 1,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[midpoint],
                y=[float(f(midpoint))],
                mode="markers",
                name="c_n" if idx == 1 else None,
                showlegend=idx == 1,
                marker={
                    "size": 8,
                    "color": _NODE_COLOR,
                    "line": {"color": "#f8fafc", "width": 0.5},
                },
                text=[f"n={int(rows[idx - 1][0])}"],
                hovertemplate="iteracion=%{text}<br>c_n=%{x:.6f}<br>f(c_n)=%{y:.6f}<extra></extra>",
            )
        )

    fig.update_layout(title=title, xaxis_title="x", yaxis_title="f(x)", legend={"orientation": "h", "yanchor": "bottom", "y": -0.18, "x": 0})
    fig.update_xaxes(showgrid=True, gridcolor=_GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID_COLOR)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return _figure_to_html(fig)


def _aux_plot_simpson_panels(
    f,
    a: float,
    b: float,
    x_nodes: np.ndarray,
    panel_size: int,
    title: str,
) -> tuple[str, int]:
    dense_x = np.linspace(a, b, 700)
    dense_y = np.asarray(f(dense_x), dtype=float)
    y_nodes = np.asarray(f(x_nodes), dtype=float)

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


def _run_fixed_point(data: dict) -> dict:
    g = _build_callable(data["gx"], variables=("x",))
    tol = float(data["tol"])
    max_iter = _ensure_positive_int(data["max_iter"], "max_iter")
    x_current = float(data["x0"])

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

    return _result_payload(
        title="Resultado: Metodo de punto fijo",
        description=f"Iteracion x(n+1) = g(x(n)), g(x) = {data['gx']}",
        summary={
            "iteraciones": len(iterations),
            "aproximacion": float(xn[-1]),
            "error final": float(iterations[-1][3]),
        },
        plot_series=[{"name": "x_n", "x": it, "y": xn, "style": "line"}],
        plot_title="Convergencia de punto fijo",
        x_label="iteracion",
        y_label="x_n",
        table_headers=["n", "x_n", "x_n+1", "error"],
        table_rows=iterations,
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

    function_plot = _aux_plot_bisection_function(f, plot_a, plot_b, rows, "Funcion y intervalos de biseccion")
    convergence_plot = _line_plot(
        [
            {
                "name": "|f(c_n)|",
                "x": np.arange(1, len(fmid_values) + 1),
                "y": np.abs(np.array(fmid_values)),
                "style": "line",
                "line": {"color": "#38bdf8", "width": 2.4},
            },
            {
                "name": "|b_n-a_n|",
                "x": np.arange(1, len(rows) + 1),
                "y": np.array([row[5] for row in rows], dtype=float),
                "style": "line",
                "line": {"color": "#f59e0b", "width": 2.0, "dash": "dot"},
            },
        ],
        "Convergencia por iteracion",
        "iteracion",
        "error",
        equal_aspect=True,
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
            {"name": "|f(c_n)|", "x": np.arange(1, len(fmid_values) + 1), "y": np.abs(np.array(fmid_values)), "style": "line"}
        ],
        plot_title="Convergencia por iteracion",
        x_label="iteracion",
        y_label="error",
        table_headers=["n", "a", "b", "c", "f(c)", "|b-a|"],
        table_rows=rows,
        plot_image=convergence_plot,
        auxiliary_plots=[
            {
                "title": "Funcion y intervalos de biseccion",
                "note": "La escala inicial se fija 1:1 para respetar la geometria del intervalo.",
                "plot": function_plot,
            }
        ],
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
    fa = float(f(a))
    fb = float(f(b))
    integral = (b - a) * (fa + fb) / 2.0

    grid = np.linspace(a, b, 250)
    y_grid = np.asarray(f(grid), dtype=float)
    combined_plot = _aux_plot_trapezoids(f, a, b, np.array([a, b]), "Trapecio simple")

    return _result_payload(
        title="Resultado: Regla del trapecio simple",
        description=f"Integral aproximada de f(x) = {data['fx']}",
        summary={"integral aproximada": float(integral), "intervalo": f"[{a}, {b}]"},
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

    x = np.linspace(a, b, n + 1)
    y = np.asarray(f(x), dtype=float)
    h = (b - a) / n
    integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    combined_plot = _aux_plot_trapezoids(f, a, b, x, "Trapecio compuesto")

    return _result_payload(
        title="Resultado: Regla del trapecio compuesta",
        description=f"Integral aproximada de f(x) = {data['fx']}",
        summary={"integral aproximada": float(integral), "subintervalos": int(n)},
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

    fa = float(f(a))
    fm = float(f(m))
    fb = float(f(b))
    integral = (b - a) * (fa + 4 * fm + fb) / 6.0

    grid = np.linspace(a, b, 250)
    y_grid = np.asarray(f(grid), dtype=float)
    combined_plot, _ = _aux_plot_simpson_panels(
        f,
        a,
        b,
        np.array([a, m, b]),
        panel_size=2,
        title="Simpson 1/3 simple",
    )

    return _result_payload(
        title="Resultado: Simpson 1/3 simple",
        description=f"Integral aproximada de f(x) = {data['fx']}",
        summary={"integral aproximada": float(integral), "intervalo": f"[{a}, {b}]"},
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

    x = np.linspace(a, b, n + 1)
    y = np.asarray(f(x), dtype=float)
    h = (b - a) / n
    integral = (h / 3.0) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))
    combined_plot, _ = _aux_plot_simpson_panels(
        f,
        a,
        b,
        x,
        panel_size=2,
        title="Simpson 1/3 compuesta",
    )

    return _result_payload(
        title="Resultado: Simpson 1/3 compuesta",
        description=f"Integral aproximada de f(x) = {data['fx']}",
        summary={"integral aproximada": float(integral), "subintervalos": int(n)},
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

    y0 = float(f(x0))
    y1 = float(f(x1))
    y2 = float(f(x2))
    y3 = float(f(x3))
    integral = (3 * h / 8.0) * (y0 + 3 * y1 + 3 * y2 + y3)

    grid = np.linspace(a, b, 250)
    y_grid = np.asarray(f(grid), dtype=float)
    combined_plot, _ = _aux_plot_simpson_panels(
        f,
        a,
        b,
        np.array([x0, x1, x2, x3]),
        panel_size=3,
        title="Simpson 3/8 simple",
    )

    return _result_payload(
        title="Resultado: Simpson 3/8 simple",
        description=f"Integral aproximada de f(x) = {data['fx']}",
        summary={"integral aproximada": float(integral), "intervalo": f"[{a}, {b}]"},
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

    x = np.linspace(a, b, n + 1)
    y = np.asarray(f(x), dtype=float)
    h = (b - a) / n

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
    )

    return _result_payload(
        title="Resultado: Simpson 3/8 compuesta",
        description=f"Integral aproximada de f(x) = {data['fx']}",
        summary={"integral aproximada": float(integral), "subintervalos": int(n)},
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
    midpoints = a + (np.arange(n) + 0.5) * h
    f_mid = np.asarray(f(midpoints), dtype=float)
    integral = h * np.sum(f_mid)

    dense_x = np.linspace(a, b, 400)
    dense_y = np.asarray(f(dense_x), dtype=float)
    combined_plot = _aux_plot_rectangles_midpoint(f, a, b, n, "Regla del rectangulo (punto medio)")

    return _result_payload(
        title="Resultado: Regla del rectangulo (punto medio)",
        description=f"Integral aproximada de f(x) = {data['fx']}",
        summary={"integral aproximada": float(integral), "subintervalos": int(n)},
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

    rng = np.random.default_rng(seed)
    x_samples = rng.uniform(a, b, samples)
    y_samples = np.asarray(f(x_samples), dtype=float)

    integral = (b - a) * float(np.mean(y_samples))

    dense_x = np.linspace(a, b, 300)
    dense_y = np.asarray(f(dense_x), dtype=float)

    return _result_payload(
        title="Resultado: Integracion Monte Carlo simple",
        description=f"Integral aproximada de f(x) = {data['fx']}",
        summary={
            "integral aproximada": float(integral),
            "muestras": int(samples),
            "semilla": seed,
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
    )


def _run_monte_carlo_double(data: dict) -> dict:
    f = _build_callable(data["fxy"], variables=("x", "y"))
    ax, bx = _ensure_interval(data["ax"], data["bx"])
    ay, by = _ensure_interval(data["ay"], data["by"])
    samples = _ensure_positive_int(data["samples"], "samples")
    seed = int(data["seed"])

    rng = np.random.default_rng(seed)
    x_samples = rng.uniform(ax, bx, samples)
    y_samples = rng.uniform(ay, by, samples)
    f_samples = np.asarray(f(x_samples, y_samples), dtype=float)

    area = (bx - ax) * (by - ay)
    integral = area * float(np.mean(f_samples))

    return _result_payload(
        title="Resultado: Integracion Monte Carlo doble",
        description=f"Integral doble aproximada de f(x, y) = {data['fxy']}",
        summary={
            "integral aproximada": float(integral),
            "muestras": int(samples),
            "semilla": seed,
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

