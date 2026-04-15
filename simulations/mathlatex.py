from __future__ import annotations

from sympy import cos, exp, log, pi, sin, sqrt, tan

_LATEX_TO_SYMPY = {
    r"\sin": "sin",
    r"\cos": "cos",
    r"\tan": "tan",
    r"\log": "log",
    r"\ln": "log",
    r"\exp": "exp",
    r"\sqrt": "sqrt",
    r"\pi": "pi",
    r"\cdot": "*",
    r"\times": "*",
    r"\left": "",
    r"\right": "",
}

def _extract_braced_group(text: str, start: int) -> tuple[str, int]:
    if start >= len(text) or text[start] != "{":
        raise ValueError("Se esperaba un bloque entre llaves.")

    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : idx], idx + 1

    raise ValueError("No se cerro un bloque entre llaves.")


def _replace_command_blocks(text: str) -> str:
    out: list[str] = []
    i = 0

    while i < len(text):
        if text.startswith(r"\frac", i):
            i += len(r"\frac")
            numerator, next_i = _extract_braced_group(text, i)
            denominator, next_i = _extract_braced_group(text, next_i)
            out.append(f"(({normalize_latex_expression(numerator)}))/(({normalize_latex_expression(denominator)}))")
            i = next_i
            continue

        if text.startswith(r"\sqrt", i):
            i += len(r"\sqrt")
            block, next_i = _extract_braced_group(text, i)
            out.append(f"sqrt(({normalize_latex_expression(block)}))")
            i = next_i
            continue

        matched = False
        for latex_token, sympy_token in _LATEX_TO_SYMPY.items():
            if text.startswith(latex_token, i):
                out.append(sympy_token)
                i += len(latex_token)
                matched = True
                break
        if matched:
            continue

        out.append(text[i])
        i += 1

    return "".join(out)


def normalize_latex_expression(expression: str) -> str:
    text = str(expression).strip()
    if not text:
        return text

    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1].strip()

    text = text.replace(" ", "")
    text = _replace_command_blocks(text)
    text = text.replace("{", "(").replace("}", ")")
    text = text.replace("\\", "")
    return text

