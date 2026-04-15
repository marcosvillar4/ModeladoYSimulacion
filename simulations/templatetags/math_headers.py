import re

from django import template
from django.utils.html import escape
from django.utils.safestring import mark_safe

register = template.Library()

_MATH_TOKEN_RE = re.compile(
    r"(\\[a-zA-Z]+(?:\{[^{}]*\})*"  # Comandos LaTeX: \frac{1}{x}, \sqrt{x}
    r"|[a-zA-Z](?:_[a-zA-Z0-9+\-]+)?\([^()]*\)"  # f(x), f(x,y), y_n(x)
    r"|[a-zA-Z]_[a-zA-Z0-9+\-]+"  # x_n, y_n+1
    r"|[a-zA-Z]\^\{[^{}]+\}"  # x^{n+1}
    r"|[a-zA-Z]\^[a-zA-Z0-9+\-]+"  # x^2
    r"|[a-zA-Z]')"  # y'
)


def _wrap_math(match: re.Match[str]) -> str:
    token = match.group(0)
    return f"\\({token}\\)"


@register.filter
def render_math_text(value):
    text = str(value)
    if not text.strip():
        return text

    escaped = escape(text)
    rendered = _MATH_TOKEN_RE.sub(_wrap_math, escaped)
    return mark_safe(rendered)

