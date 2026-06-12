from __future__ import annotations

import re
import textwrap
from typing import Any


PLACEHOLDER = re.compile(r"{{\s*([a-zA-Z0-9_.]+)\s*}}")


def render_sql(template: str, context: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        path = match.group(1).split(".")
        value: Any = context
        for key in path:
            value = value[key]
        return str(value)

    sql = PLACEHOLDER.sub(replace, template)
    lines = [line.rstrip() for line in textwrap.dedent(sql).strip().splitlines()]
    sql = "\n".join(line for line in lines if line.strip())
    return sql
