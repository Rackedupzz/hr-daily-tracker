"""Link rewriting shared by the static publisher and the live Vercel function.

Kept apart from mlb_hr.publish so the function can import it without pulling
in Flask, pandas and the model stack.
"""
import re

_QUERY_LINK = re.compile(r"/(results|homer)\?date=(\d{4}-\d{2}-\d{2})")


def staticize(html: str) -> str:
    """`/results?date=X` -> `/results/X/`: a static host cannot route on a query."""
    return _QUERY_LINK.sub(r"/\1/\2/", html)
