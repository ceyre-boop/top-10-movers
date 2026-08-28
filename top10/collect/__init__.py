"""Live collection of Robinhood's movers feeds.

See docs/LABEL_SPEC.md "Proxy validation" -- `fetch_top_movers` /
`load_captured_movers` are the only source of ground truth against which
the proxy label can ever be validated, and it can only be collected
forward from today. There is no historical archive. `fetch_sp500_movers`
is a separate, mega-cap-only feed kept for context and must never be
mistaken for the true top-movers list -- see top10/collect/rh_movers.py's
module docstring and top10/collect/overlap.py for the guardrails.
"""

from top10.collect.rh_movers import load_captured_movers

__all__ = ["load_captured_movers"]
