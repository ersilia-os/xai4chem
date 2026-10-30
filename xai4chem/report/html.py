"""Self-contained HTML in the Ersilia house style.

The stylesheet (``assets/ersilia.css``) is the Ersilia design system, inlined into every page, and
molecule drawings are inline SVG and the model report's plots are embedded PNGs, so a report is one file that can be emailed, opened offline or
published as an Artifact. The pages commit to a single light theme on purpose: the Ersilia brand is light.
"""

import html
from importlib import resources

SOURCE_URL = "https://github.com/ersilia-os/xai4chem"
FAVICON = "%236C5CE7"

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>{title}</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2032%2032'%3E%3Ccircle%20cx='16'%20cy='16'%20r='16'%20fill='{favicon}'/%3E%3C/svg%3E">
<style>
{css}
{extra_css}
</style>
</head>
<body>
<main class="{layout}">
<header class="brandhead">
  <span class="eyebrow brand">{eyebrow} · Ersilia Open Source Initiative</span>
  <h1 class="wordmark">{wordmark}</h1>
  <p class="muted" style="margin:.4em 0 0; max-width:72ch;">{lede}</p>
</header>
{body}
<footer class="credit">
  Brought to you by the <a href="https://ersilia.io" target="_blank" rel="noopener">Ersilia Open Source Initiative</a>
  — a tech-nonprofit fueling sustainable research in the Global South.<br>
  <a href="{source}" target="_blank" rel="noopener">⌥ Source code on GitHub ↗</a>
</footer>
</main>
</body>
</html>
"""

# Page-specific additions, all built from the design-system tokens.
EXTRA_CSS = """
.kpis { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:12px; margin-top:14px; }
.figrow { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:14px; }
.figrow figure { margin:0; }
.figrow2 { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,2fr); gap:14px; align-items:start; }
@media (max-width:760px) { .figrow2 { grid-template-columns:1fr; } }
.figrow2 figure { margin:0; }
.ibar { display:inline-block; vertical-align:middle; height:6px; border-radius:999px; background:var(--brand); opacity:.75; }
td .frag svg { width:64px; height:64px; display:block; }
figcaption { font-size:11.5px; color:var(--muted); margin-top:6px; }
.figgrid { display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:16px; margin-top:14px; align-items:start; }
.figgrid figure { margin:0; }
.figgrid img { width:100%; height:auto; display:block; }
.figgrid figcaption b { color:var(--ink); font-weight:600; }
.contrib { display:grid; grid-template-columns:minmax(0,1fr) 120px 58px; gap:4px 10px; align-items:center; font-size:12px; }
.contrib .lbl { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.contrib .bar { position:relative; height:8px; background:var(--surface-2); border-radius:999px; }
.contrib .bar i { position:absolute; top:0; bottom:0; border-radius:999px; }
.contrib .bar i.pos { left:50%; background:var(--crimson); }
.contrib .bar i.neg { right:50%; background:var(--cobalt); }
.contrib .v { text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums; }
.molcard { margin-top:18px; }
.molcard .head { display:flex; flex-wrap:wrap; gap:8px 18px; align-items:baseline; }
.molcard .smi { font-family:var(--mono); font-size:11.5px; color:var(--muted); word-break:break-all; }
.views { display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,205px)); gap:10px; margin-top:10px; }
.views figure { margin:0; text-align:center; }
.views svg { width:100%; height:auto; }
.split { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:18px; margin-top:10px; }
.legend { display:flex; gap:14px; font-size:11.5px; color:var(--muted); margin-top:6px; }
.legend span::before { content:""; display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:-1px; background:var(--c); }
details > summary { cursor:pointer; color:var(--brand); font-weight:450; }
"""


def esc(value):
    """HTML-escape any value as text."""
    return html.escape(str(value))


def css():
    """The vendored Ersilia design-system stylesheet."""
    return resources.files("xai4chem").joinpath("report/assets/ersilia.css").read_text()


def page(title, eyebrow, wordmark, lede, body, layout="document"):
    """A complete, script-free HTML page around ``body``."""
    return _PAGE.format(
        title=esc(title),
        favicon=FAVICON,
        css=css(),
        extra_css=EXTRA_CSS,
        layout=layout,
        eyebrow=esc(eyebrow),
        wordmark=wordmark,
        lede=lede,
        body=body,
        source=SOURCE_URL,
    )


def section(title, content, hint=None):
    """A titled section, with an optional faint hint after the title."""
    hint_html = f' <span class="faint" style="font-weight:400">· {esc(hint)}</span>' if hint else ""
    return f'<section class="section"><h2>{esc(title)}{hint_html}</h2>{content}</section>'


def stat(key, value, sub=None, tip=None):
    """One labelled number for a row of KPIs, with an optional sub-line and tooltip."""
    tip_attr = f' title="{esc(tip)}"' if tip else ""
    sub_html = f'<div class="d">{sub}</div>' if sub else ""
    return f'<div class="stat"{tip_attr}><div class="k">{esc(key)}</div><div class="v">{value}</div>{sub_html}</div>'


def kpis(stats):
    """A row of :func:`stat` tiles."""
    return f'<div class="kpis">{"".join(stats)}</div>'


def badge(text, kind=None, hue=None):
    """A small pill of text, optionally coloured by kind or hue."""
    cls = f"badge {kind}" if kind else "badge"
    style = f' style="--c:var(--{hue})"' if hue else ""
    return f'<span class="{cls}"{style}>{esc(text)}</span>'


def table(headers, rows, numeric=(), table_id=None):
    """``rows`` are lists of already-escaped HTML cells; ``numeric`` are column indices to right-align."""
    numeric = set(numeric)
    head = "".join(
        f'<th class="num">{esc(h)}</th>' if i in numeric else f"<th>{esc(h)}</th>" for i, h in enumerate(headers)
    )
    body = "".join(
        "<tr>"
        + "".join(f'<td class="num">{c}</td>' if i in numeric else f"<td>{c}</td>" for i, c in enumerate(r))
        + "</tr>"
        for r in rows
    )
    tid = f' id="{table_id}"' if table_id else ""
    return f'<div class="scrollwrap" style="max-height:520px"><table class="data"{tid}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def contributions(items, scale=None):
    """Signed contribution rows: ``items`` = [(label, value)], drawn as diverging microbars."""
    if not items:
        return '<p class="faint">No contributing features.</p>'
    scale = scale or max(abs(v) for _, v in items) or 1.0
    rows = []
    for label, v in items:
        w = min(50.0, 50.0 * abs(v) / scale)
        cls = "pos" if v > 0 else "neg"
        rows.append(
            f'<div class="lbl" title="{esc(label)}">{esc(label)}</div>'
            f'<div class="bar"><i class="{cls}" style="width:{w:.1f}%"></i></div>'
            f'<div class="v">{v:+.3f}</div>'
        )
    return f'<div class="contrib">{"".join(rows)}</div>'


def image(fig, href_prefix=""):
    """A figure drawn by :func:`xai4chem.report.figures.render`, embedded, with its caption and a PDF link."""
    pdf = f' · <a href="{esc(href_prefix + fig["pdf"])}">PDF</a>' if fig.get("pdf") else ""
    return (
        f'<figure><img src="{fig["data_uri"]}" alt="{esc(fig["title"])}">'
        f"<figcaption><b>{esc(fig['title'])}</b> · {esc(fig['caption'])}{pdf}</figcaption></figure>"
    )


def figures(figs, href_prefix=""):
    """A responsive grid of figures; ``None`` entries (a plot that declined) are left out."""
    figs = [f for f in figs if f is not None]
    return f'<div class="figgrid">{"".join(image(f, href_prefix) for f in figs)}</div>' if figs else ""
