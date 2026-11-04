#!/usr/bin/env python3
"""Render blog/index.md to docs/blog.html: a self-contained page (inline CSS, no CDN, no JavaScript) for GitHub Pages.
Figure paths are rewritten from ../docs/figures/ to figures/ so the page works from the docs/ root.

  /Users/scasella/Downloads/qwen-3/toolpatch/.venv/bin/python scripts/build_blog_html.py"""
import re, html
from pathlib import Path
import markdown
ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / 'blog/index.md'; OUT = ROOT / 'docs/blog.html'
md = SRC.read_text().replace('../docs/figures/', 'figures/')
title = re.search(r'^# (.+)$', md, re.M).group(1)
body = markdown.markdown(md, extensions=['tables', 'fenced_code', 'toc', 'attr_list'], output_format='html5')
body = re.sub(r'<p><img ([^>]*)></p>', lambda m: f'<figure><img {m.group(1)}></figure>', body)  # figure wrapper
body = re.sub(r'<p><strong>(Figure \d+\.)</strong>', r'<p class="caption"><strong>\1</strong>', body)
CSS = """
:root{--ink:#111827;--muted:#4b5563;--rule:#e5e7eb;--accent:#0f766e;--bg:#ffffff;--code:#f3f4f6;--quote:#f8fafc}
@media (prefers-color-scheme:dark){:root{--ink:#e5e7eb;--muted:#9ca3af;--rule:#374151;--accent:#5eead4;--bg:#0b1220;--code:#111827;--quote:#0f172a} img{background:#fff;border-radius:6px;padding:6px}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
main{max-width:860px;margin:0 auto;padding:40px 16px 80px}
h1{font-size:2.1rem;line-height:1.2;margin:0 0 .4em;letter-spacing:-.01em}
h1+p>em{font-size:1.15rem;color:var(--muted);display:block;margin-bottom:1.2em;font-style:italic}
h2{font-size:1.45rem;margin:2.2em 0 .6em;border-top:1px solid var(--rule);padding-top:1.2em}
p,li{max-width:72ch}
a{color:var(--accent)}
code{font:0.9em/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--code);padding:.1em .3em;border-radius:4px}
pre{background:var(--code);padding:14px 16px;border-radius:8px;overflow-x:auto;font-size:.82rem;line-height:1.45}
pre code{background:none;padding:0;font-size:inherit}
table{border-collapse:collapse;font-size:.88rem;margin:1.2em 0;display:block;overflow-x:auto;max-width:100%}
th,td{border-bottom:1px solid var(--rule);padding:.45em .7em;text-align:left;vertical-align:top}
th{background:var(--code)}
figure{margin:2em 0 .6em}figure img{width:100%;height:auto}
p.caption{font-size:.92rem;color:var(--muted);max-width:none;margin-top:0;border-left:3px solid var(--accent);padding-left:12px}
blockquote{margin:1em 0;padding:.2em 1em;background:var(--quote);border-left:3px solid var(--rule)}
footer{margin-top:4em;color:var(--muted);font-size:.85rem;border-top:1px solid var(--rule);padding-top:1em}
"""
page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="Two closed-form weight edits of Qwen3.5-0.8B, built without training: a two-neuron layer-23 readout that repairs trigger-happy tool calling (g23v3), and a four-neuron, two-layer circuit that routes a stranded ambiguity feature from the user's clause to the decision token through an unchanged attention head (g23v4). 12,281 weights, 0.0016% of the model.">
<style>{CSS}</style>
</head>
<body>
<main>
{body}
<footer>Built from <code>blog/index.md</code> by <code>scripts/build_blog_html.py</code>. Figures: <code>scripts/make_figures.py</code>, computed from the campaign records under <code>experiments/</code>. Apache-2.0.</footer>
</main>
</body>
</html>
"""
OUT.write_text(page); print('wrote', OUT, len(page), 'bytes')
