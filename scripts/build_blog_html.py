#!/usr/bin/env python3
"""Render blog/index.md to docs/blog.html for GitHub Pages.
The page shares docs/styles.css (design tokens, top bar, footer, headline tiles) and docs/theme.js (the 20-line colour-theme
toggle) with the explorer, loads its fonts from Google Fonts, and carries its own article CSS inline. No other JavaScript, no
frameworks. Figure paths are rewritten from ../docs/figures/ to figures/ so the page works from the docs/ root.

  /Users/scasella/Downloads/qwen-3/toolpatch/.venv/bin/python scripts/build_blog_html.py"""
import re, html
from pathlib import Path
import markdown
ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / 'blog/index.md'; OUT = ROOT / 'docs/blog.html'
md_src = SRC.read_text().replace('../docs/figures/', 'figures/')

# ---- masthead pieces: the H1, the italic deck and the byline paragraph come out of the body and go into the masthead
title = re.search(r'^# (.+)$', md_src, re.M).group(1)
deck_m = re.search(r'^\*(.+)\*$', md_src, re.M); deck = deck_m.group(1)
byline_m = re.search(r'^Research agent: .+$', md_src, re.M); byline_md = byline_m.group(0)
body_md = md_src.replace(f'# {title}\n', '', 1).replace(deck_m.group(0) + '\n', '', 1).replace(byline_md + '\n', '', 1)
words = len(re.findall(r'\w+', body_md)); minutes = max(1, round(words / 230))

conv = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc', 'attr_list'], output_format='html5')
body = conv.convert(body_md)
byline = markdown.markdown(byline_md, output_format='html5')  # one <p>
toc = ''.join(f'<li><a href="#{t["id"]}">{html.escape(t["name"])}</a></li>' for t in conv.toc_tokens if t['level'] == 2)

# figures: <p><img></p> followed by the "**Figure N.** caption" paragraph -> one <figure> with a <figcaption>
body = re.sub(r'<p><img ([^>]*)></p>\s*<p><strong>(Figure \d+\.)</strong>(.*?)</p>',
              lambda m: f'<figure class="fig wide"><img {m.group(1)} loading="lazy"><figcaption><b>{m.group(2)}</b>{m.group(3)}</figcaption></figure>', body, flags=re.S)
body = re.sub(r'<p><img ([^>]*)></p>', lambda m: f'<figure class="fig"><img {m.group(1)} loading="lazy"></figure>', body)
body = re.sub(r'<table>', '<div class="tablewrap"><table>', body).replace('</table>', '</table></div>')
body = re.sub(r'<p>(?=(?:(?!</p>).)*href="index\.html")', '<p class="try">', body)  # the "see it rather than read it" pointer
# deck: mark the agent name
deck_html = markdown.markdown(deck, output_format='html5')[3:-4]

CSS = """
.post{padding:0 0 40px}
.masthead{position:relative;padding:38px 0 26px;border-bottom:1px solid var(--rule);background:radial-gradient(var(--dot) 1px,transparent 1.2px) 0 0/18px 18px}
.masthead::before{content:"";position:absolute;inset:0;background:linear-gradient(180deg,transparent 0%,var(--bg) 92%);pointer-events:none}
.masthead .wrap{position:relative;max-width:1100px}
.masthead h1{font-family:var(--display);font-weight:600;font-size:clamp(1.8rem,3.4vw,2.9rem);line-height:1.1;letter-spacing:-.02em;font-variation-settings:"opsz" 72;max-width:26ch;margin:0}
.masthead .deck{font-family:var(--serif);font-style:italic;font-size:1.18rem;line-height:1.5;max-width:62ch;color:var(--ink-2);margin:16px 0 0}
.byline{margin:18px 0 0;font-size:.9rem;color:var(--muted);max-width:90ch;line-height:1.6}
.byline a{color:var(--accent-ink)}
.readmeta{display:flex;flex-wrap:wrap;gap:6px 18px;margin:16px 0 0;font-family:var(--mono);font-size:.76rem;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
.masthead .headline{margin-top:22px;max-width:1060px}
.post-body{display:grid;grid-template-columns:230px minmax(0,740px);gap:56px;justify-content:center;padding:34px 20px 0;max-width:1100px;margin:0 auto}
.toc{position:sticky;top:70px;align-self:start;font-size:.86em}
.toc h2{font-family:var(--sans);font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:500;margin:0 0 8px}
.toc ol{list-style:none;margin:0;padding:0;display:grid;gap:2px}
.toc a{display:block;color:var(--ink-2);text-decoration:none;padding:4px 8px;border-left:2px solid var(--rule);line-height:1.35}
.toc a:hover{color:var(--ink);border-color:var(--ink)}
.toc .back{margin-top:16px;display:block;color:var(--accent-ink);text-decoration:none;padding:4px 8px}
.toc details summary{display:none}
.prose{font-family:var(--serif);font-size:1.1rem;line-height:1.66;color:var(--ink)}
.prose>p,.prose>ul,.prose>ol{max-width:68ch}
.prose p{margin:0 0 1.15em}
.prose ul,.prose ol{padding-left:1.3em;margin:0 0 1.15em}
.prose li{margin:.35em 0}
.prose li::marker{color:var(--muted)}
.prose h2{font-family:var(--display);font-size:1.75rem;line-height:1.15;letter-spacing:-.015em;font-variation-settings:"opsz" 36;margin:2.4em 0 .7em;padding-top:1.1em;border-top:1px solid var(--rule);font-weight:600}
.prose h2:first-child{margin-top:0;padding-top:0;border-top:none}
.prose h3{font-family:var(--sans);font-size:1.02rem;font-weight:600;letter-spacing:0;margin:1.9em 0 .55em;color:var(--ink)}
.prose strong{font-weight:600}
.prose em{font-style:italic}
.prose a{color:var(--accent-ink);text-decoration-thickness:1px;text-underline-offset:.16em}
.prose a:hover{color:var(--ink)}
.prose code{font-size:.84em;background:var(--panel-2);border:1px solid var(--rule);padding:.08em .35em;border-radius:5px;color:var(--ink);font-family:var(--mono)}
.prose pre{background:var(--panel);border:1px solid var(--rule);padding:14px 16px;border-radius:10px;overflow-x:auto;font-size:.82rem;line-height:1.5;margin:0 0 1.3em;font-family:var(--mono);box-shadow:var(--shadow)}
.prose pre code{background:none;border:none;padding:0;font-size:inherit}
.prose blockquote{margin:1.2em 0;padding:.4em 1.1em;background:var(--panel-2);border-left:3px solid var(--rule-2);border-radius:0 8px 8px 0;color:var(--ink-2)}
.tablewrap{overflow-x:auto;margin:1.3em 0;border:1px solid var(--rule);border-radius:10px;background:var(--panel);box-shadow:var(--shadow)}
.prose table{border-collapse:collapse;font-family:var(--sans);font-size:.86rem;width:100%;min-width:520px}
.prose th,.prose td{border-bottom:1px solid var(--rule);padding:.55em .8em;text-align:left;vertical-align:top}
.prose th{background:var(--panel-2);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.prose tr:last-child td{border-bottom:none}
.prose td code,.prose th code{font-size:.9em}
.fig{margin:1.8em 0 1.8em;width:100%}
.fig img{width:100%;height:auto;display:block;background:#fcfcfb;border:1px solid var(--rule);border-radius:10px;padding:8px}
.fig figcaption{font-family:var(--sans);font-size:.86rem;color:var(--muted);line-height:1.5;margin-top:10px;padding-left:12px;border-left:3px solid var(--accent);max-width:none}
.fig figcaption b{color:var(--ink);font-weight:600}
.prose .try{background:var(--accent-soft);border:1px solid color-mix(in srgb,var(--accent) 35%,var(--rule));border-radius:12px;padding:14px 18px;font-family:var(--sans);font-size:.98rem;line-height:1.55;max-width:none}
.prose .try::before{content:"Try it";display:block;font-family:var(--mono);font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--accent-ink);margin-bottom:4px}
@media (min-width:1101px){.fig.wide{width:calc(100% + 120px);margin-left:-60px}}
@media (max-width:1100px){
  .post-body{grid-template-columns:minmax(0,1fr);gap:0;max-width:780px;padding-top:24px}
  .toc{position:static;margin-bottom:14px}.toc h2{display:none}.toc ol{display:none}.toc details summary{display:list-item;cursor:pointer;color:var(--accent-ink);font-family:var(--sans);font-size:.9em}.toc details[open] ol{display:grid;margin-top:8px}.toc .back{display:none}
}
@media (max-width:640px){
  .masthead{padding:24px 0 20px}.masthead .deck{font-size:1.05rem}.prose{font-size:1.02rem}.prose h2{font-size:1.45rem}
  .prose pre{font-size:.76rem;padding:12px}
}
"""
page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="Two closed-form weight edits of Qwen3.5-0.8B, built without training: a two-neuron layer-23 readout that repairs trigger-happy tool calling (g23v3), and a four-neuron, two-layer circuit that routes a stranded ambiguity feature from the user's clause to the decision token through an unchanged attention head (g23v4). 12,281 weights, 0.0016% of the model.">
<meta name="theme-color" content="#f4f1ea" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#131412" media="(prefers-color-scheme: dark)">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;1,9..144,400;1,9..144,500&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<link rel="stylesheet" href="styles.css?v=20261104b">
<script src="theme.js?v=20261104b"></script>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Cpath d='M5 21C10 7 22 7 27 21' fill='none' stroke='%23888' stroke-width='1.5'/%3E%3Ccircle cx='5' cy='21' r='4' fill='%236250d6'/%3E%3Ccircle cx='12.3' cy='21' r='4' fill='%23eb6834'/%3E%3Ccircle cx='19.7' cy='21' r='4' fill='%23d6409f'/%3E%3Ccircle cx='27' cy='21' r='4' fill='%231baf7a'/%3E%3C/svg%3E">
<style>{CSS}</style>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="topbar">
  <div class="wrap navbar">
    <a class="brand" href="./" aria-label="qwen3.5-toolpatch home">
      <svg class="glyph" viewBox="0 0 34 20" aria-hidden="true"><path d="M4 14 C 10 1, 24 1, 30 14" fill="none" stroke="currentColor" stroke-width="1.2" opacity=".45"/><circle cx="4" cy="14" r="3.3" fill="var(--n-j0)"/><circle cx="12.7" cy="14" r="3.3" fill="var(--n-j1)"/><circle cx="21.3" cy="14" r="3.3" fill="var(--n-j2)"/><circle cx="30" cy="14" r="3.3" fill="var(--n-j3)"/></svg>
      <span class="name">qwen3.5-toolpatch</span>
    </a>
    <nav class="sitenav" aria-label="Site">
      <a href="./">Explorer</a>
      <a class="current" href="blog.html" aria-current="page">Blog post</a>
      <a class="ext" href="https://github.com/scasella/qwen3.5-toolpatch">GitHub</a>
      <a class="ext" href="https://huggingface.co/scasella91/Qwen3.5-0.8B-toolpatch-g23v4">Hugging Face</a>
    </nav>
    <button id="theme-toggle" class="iconbtn" type="button" aria-label="Theme">
      <svg class="auto" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><circle cx="12" cy="12" r="8.5"/><path d="M12 3.5v17A8.5 8.5 0 0 0 12 3.5z" fill="currentColor" stroke="none"/></svg>
      <svg class="sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2.5v2.5M12 19v2.5M2.5 12H5M19 12h2.5M5.3 5.3l1.8 1.8M16.9 16.9l1.8 1.8M5.3 18.7l1.8-1.8M16.9 7.1l1.8-1.8"/></svg>
      <svg class="moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" aria-hidden="true"><path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z"/></svg>
    </button>
  </div>
</header>
<main id="main">
<article class="post">
  <header class="masthead">
    <div class="wrap">
      <p class="kicker">Weight-space programming · Qwen3.5-0.8B · research agent Mythos 5.1</p>
      <h1>{html.escape(title)}</h1>
      <p class="deck">{deck_html}</p>
      <div class="byline">{byline[3:-4]}</div>
      <p class="readmeta"><span>{minutes} min read</span><span>6 sections · 4 figures</span><span>November 2026</span></p>
      <!-- Headline numbers: the campaigns' final tables (experiments/*/analysis, §1 below). -->
      <ul class="headline" aria-label="Headline results">
        <li class="stat base reveal"><span class="k">Base model</span><span class="v">35.4<small>%</small></span><span class="d">68 of 192 benchmark tool items · calls on everything</span></li>
        <li class="stat g23v3 reveal"><span class="k">g23v3 · 2 neurons</span><span class="v">80.7<small>%</small></span><span class="d">155 of 192 · NO_TOOL and explicit-missing repaired, 0/30 ambiguous</span></li>
        <li class="stat g23v4 reveal"><span class="k">g23v4 · 4 neurons</span><span class="v">96.4<small>%</small></span><span class="d">185 of 192 · +117 wrong→right, 0 right→wrong, 30/30 ambiguous</span></li>
        <li class="stat weights reveal"><span class="k">Unseen phrasings</span><span class="v">0<span class="arrow">→</span>9<span class="arrow">→</span>27</span><span class="d">of 32 stress-suite ambiguous items; all 32 resolved calls token-identical</span></li>
      </ul>
    </div>
  </header>
  <div class="post-body">
    <aside class="toc" aria-label="Contents">
      <h2>Contents</h2>
      <details open><summary>Contents</summary><ol>{toc}</ol></details>
      <a class="back" href="./">← Open the interactive explorer</a>
    </aside>
    <div class="prose">
{body}
    </div>
  </div>
</article>
</main>
<footer class="foot">
  <div class="wrap">
    <p>Built from <code>blog/index.md</code> by <code>scripts/build_blog_html.py</code>. Figures: <code>scripts/make_figures.py</code>, computed from the campaign records under <code>experiments/</code>. Code, data, ledgers and the explorer: <a href="https://github.com/scasella/qwen3.5-toolpatch">github.com/scasella/qwen3.5-toolpatch</a>. Apache-2.0.</p>
    <p class="credit">Research agent: Mythos 5.1 · Apache-2.0</p>
  </div>
</footer>
</body>
</html>
"""
OUT.write_text(page); print('wrote', OUT, len(page), 'bytes;', words, 'words,', minutes, 'min;', len(conv.toc_tokens), 'top-level headings')
