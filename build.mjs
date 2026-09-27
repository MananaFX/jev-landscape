import { readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { marked } from 'marked';

const source = await readFile(new URL('./report.md', import.meta.url), 'utf8');
const displayMath = new Map([
  [String.raw`\mathbf{p} = \operatorname{softmax}([\operatorname{logit}(A),\operatorname{logit}(B),\operatorname{logit}(C)])`, `<div class="math-display"><math display="block" aria-label="候选概率向量等于 A、B、C 三个答案码的 logits 经 softmax"><mrow><mi mathvariant="bold">p</mi><mo>=</mo><mi mathvariant="normal">softmax</mi><mo>(</mo><mo>[</mo><mi mathvariant="normal">logit</mi><mo>(</mo><mi>A</mi><mo>)</mo><mo>,</mo><mi mathvariant="normal">logit</mi><mo>(</mo><mi>B</mi><mo>)</mo><mo>,</mo><mi mathvariant="normal">logit</mi><mo>(</mo><mi>C</mi><mo>)</mo><mo>]</mo><mo>)</mo></mrow></math></div>`],
]);
const sourceWithMath = source.replace(/^\$\$\n([^\n]+)\n\$\$$/gm, (_, formula) => {
  const rendered = displayMath.get(formula);
  if (!rendered) throw new Error(`Unrecognized display formula: ${formula}`);
  return rendered;
});
const rendered = marked.parse(sourceWithMath, { gfm: true });
const toc = [];
let section = 0;
const article = rendered.replace(/<h2>([\s\S]*?)<\/h2>/g, (_, inner) => {
  section += 1;
  const label = inner.replace(/<[^>]+>/g, '').replace(/&amp;/g, '&');
  toc.push(`<a href="#section-${section}">${label}</a>`);
  return `<h2 id="section-${section}">${inner}</h2>`;
});

const html = `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <meta name="description" content="Jev 风格开源决策模型的概率读取、动态分类头、训练数据、RLCD 实验与校准方法综述。">
  <title>Jev 开源实现综述</title>
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%2314253d'/%3E%3Cpath d='M18 19h28M18 32h19M18 45h28' stroke='%2387baff' stroke-width='6' stroke-linecap='round'/%3E%3Ccircle cx='45' cy='32' r='6' fill='%23fff'/%3E%3C/svg%3E">
  <style>
    :root { color-scheme: light; --ink:#17273d; --muted:#586a7f; --line:#dce4ee; --blue:#175bb0; --paper:#fff; --bg:#f4f7fb; }
    * { box-sizing:border-box; }
    html { scroll-behavior:smooth; }
    body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.72 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; }
    a { color:var(--blue); text-underline-offset:.2em; }
    a:hover { color:#0c3a78; }
    .top { background:#14253d; color:#f5f9ff; border-bottom:3px solid #7caff5; }
    .top-inner { max-width:1240px; margin:auto; padding:13px 28px; display:flex; align-items:center; justify-content:space-between; gap:20px; }
    .brand { font-size:.92rem; font-weight:750; letter-spacing:.025em; }
    .top a { color:#c8ddf9; font-size:.88rem; }
    .layout { max-width:1240px; margin:0 auto; padding:34px 28px 80px; display:grid; grid-template-columns:235px minmax(0,1fr); gap:38px; align-items:start; }
    .sidebar { position:sticky; top:22px; max-height:calc(100vh - 44px); overflow:auto; padding:18px 16px; border:1px solid var(--line); border-radius:12px; background:var(--paper); }
    .sidebar-label { margin:0 0 12px; font-size:.76rem; color:var(--muted); text-transform:uppercase; letter-spacing:.09em; font-weight:750; }
    .sidebar nav { display:grid; gap:3px; }
    .sidebar nav a { display:block; padding:6px 8px; color:#38516e; border-radius:7px; font-size:.89rem; text-decoration:none; line-height:1.4; }
    .sidebar nav a:hover { background:#eaf2fd; color:#124982; }
    .article { min-width:0; background:var(--paper); border:1px solid var(--line); border-radius:14px; padding:40px clamp(24px,5vw,64px) 64px; box-shadow:0 5px 26px rgba(22,44,73,.04); }
    h1,h2,h3 { color:#10233d; line-height:1.32; letter-spacing:-.025em; }
    h1 { margin:0 0 12px; font-size:clamp(2rem,3.7vw,2.8rem); }
    h2 { font-size:1.55rem; margin:2.8em 0 .8em; padding-top:.15em; border-bottom:1px solid var(--line); padding-bottom:.42em; scroll-margin-top:20px; }
    h3 { font-size:1.15rem; margin:2em 0 .5em; }
    p { margin:0 0 1.1em; }
    h1 + p { color:var(--muted); font-size:.95rem; margin-bottom:2em; }
    ul,ol { padding-left:1.5em; margin:0 0 1.25em; }
    li { margin:.33em 0; }
    blockquote { margin:1.5em 0; padding:10px 20px; border-left:4px solid #5a94dc; background:#edf4fd; border-radius:0 7px 7px 0; color:#304963; }
    blockquote p:last-child { margin-bottom:0; }
    table { display:block; width:100%; overflow-x:auto; border-collapse:collapse; white-space:normal; margin:1.4em 0 1.8em; font-size:.9rem; line-height:1.5; }
    th,td { border:1px solid var(--line); padding:10px 12px; min-width:110px; vertical-align:top; }
    th { background:#edf3fa; color:#233c59; font-weight:700; text-align:left; }
    tr:nth-child(even) td { background:#fafcff; }
    code { font:.9em/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; background:#eef3f8; padding:.13em .35em; border-radius:4px; }
    pre { padding:18px 20px; overflow:auto; border-radius:9px; background:#17283d; color:#ebf3fb; font-size:.88rem; line-height:1.55; }
    pre code { background:none; padding:0; color:inherit; }
    article img { display:block; max-width:100%; height:auto; margin:1.5em auto .4em; border-radius:8px; }
    article p:has(> img[src$="-readout.svg"]) { max-width:100%; overflow-x:auto; margin:1.35em 0 .4em; border-radius:8px; }
    article img[src="./figures/semif-readout.svg"] { width:1040px; max-width:none; margin:0; }
    article img[src="./figures/kev-readout.svg"] { width:1080px; max-width:none; margin:0; }
    .math-display { margin:1.35em 0; padding:10px 0; overflow-x:auto; text-align:center; }
    .math-display math { font-size:1.17rem; }
    strong { color:#122944; }
    .footer { max-width:1240px; padding:0 28px 35px; margin:auto; color:var(--muted); font-size:.85rem; }
    @media (max-width:850px) { .layout { grid-template-columns:1fr; gap:18px; padding:20px 14px 50px; } .sidebar { position:static; max-height:none; padding:13px; } .sidebar nav { display:flex; overflow:auto; white-space:nowrap; } .sidebar nav a { flex:none; } .article { padding:28px 20px 44px; } .top-inner { padding:12px 16px; } h2 { margin-top:2.2em; } }
    @media (max-width:520px) { .top-inner { align-items:flex-start; flex-direction:column; gap:2px; } .article { padding:24px 16px 38px; } table { font-size:.84rem; } th,td { padding:8px 9px; } }
  </style>
</head>
<body>
  <header class="top"><div class="top-inner"><span class="brand">Jev 开源实现综述</span><a href="./report.md">查看 Markdown 原文</a></div></header>
  <div class="layout">
    <aside class="sidebar" aria-label="文章目录"><p class="sidebar-label">目录</p><nav>${toc.join('')}</nav></aside>
    <main class="article"><article>${article}</article></main>
  </div>
  <footer class="footer">资料截至 2026-09-28。仓库会继续变化，请以链接中的代码、文档和实验记录为准。</footer>
</body>
</html>`;

await mkdir(new URL('./dist/', import.meta.url), { recursive: true });
await mkdir(new URL('./dist/figures/', import.meta.url), { recursive: true });
await Promise.all(['semif-readout.svg', 'kev-readout.svg'].map((name) =>
  copyFile(new URL(`./figures/${name}`, import.meta.url), new URL(`./dist/figures/${name}`, import.meta.url))));
await writeFile(new URL('./dist/index.html', import.meta.url), html, 'utf8');
await writeFile(new URL('./dist/report.md', import.meta.url), source, 'utf8');
console.log(`Rendered ${section} sections to dist/index.html`);
