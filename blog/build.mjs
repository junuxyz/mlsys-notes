import { readFile, writeFile, mkdir, rm, cp, stat } from 'node:fs/promises';
import path from 'node:path';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Marked } from 'marked';
import markedKatex from 'marked-katex-extension';
import katex from 'katex';
import hljs from 'highlight.js';

const root = path.dirname(fileURLToPath(import.meta.url));
const repo = path.dirname(root);
const out = path.join(root, 'dist');
const overrides = JSON.parse(await readFile(path.join(root, 'posts.json'), 'utf8'));
const site = JSON.parse(await readFile(path.join(root, 'site.json'), 'utf8'));
const posts = [];
const escape = text => String(text).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const github = site.repositoryUrl;
const sourceURL = `${github}/blob/${site.sourceBranch}`;
const readme = await readFile(path.join(repo, 'README.md'), 'utf8');
const readmeBody = readme.replace(/^# [^\n]+\n+/, '');
const readmeIntro = readmeBody.split(/^## /m)[0].trim();
const readmeTitleLinks = new Map([...readme.matchAll(/\[([^\]]+)\]\(([^)]+)\)/g)].map(([, title, source]) => [/^[a-z][a-z\d+.-]*:|^\/\//i.test(source) ? source : path.posix.normalize(source), title]));
const readmeSources = [...readmeTitleLinks.keys()].filter(source => !/^[a-z][a-z\d+.-]*:|^\/\//i.test(source) && source.endsWith('.md'));
const markdownBySource = new Map();
const configured = new Map(overrides.map(post => [post.source, post]));
const usedSlugs = new Set();
const titleReader = new Marked();
for (const source of new Set([...overrides.map(post => post.source), ...readmeSources])) {
  if (!readmeSources.includes(source) || configured.get(source)?.unpublished) continue;
  const file = path.resolve(repo, source);
  if (!file.startsWith(repo + path.sep) || !(await stat(file).catch(() => null))?.isFile()) continue;
  const markdown = await readFile(file, 'utf8');
  if (!markdown.trim()) continue;
  markdownBySource.set(source, markdown);
  const first = titleReader.lexer(markdown).find(token => token.type !== 'space');
  const title = configured.get(source)?.title || readmeTitleLinks.get(source) || (first?.type === 'heading' && first.depth === 1 ? first.text : path.basename(source, '.md'));
  const slug = configured.get(source)?.slug || source.replace(/\.md$/, '').replace(/[^\p{L}\p{N}]+/gu, '-').replace(/^-|-$/g, '').toLowerCase();
  if (usedSlugs.has(slug)) throw new Error(`Duplicate article URL: ${slug}`);
  usedSlugs.add(slug);
  const topic = configured.get(source)?.topic || (path.dirname(source) === 'notes' ? 'Notes' : path.dirname(source).split('/').at(-1));
  posts.push({ source, slug, title, topic });
}
const readmeRenderer = new Marked();
const renderReadme = markdown => readmeRenderer.parse(markdown).replace(/href="([^"#]+)"/g, (_, href) => {
  if (/^[a-z][a-z\d+.-]*:|^\/\//i.test(href)) return `href="${href}"`;
  const post = posts.find(p => p.source === href);
  return `href="${post ? `${post.slug}.html` : `${sourceURL}/${href}`}"`;
});
const introHTML = renderReadme(readmeIntro);
const aboutHTML = renderReadme(readmeBody);

await rm(out, { recursive: true, force: true });
await mkdir(path.join(out, 'assets'), { recursive: true });
await cp(path.join(root, 'theme'), path.join(out, 'theme'), { recursive: true });
await cp(path.join(root, 'THIRD_PARTY_NOTICES.md'), path.join(out, 'THIRD_PARTY_NOTICES.md'));
await cp(path.join(root, 'node_modules/katex/dist'), path.join(out, 'theme/katex'), { recursive: true });
await cp(path.join(root, 'node_modules/highlight.js/styles/github.css'), path.join(out, 'theme/highlight.css'));
const assetCopies = new Set();
const missing = new Set();
function localURL(value, source, image = false) {
  if (/^(?:[a-z][a-z\d+.-]*:|#|\/\/)/i.test(value)) return value;
  const [, file, suffix = ''] = value.match(/^([^#?]*)(.*)$/);
  const target = file.startsWith('/') ? path.resolve(repo, `.${decodeURI(file)}`) : path.resolve(repo, path.dirname(source), decodeURI(file));
  if (!target.startsWith(repo + path.sep)) return value;
  const relative = path.relative(repo, target).split(path.sep).join('/');
  const post = posts.find(p => p.source === relative);
  if (post) return `${post.slug}.html${suffix}`;
  if (image || /\.(png|jpe?g|gif|svg|webp|pdf|avif)$/i.test(file)) {
    assetCopies.add(relative);
    return `assets/${relative.split('/').map(encodeURIComponent).join('/')}${suffix}`;
  }
  return `${sourceURL}/${relative.split('/').map(encodeURIComponent).join('/')}${suffix}`;
}
function shell(title, description, body, active = '') {
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><link rel="icon" href="data:,"><title>${escape(title === site.title ? site.title : `${title} | ${site.title}`)}</title><meta name="description" content="${escape(description)}"><meta property="og:title" content="${escape(title)}"><meta property="og:description" content="${escape(description)}"><meta property="og:type" content="${active === 'article' ? 'article' : 'website'}"><link rel="stylesheet" href="theme/highlight.css"><link rel="stylesheet" href="theme/katex/katex.min.css"><link rel="stylesheet" href="theme/style.css"></head>
<body><a class="skip" href="#main">Skip to content</a><header class="site-header" role="banner"><div class="wrapper"><a class="site-title" rel="author" href="index.html">${escape(site.title)}</a><nav class="site-nav" aria-label="Main navigation"><input type="checkbox" id="nav-trigger" class="nav-trigger" aria-label="Toggle navigation"><label for="nav-trigger"><span class="menu-icon"><svg aria-hidden="true" viewBox="0 0 18 15" width="18" height="15"><path d="M0 0h18v3H0zm0 6h18v3H0zm0 6h18v3H0z"/></svg></span></label><div class="trigger"><a class="page-link" href="archive.html">Blog Archive</a><a class="page-link" href="about.html">About</a></div></nav></div></header><main id="main" class="page-content" aria-label="Content"><div class="wrapper">${body}</div></main></body></html>`;
}

for (const post of posts) {
  let markdown = markdownBySource.get(post.source);
  markdown = markdown.replace(/^# [^\n]+\n+/, '');
  post.minutes = Math.max(1, Math.ceil(markdown.split(/\s+/).length / 220));
  const headings = [];
  const ids = new Map();
  const marked = new Marked();
  const mathExtension = markedKatex({ throwOnError: false, nonStandard: true });
  // Let display math interrupt a paragraph, including inside blockquotes.
  mathExtension.extensions.find(extension => extension.name === 'blockKatex').start = src => src.match(/^\$\$[ \t]*\n/m)?.index;
  marked.use(mathExtension);
  marked.use({ extensions: [{
    name: 'wikiReference', level: 'inline',
    start: src => src.search(/!\[\[|\[\[#/),
    tokenizer(src) {
      const match = src.match(/^(!?)\[\[([^\]\n]+)\]\]/);
      if (!match || (!match[1] && !match[2].startsWith('#'))) return;
      return { type: 'wikiReference', raw: match[0], image: !!match[1], target: match[2] };
    },
    renderer(token) {
      const [target, alias] = token.target.split('|');
      if (!token.image) {
        const label = target.slice(1);
        const id = label.toLowerCase().replace(/[^\p{L}\p{N}\s-]/gu, '').trim().replace(/\s+/g, '-');
        return `<a href="#${id}">${escape(alias || label)}</a>`;
      }
      const file = existsSync(path.join(repo, target)) ? path.join(repo, target) : path.resolve(repo, path.dirname(post.source), target);
      const relative = path.relative(path.join(repo, path.dirname(post.source)), file).split(path.sep).join('/');
      return `<img src="${escape(relative)}" alt="${escape(target)}">`;
    }
  }] });
  marked.use({ renderer: {
    heading({ tokens, depth }) {
      const text = this.parser.parseInline(tokens);
      const plain = text.replace(/<[^>]+>/g, '').replace(/&(#x[0-9a-f]+|#\d+|amp|quot|apos|lt|gt);/gi, (_, entity) => entity[0] === '#' ? String.fromCodePoint(entity[1].toLowerCase() === 'x' ? parseInt(entity.slice(2), 16) : Number(entity.slice(1))) : ({amp:'&',quot:'"',apos:"'",lt:'<',gt:'>'})[entity.toLowerCase()]);
      const base = plain.toLowerCase().replace(/[^\p{L}\p{N}\s-]/gu, '').trim().replace(/\s+/g, '-') || 'section';
      const count = ids.get(base) || 0;
      ids.set(base, count + 1);
      const id = base + (count ? `-${count}` : '');
      if (depth <= 3) headings.push({ text: plain, id, depth });
      return `<h${depth} id="${id}">${text}<a class="heading-anchor" href="#${id}" aria-label="Link to ${escape(plain)}">#</a></h${depth}>`;
    },
    blockquote({ text, tokens }) {
      const marker = text.match(/^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\](?:[ \t]+([^\n]*))?[ \t]*(?:\n|$)/i)
        || text.match(/^\*\*(Note|Tip|Important|Warning|Caution)(?::[ \t]*([^\n]*?))?\*\*[ \t]*(?:\n|$)/i);
      if (!marker) return `<blockquote>${this.parser.parse(tokens)}</blockquote>`;
      const label = marker[1][0].toUpperCase() + marker[1].slice(1).toLowerCase();
      const title = marker[2] ? `<span class="note-title">${marked.parseInline(marker[2])}</span>` : '';
      const body = marked.parse(text.slice(marker[0].length).trim());
      return `<aside class="note-callout" role="note" aria-label="${label}"><div class="note-header"><span class="note-label">${label}</span>${title}</div><div class="note-body">${body}</div></aside>`;
    },
    code({ text, lang }) {
      if (lang === 'math') return `<div class="math-block">${katex.renderToString(text, { displayMode: true, throwOnError: false })}</div>`;
      const language = (lang || '').split(/\s/)[0];
      const highlighted = hljs.getLanguage(language) ? hljs.highlight(text, { language }).value : escape(text);
      return `<pre><code class="hljs language-${escape(language)}">${highlighted}</code></pre>`;
    }
  }});
  let html = marked.parse(markdown);
  html = html.replace(/\b(src|href)="([^"]+)"/g, (_, attr, url) => `${attr}="${escape(localURL(url.replace(/&amp;/g, '&'), post.source, attr === 'src'))}"`);
  html = html.replace(/<img\b[^>]*>/g, tag => {
    const src = tag.match(/\bsrc="([^"]+)"/)?.[1];
    if (!src?.startsWith('assets/')) return tag;
    const asset = decodeURI(src.slice('assets/'.length));
    if (existsSync(path.join(repo, asset))) return tag;
    missing.add(asset);
    const alt = tag.match(/\balt="([^"]*)"/)?.[1] || escape(path.basename(asset));
    return `<span class="missing-image">${alt} <small>(image missing from source)</small></span>`;
  });
  html = html.replace(/<table>([\s\S]*?)<\/table>/g, '<div class="table-scroll" tabindex="0" role="region" aria-label="Scrollable table"><table>$1</table></div>');
  const htmlIds = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]));
  html = html.replace(/href="#([^"]+)"/g, (attribute, id) => {
    if (htmlIds.has(id)) return attribute;
    const matches = [...htmlIds].filter(candidate => candidate.startsWith(`${id}-`));
    return matches.length === 1 ? `href="#${matches[0]}"` : attribute;
  });
  html = html.replace(/<a\b([^>]*?)href="(\.\.\/[^"#]+)"([^>]*)>/g, (_, before, href, after) => `<a${before}aria-disabled="true" title="Linked file is outside this repository"${after}>`);
  const next = posts[(posts.indexOf(post) + 1) % posts.length];
  const tocTree = [];
  const tocStack = [{ depth: 0, children: tocTree }];
  for (const heading of headings) {
    while (tocStack.length > 1 && tocStack.at(-1).depth >= heading.depth) tocStack.pop();
    const node = { ...heading, children: [] };
    tocStack.at(-1).children.push(node);
    tocStack.push(node);
  }
  const tocList = nodes => `<ul>${nodes.map(h => `<li><a href="#${h.id}">${escape(h.text)}</a>${h.children.length ? tocList(h.children) : ''}</li>`).join('')}</ul>`;
  const toc = `<h2 id="table-of-contents">Table of contents</h2>${tocList(tocTree)}`;
  const firstHeading = html.search(/<h[1-6] /);
  html = firstHeading >= 0 ? html.slice(0, firstHeading) + toc + html.slice(firstHeading) : html;
  const previous = posts[(posts.indexOf(post) + posts.length - 1) % posts.length];
  const body = `<article class="post h-entry"><header class="post-header"><h1 class="post-title p-name">${escape(post.title)}</h1></header><div class="post-content e-content">${html}</div><hr><div class="post_navi"><a class="post_navi-item nav_prev" href="${previous.slug}.html"><div class="post_navi-arrow">❮</div><div class="post_navi-label">Previous post</div><div>${escape(previous.title)}</div></a><a class="post_navi-item nav_next" href="${next.slug}.html"><div class="post_navi-arrow">❯</div><div class="post_navi-label">Next post</div><div>${escape(next.title)}</div></a></div></article>`;
  await writeFile(path.join(out, `${post.slug}.html`), shell(post.title, post.title, body, 'article'));
}
const topics = [...new Set(posts.map(p => p.topic))];
const entries = posts.map(p => `<div class="post-item"><div class="post-date">${escape(p.topic)}</div><h3 class="post-title"><a class="post-link" href="${p.slug}.html">${escape(p.title)}</a></h3><div class="post-meta">${p.minutes} min read</div></div>`).join('');
await writeFile(path.join(out, 'index.html'), shell(site.title, readmeIntro.replace(/\n+/g, ' '), `<div class="home"><div class="profile-container"><div class="profile-content">${introHTML}</div></div><div class="home-sections"><h2 class="post-list-heading">Posts</h2><div class="post-list">${entries}</div></div></div>`));
await writeFile(path.join(out, 'archive.html'), shell('Blog Archive', 'Articles grouped by topic.', `<article class="post"><header class="post-header"><h1 class="post-title">Blog Archive</h1></header><div class="post-content">${topics.map(topic => `<section><h3>${escape(topic)}</h3><ul>${posts.filter(p => p.topic === topic).map(p => `<li><a href="${p.slug}.html">${escape(p.title)}</a></li>`).join('')}</ul></section>`).join('')}</div></article>`));
await writeFile(path.join(out, 'about.html'), shell('About', readmeIntro.replace(/\n+/g, ' '), `<article class="post"><header class="post-header"><h1 class="post-title">${escape(site.title)}</h1></header><div class="post-content">${aboutHTML}</div></article>`));
for (const asset of assetCopies) {
  const source = path.join(repo, asset);
  if (!(await stat(source).catch(() => null))?.isFile()) { missing.add(asset); continue; }
  const destination = path.join(out, 'assets', asset);
  await mkdir(path.dirname(destination), { recursive: true });
  await cp(source, destination);
}
if (missing.size) console.warn(`Missing source images shown as labels:\n${[...missing].join('\n')}`);
await writeFile(path.join(out, '.nojekyll'), '');
console.log(`Built ${posts.length} notes and ${assetCopies.size} assets in ${out}`);
