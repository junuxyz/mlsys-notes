# mlsys-notes blog

A static blog for the published notes in this repository, using a Minima-based layout. Theme attribution and license terms are in `THIRD_PARTY_NOTICES.md`.

## Run locally

Requires Node.js 22 or newer.

```sh
npm --prefix blog ci
npm --prefix blog run dev
```

Open http://127.0.0.1:4321. After editing a source file, run `npm --prefix blog run build` and refresh the browser. Set `PORT` to choose another local port.

## Published content

The build includes nonempty Markdown files linked from the repository README. Other notes and drafts are excluded from the generated pages and assets. `posts.json` sets optional article URLs, titles, topics, and ordering. An entry with `"unpublished": true` is excluded even if the README links to it.

The homepage introduction and About page come from the repository README. `site.json` contains the site title, repository URL, and source branch. Article links, styles, and images use relative URLs so the same build works at `/mlsys-notes/`.

Markdown supports math, code highlighting, tables, heading links, Obsidian image embeds, and note callouts such as `> [!NOTE]`. Referenced images and PDFs are copied into the output. Missing images produce a build warning. Embedded HTML is retained, so review article sources before publishing.

## Publish

```sh
npm --prefix blog run build
```

The output is `blog/dist/`. The GitHub Actions workflow builds pull requests and deploys changes on `main` to GitHub Pages.

In the repository's **Settings > Pages**, select **GitHub Actions** as the publishing source. Leave the repository's custom domain empty. The account site already uses `junupark.xyz`, so the project site inherits it at https://junupark.xyz/mlsys-notes/. Do not add a `CNAME` file to this project.

Only `blog/dist/` is uploaded for hosting. Dependencies and generated files are ignored by Git.
