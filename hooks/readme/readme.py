"""MkDocs hook: the documentation home page is the repository README.

The README is written for GitHub (paths relative to the repository root); this
hook rewrites those paths so the same content renders on the site.
"""

import re
from pathlib import Path

README = Path(__file__).resolve().parents[2] / "README.md"

REPO_BLOB = "https://github.com/adaj/predspot/blob/master/"
LINKS = {
    REPO_BLOB + "CONTRIBUTING.md": "contributing.md",
    REPO_BLOB + "CHANGELOG.md": "changelog.md",
    REPO_BLOB + "examples/natal.ipynb": "examples/natal.ipynb",
    "https://adaj.github.io/predspot/": "index.md",
}


def on_page_markdown(markdown, page, config, files):
    if page.file.src_path != "index.md":
        return markdown
    text = README.read_text(encoding="utf-8")
    text = text.replace('src="docs/assets/', 'src="assets/')
    text = text.replace("](docs/assets/", "](assets/")
    for url, target in LINKS.items():
        text = text.replace(f"]({url})", f"]({target})")
    # Keep the page's own front matter (title etc.), if any.
    front = re.match(r"^---\n.*?\n---\n", markdown, flags=re.S)
    return (front.group(0) if front else "") + text
