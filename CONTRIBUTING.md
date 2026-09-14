# Contributing

Thanks for your interest in Predspot! Issues and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/adaj/predspot.git
cd predspot
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,contour,examples]"
```

## Checks

```bash
ruff check src tests          # lint
ruff format src tests         # format
pytest                        # tests (~10 s)
PREDSPOT_NETWORK_TESTS=1 pytest tests/test_load_study_area.py  # also query OpenStreetMap
```

CI runs the same checks on Python 3.10 to 3.13 for every pull request.

## Documentation

The site is built with [MkDocs](https://www.mkdocs.org/) and
[Material](https://squidfunk.github.io/mkdocs-material/); API pages come from the
docstrings via mkdocstrings.

```bash
pip install -e ".[docs]"
mkdocs serve            # live preview at http://127.0.0.1:8000
mkdocs build --strict   # what CI runs
```

Pushing to `master` deploys the site to GitHub Pages automatically. The home
page is generated from `README.md` (see `docs/hooks/readme.py`), and the
example notebook is rendered from `examples/natal.ipynb` with its stored
outputs. After editing the notebook, re-execute it so the outputs stay in sync
(it needs network access for `get_city_shape`):

```bash
pip install -e ".[examples]"
jupyter nbconvert --to notebook --execute --inplace examples/natal.ipynb
```

## Releasing

1. Bump `__version__` in `src/predspot/__init__.py` and update `CHANGELOG.md`.
2. Merge to `master`, then tag and push: `git tag v0.2.0 && git push origin v0.2.0`.
3. The `Publish to PyPI` workflow builds the distribution and uploads it via
   PyPI trusted publishing.
