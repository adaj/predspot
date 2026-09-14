# Contributing

Thanks for your interest in Predspot! Issues and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/adaj/predspot.git
cd predspot
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,osm,contour]"
```

## Checks

```bash
ruff check src tests          # lint
ruff format src tests         # format
pytest                        # tests (~10 s)
PREDSPOT_NETWORK_TESTS=1 pytest tests/test_load_study_area.py  # also query OpenStreetMap
```

CI runs the same checks on Python 3.10 to 3.13 for every pull request.

## Releasing

1. Bump `__version__` in `src/predspot/__init__.py` and update `CHANGELOG.md`.
2. Merge to `master`, then tag and push: `git tag v0.2.0 && git push origin v0.2.0`.
3. The `Publish to PyPI` workflow builds the distribution and uploads it via
   PyPI trusted publishing.
