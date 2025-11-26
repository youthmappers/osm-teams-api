OSM Teams API
===

Python client and YouthMappers utilities that sit on top of the [OSM Teams API](https://mapping.team/docs/api).

## Getting started

This project now uses [uv](https://docs.astral.sh/uv/) for dependency management. To create an isolated environment with runtime dependencies (add `--group dev` if you also want the tooling for tests):

```
uv sync
```

Run the YouthMappers CLI after syncing (you will need the `google` optional dependencies installed or specify `uv sync --extra google`):

```
uv run youthmappers --help
```

## Tests

```
uv run --group dev pytest
```

## Linting & formatting

Ruff and Black live in the `dev` dependency group. Run them locally with:

```
uv run --group dev ruff check .
uv run --group dev black .
```

## Status
![Workflow Status](https://github.com/youthmappers/osm-teams-api/actions/workflows/run_youthmappers.yml/badge.svg)
