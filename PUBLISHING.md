# Publishing Catalyst to PyPI

Catalyst is published on PyPI under the distribution name `catalyst-gnn` while retaining the Python import package name `catalyst`.

Users install Catalyst with:

```bash
python -m pip install catalyst-gnn
```

Python imports remain unchanged:

```python
from catalyst.ml.gnn import GNNTask, build_task_model
```

## One-time PyPI setup

1. Create or sign in to a PyPI account.
2. Open the PyPI account Publishing page and add a pending GitHub Trusted Publisher.
3. Use these values:

   * PyPI project name: `catalyst-gnn`
   * GitHub owner: `Materials-Informatics-Laboratory`
   * Repository: `Catalyst`
   * Workflow: `publish-to-pypi.yml`
   * Environment: `pypi`

4. In the GitHub repository, create an environment named `pypi` under Settings > Environments.
5. Configure required reviewers/manual approval for the `pypi` environment before production publishing.

The PyPI project is created automatically the first time the pending Trusted Publisher successfully publishes.

## Release process

1. Update `src/catalyst/_version.py` to the new release version.
2. Commit and push the release changes.
3. Create a GitHub Release with a matching tag, for example `v2.2.0`.
4. Publish the GitHub Release.
5. The `Publish Catalyst to PyPI` workflow will build the wheel and source distribution, validate them with Twine, and publish them to PyPI through Trusted Publishing.

Do not reuse an already-published version number. PyPI release files are immutable.

## Local package validation

Install the development tools:

```bash
python -m pip install -e ".[dev]"
```

Build and validate the distributions:

```bash
python -m build
python -m twine check dist/*
```

Test the wheel itself in a clean environment before a release when practical.
