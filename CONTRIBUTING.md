# Contributing to Catalyst

Thank you for your interest in contributing to Catalyst. Contributions that improve the software, documentation, tests, examples, or usability are welcome.

Catalyst is developed as an open-source research software project. Contributions should preserve the modular design of the package and, where applicable, include tests or examples that demonstrate the intended behavior.

## Reporting bugs

If you encounter a bug, please open an issue in the Catalyst GitHub repository:

https://github.com/Materials-Informatics-Laboratory/Catalyst/issues

When reporting a bug, please include as much of the following information as possible:

- Catalyst version
- Python version
- operating system
- PyTorch and PyTorch Geometric versions
- CPU or GPU environment, if relevant
- a minimal example that reproduces the problem
- the complete error message or traceback
- the behavior you expected

You can check the installed Catalyst version with:

```bash
python -c "import catalyst; print(catalyst.__version__)"
```

## Requesting features

Feature requests are also welcome through GitHub Issues.

Please describe:

- the scientific or software problem you are trying to solve,
- the behavior or API you would like Catalyst to provide,
- why the proposed feature belongs in the general Catalyst framework rather than a project-specific workflow, and
- any relevant examples, references, or implementation ideas.

For substantial changes to the public API or software architecture, opening an issue before beginning implementation is encouraged so that the proposed design can be discussed.

## Development installation

To work on Catalyst itself, clone the repository and install it in editable mode with the development dependencies:

```bash
git clone https://github.com/Materials-Informatics-Laboratory/Catalyst.git
cd Catalyst
python -m pip install -e ".[dev]"
```

An editable installation means that changes made to the local source tree are immediately available to the installed package.

Catalyst supports Python 3.10 and newer. Compatibility with a particular Python version may also depend on the supported versions of PyTorch and PyTorch Geometric.

If you require a particular CUDA-enabled PyTorch build, install the appropriate PyTorch build for your system before installing Catalyst.

## Running the test suite

Before submitting a pull request, run the complete test suite from the repository root:

```bash
python -m pytest -q
```

The test suite includes checks for task contracts, graph construction, periodic neighbors, equivariance and invariance, checkpoint/restart behavior, parameter validation, smoke workflows, and other release regressions.

New functionality should include appropriate tests whenever possible. Bug fixes should ideally include a regression test that fails before the fix and passes afterward.

## Running the examples

Example workflows are maintained in the repository under:

```text
examples/
```

These examples exercise the public Catalyst API and are useful for checking that changes remain compatible with supported workflows.

When modifying public APIs, graph construction, model builders, tasks, training behavior, checkpointing, or inference, please run the relevant examples in addition to the automated test suite.

## Making changes

A typical contribution workflow is:

1. Fork the Catalyst repository.
2. Create a branch for your change.
3. Install the development version with `python -m pip install -e ".[dev]"`.
4. Make your changes.
5. Add or update tests and documentation as appropriate.
6. Run `python -m pytest -q`.
7. Commit your changes with a clear description.
8. Push the branch to your fork.
9. Open a pull request against the Catalyst `main` branch.

Please keep pull requests focused on a specific change where practical. Large unrelated changes are easier to review when separated into multiple pull requests.

## Pull requests

A pull request should clearly explain:

- what was changed,
- why the change is needed,
- any changes to the public API or expected behavior,
- how the change was tested, and
- any limitations or follow-up work.

Please update the README, examples, docstrings, or other documentation when user-facing behavior changes.

Changes should preserve backward compatibility when practical. If a breaking change is necessary, describe it explicitly in the pull request.

## Code and API considerations

Catalyst is intended to provide reusable infrastructure for graph learning in materials and scientific machine learning. Contributions should therefore favor general interfaces over application-specific assumptions when possible.

In particular:

- task semantics and tensor shapes should remain explicit,
- configuration errors should fail clearly rather than being silently ignored,
- new graph or model functionality should integrate with the existing modular interfaces where practical,
- CPU behavior should remain functional unless a feature is inherently GPU-specific, and
- changes to scientific behavior should be tested against clear expected outcomes.

## Documentation contributions

Documentation improvements are welcome and do not need to involve changes to the Catalyst backend.

Useful contributions include:

- correcting unclear installation or usage instructions,
- adding examples,
- improving docstrings,
- documenting common failure modes,
- clarifying task or model semantics, and
- improving explanations of graph construction and training workflows.

## Questions and support

For questions about using Catalyst, unexpected behavior, or uncertainty about whether a proposed contribution fits the project, please open a GitHub Issue:

https://github.com/Materials-Informatics-Laboratory/Catalyst/issues

Using public issues when possible keeps solutions searchable and useful to other Catalyst users.

## License

By contributing to Catalyst, you agree that your contributions will be distributed under the repository's MIT License.
