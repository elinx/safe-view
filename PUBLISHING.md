# Publishing to PyPI

To publish the SafeView package to PyPI, follow these steps:

## Prerequisites

1. Make sure you have `build` and `twine` installed:
```bash
pip install build twine
```

2. Create an account on [PyPI](https://pypi.org/account/register/) and [Test PyPI](https://test.pypi.org/account/register/)

## Publishing Process

### 1. Update Version Number
Update the version number in `pyproject.toml`:
```toml
[project]
version = "0.2.0"  # Update this before publishing
```

### 2. Create a Build
```bash
uv build
# or
python -m build
```

### 3. Upload to Test PyPI (Optional but Recommended)
```bash
twine upload --repository testpypi dist/*
```

### 4. Upload to PyPI
```bash
twine upload dist/*
```

## Verification Steps

Before publishing, ensure:
- All dependencies are properly listed in `pyproject.toml`
- The README renders correctly on PyPI
- The package builds without errors
- The command-line interface works correctly
- All necessary metadata is included (author, description, classifiers, etc.)

## Post-Publishing

After successfully publishing to PyPI, users can install SafeView with:
```bash
pip install safe-view
```

## Notes

- Always test on Test PyPI first
- Make sure version numbers follow semantic versioning
- Consider creating a GitHub release alongside PyPI publication