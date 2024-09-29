#!/bin/bash

# Read the version from pyproject.toml and create a tag
VERSION=$(python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])")

echo "Version: $VERSION"
git tag v$VERSION

# Push the new tag to origin
git push origin v$VERSION