#!/usr/bin/env bash
previous_tag=$(git describe --tags --abbrev=0)
echo "Release ${previous_tag} to "
git log "${previous_tag}..HEAD" --oneline

