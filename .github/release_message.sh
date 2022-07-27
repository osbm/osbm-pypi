#!/usr/bin/env bash
previous_tag=$(git tag --sort=-creatordate | sed -n 2p)
echo "Release ${previous_tag} to "
git log "${previous_tag}..HEAD" --oneline > release_message.md

