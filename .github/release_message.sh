#!/usr/bin/env bash
previous_tag=$(git tag --sort=-creatordate | sed -n 2p)
git log "${previous_tag}..HEAD" --oneline

