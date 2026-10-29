#!/usr/bin/env bash
# Run once from inside this folder:  bash push.sh
set -e
git init
git add .
git commit -m "Name That LLM: stylometric 18-class attribution + Kaggle notebook + demo website"
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/ganeshb2023-dev/NameTheLLM.git
git push -u origin main
