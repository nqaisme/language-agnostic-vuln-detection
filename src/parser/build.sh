#!/usr/bin/env bash

git clone https://github.com/tree-sitter/tree-sitter-c
git clone https://github.com/tree-sitter/tree-sitter-cpp
git clone https://github.com/tree-sitter/tree-sitter-python

python3 build.py