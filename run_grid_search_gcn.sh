#!/usr/bin/env bash

set -e

python grid_search_cora.py
python grid_search_citeseer.py
python grid_search_pubmed.py
python grid_search_arxiv.py
python grid_search_twitch.py
python grid_search_elliptic.py
