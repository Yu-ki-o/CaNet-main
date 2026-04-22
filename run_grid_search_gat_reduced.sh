#!/usr/bin/env bash

set -e

python grid_search_gat_reduced_cora.py
python grid_search_gat_reduced_citeseer.py
python grid_search_gat_reduced_pubmed.py
python grid_search_gat_reduced_arxiv.py
python grid_search_gat_reduced_twitch.py
python grid_search_gat_reduced_elliptic.py
