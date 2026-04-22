#!/usr/bin/env bash

set -e

python grid_search_mlei_gat_cora.py
python grid_search_mlei_gat_citeseer.py
python grid_search_mlei_gat_pubmed.py
python grid_search_mlei_gat_arxiv.py
python grid_search_mlei_gat_twitch.py
python grid_search_mlei_gat_elliptic.py
