from grid_search_frontdoor_common import run_grid_search
from grid_search_frontdoor_configs import build_frontdoor_search


DATASET = "citeseer"
BACKBONE = "gat"
SEARCH_STAGE = "stage1"  # choose from: stage1, stage2

BASE_CMD, STAGE_GRIDS = build_frontdoor_search(DATASET, BACKBONE)


def main():
    run_grid_search(DATASET, BACKBONE, SEARCH_STAGE, BASE_CMD, STAGE_GRIDS)


if __name__ == "__main__":
    main()
