import time
import torch

from data_utils import reindex_env


def _ica_style_projection(x, n_components):
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    n_components = min(n_components, x.size(0), x.size(1))

    # ICA usually starts from whitening. We use an SVD-based whitening step so
    # the preprocessing works without external dependencies.
    u, s, v = torch.pca_lowrank(x, q=n_components, center=False)
    projected = x @ v[:, :n_components]
    scale = torch.clamp(s[:n_components], min=1e-6)
    whitened = projected / scale.unsqueeze(0)
    return whitened


def _run_kmeans(x, num_clusters, num_iters=30, seed=42):
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)

    if x.size(0) < num_clusters:
        raise ValueError(f"num_clusters={num_clusters} is larger than num_nodes={x.size(0)}")

    perm = torch.randperm(x.size(0), generator=generator, device=x.device)
    centroids = x[perm[:num_clusters]].clone()
    assignments = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    for _ in range(num_iters):
        dist = torch.cdist(x, centroids)
        new_assignments = dist.argmin(dim=1)
        if torch.equal(new_assignments, assignments):
            break
        assignments = new_assignments

        for cluster_id in range(num_clusters):
            mask = assignments == cluster_id
            if mask.any():
                centroids[cluster_id] = x[mask].mean(dim=0)
            else:
                fallback_idx = torch.randint(0, x.size(0), (1,), generator=generator, device=x.device)
                centroids[cluster_id] = x[fallback_idx].squeeze(0)

    return assignments


def infer_pseudo_envs_with_ica(
    dataset,
    env_num,
    n_components=64,
    num_iters=30,
    seed=42,
    debug=True,
):
    """
    Infer pseudo environments from node features with an ICA-inspired pipeline:
    centering -> whitening -> k-means clustering.
    """
    if not hasattr(dataset, "x"):
        raise ValueError("dataset must contain node features 'x'")

    start_time = time.time()
    x = dataset.x.detach().cpu()
    whitened = _ica_style_projection(x, n_components=n_components)
    pseudo_env = _run_kmeans(whitened, num_clusters=env_num, num_iters=num_iters, seed=seed).cpu()

    dataset.env = pseudo_env.to(torch.long)
    dataset.env_num = int(env_num)
    dataset.train_env_num = reindex_env(dataset, debug=False)

    if debug:
        train_envs = torch.unique(dataset.env[dataset.train_idx]).numel()
        all_envs = torch.unique(dataset.env).numel()
        print(
            f"[INFO] ICA-inspired pseudo environments ready in {time.time() - start_time:.2f}s | "
            f"all envs: {all_envs} | train envs: {train_envs}"
        )

    return dataset
