import argparse
import math
import os
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            print('[WARN] tensorboard is not installed; scalar logging is disabled.')

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

from dataset import *
from eval import eval_acc, eval_f1, eval_rocauc, evaluate_full
from ica_utils import infer_pseudo_envs_with_ica
from logger import Logger
# from model_gmm3_reviewed1_graph_cfam_nego import GraphFrontDoorDAG as FrontDoorNeGoModel
from model_gmm3_reviewed1_layerwise_samediff import GraphFrontDoorDAG as FrontDoorNeGoModel
from model_graph_cfam_gate_direct import GraphFrontDoorDAG as GateDirectModel
from parse import parser_add_main_args


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_frontdoor_dag_args(parser):
    parser.set_defaults(use_cipt_schedule=True, use_cosine_lr=True)
    parser.add_argument('--model_variant', type=str, default='frontdoor_nego',
                        choices=['frontdoor_nego', 'gate_direct'],
                        help='frontdoor_nego keeps the previous model; gate_direct uses the direct Graph-CFAM gate model')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='EMA momentum for spurious context statistics')
    parser.add_argument('--lambda_l1', type=float, default=1e-5,
                        help='L1 sparsity weight inside the DAG regularizer')
    parser.add_argument('--lambda_ind', type=float, default=0.0,
                        help='deprecated in DAG-Core: mediator-spurious decorrelation is logged as zero')
    parser.add_argument('--lambda_dag', type=float, default=0.05,
                        help='weight of DAG acyclicity/sparsity regularization')
    parser.add_argument('--lambda_med', type=float, default=0.0,
                        help='deprecated in DAG-Core: mediator-only supervision is disabled')
    parser.add_argument('--lambda_spu', type=float, default=0.05,
                        help='weight of spurious branch pseudo-environment loss')
    parser.add_argument('--lambda_role', type=float, default=0.0,
                        help='weight of label-free causal/spurious role supervision')
    parser.add_argument('--role_med_y_weight', type=float, default=1.0,
                        help='inside role loss: mediator label-predictive CE weight')
    parser.add_argument('--role_spu_y_weight', type=float, default=1.0,
                        help='inside role loss: spurious label-uniform weight')
    parser.add_argument('--role_spu_env_weight', type=float, default=1.0,
                        help='inside role loss: spurious pseudo-env confidence weight')
    parser.add_argument('--role_med_env_weight', type=float, default=1.0,
                        help='inside role loss: mediator pseudo-env-uniform weight')
    parser.add_argument('--lambda_fd', type=float, default=0.5,
                        help='weight of front-door aggregation loss')
    parser.add_argument('--lambda_cf', type=float, default=0.0,
                        help='weight of CauVQ-style counterfactual consistency loss')
    parser.add_argument('--cf_target', type=str, default='mediator',
                        choices=['mediator', 'spurious', 'both'],
                        help='representation branch intervened by counterfactual regularization')
    parser.add_argument('--cf_mode', type=str, default='shuffle',
                        choices=['shuffle', 'noise', 'zero',
                                 'spurious_shuffle', 'spurious_noise', 'spurious_zero'],
                        help='counterfactual intervention applied to the selected branch')
    parser.add_argument('--cf_samples', type=int, default=1,
                        help='number of counterfactual perturbations per step')
    parser.add_argument('--cf_consistency', type=str, default='cauvq', choices=['cauvq', 'kl', 'mse'],
                        help='prediction consistency metric for counterfactual regularization')
    parser.add_argument('--cf_temp', type=float, default=1.0,
                        help='temperature for KL counterfactual consistency')
    parser.add_argument('--cf_beta', type=float, default=1.0,
                        help='entropy weight inside CauVQ-style counterfactual consistency')
    parser.add_argument('--cf_noise_std', type=float, default=1.0,
                        help='noise scale used when cf_mode=noise')
    parser.add_argument('--lambda_fd_aug', type=float, default=0.0,
                        help='deprecated in DAG-Core: mixed-context extra supervision is disabled')
    parser.add_argument('--lambda_var', type=float, default=0.0,
                        help='weight of front-door context prediction variance minimization')
    parser.add_argument('--lambda_env', type=float, default=0.05,
                        help='weight of environment-uniform loss on mediator branch')
    parser.add_argument('--lambda_inv', type=float, default=0.0,
                        help='deprecated in DAG-Core: cross-environment prediction invariance is disabled')
    parser.add_argument('--lambda_gate', type=float, default=0.0,
                        help='optional mean-gate sparsity regularizer inside the DAG loss')
    parser.add_argument('--lambda_dag_label', type=float, default=0.05,
                        help='weight of direct DAG node/edge latent -> label supervision')
    parser.add_argument('--lambda_sem', type=float, default=0.0,
                        help='deprecated in DAG-Core: semantic reconstruction is removed')
    parser.add_argument('--lambda_spu_y', type=float, default=0.0,
                        help='deprecated in DAG-Core: environment-conditioned spurious label supervision is disabled')
    parser.add_argument('--dag_latent_dim', type=int, default=16,
                        help='bottleneck dimension used for node/edge variables in the learned DAG')
    parser.add_argument('--mediator_temp', type=float, default=8.0,
                        help='temperature of the DAG-based soft mediator selector')
    parser.add_argument('--low_temp', type=float, default=8.0,
                        help='temperature used to detect low-score features')
    parser.add_argument('--low_threshold', type=float, default=0.35,
                        help='threshold for identifying low-score features')
    parser.add_argument('--mediator_threshold', type=float, default=0.5,
                        help='threshold for activating mediator dimensions')
    parser.add_argument('--pollution_coeff', type=float, default=1.0,
                        help='penalty coefficient for feature pollution from low-score nodes')
    parser.add_argument('--edge_pollution_coeff', type=float, default=0.5,
                        help='penalty coefficient for pseudo-env-sensitive edge influence')
    parser.add_argument('--causal_support_coeff', type=float, default=0.5,
                        help='bonus for support from high-label-effect bottleneck features')
    parser.add_argument('--counterexample_coeff', type=float, default=0.0,
                        help='penalty strength for DAG dimensions that fail on hard counterexample samples')
    parser.add_argument('--counterexample_top_frac', type=float, default=0.2,
                        help='fraction of hardest train samples used to estimate per-dimension counterexample penalty')
    parser.add_argument('--counterexample_momentum', type=float, default=0.9,
                        help='EMA momentum for the per-dimension counterexample penalty')
    parser.add_argument('--dag_ablate_label_effect', action='store_true',
                        help='ablation: remove DAG total-effect node->label signal from mediator selection')
    parser.add_argument('--dag_ablate_causal_support', action='store_true',
                        help='ablation: remove high-label-effect node support bonus from mediator selection')
    parser.add_argument('--dag_ablate_pollution', action='store_true',
                        help='ablation: remove low-effect/edge incoming pollution penalty from mediator selection')
    parser.add_argument('--dag_ablate_acyclic_loss', action='store_true',
                        help='ablation: skip the NOTEARS acyclicity term while keeping other DAG terms')
    parser.add_argument('--dag_ablate_flow_consistency', action='store_true',
                        help='ablation: skip mediator-gate alignment with DAG label-flow')
    parser.add_argument('--pseudo_env_balance', type=float, default=1.0,
                        help='balance weight inside label-free pseudo-environment discovery')
    parser.add_argument('--edge_env_momentum', type=float, default=0.9,
                        help='EMA momentum for edge-dimension pseudo-env sensitivity')
    parser.add_argument('--edge_score_temp', type=float, default=2.0,
                        help='temperature for edge semantic gate logits; larger values produce smoother gates')
    parser.add_argument('--edge_blend', type=float, default=0.2,
                        help='residual strength of edge-aware neighbor aggregation')
    parser.add_argument('--disable_node_edge_norm', action='store_true',
                        help='skip final LayerNorm after node-edge fusion for plain-backbone ablations')
    parser.add_argument('--edge_relation_model', type=str, default='mlp',
                        choices=['mlp', 'transformer'],
                        help='edge relation scorer used by direct gate models')
    parser.add_argument('--edge_transformer_heads', type=int, default=1,
                        help='number of attention heads for transformer edge relation scoring')
    parser.add_argument('--nonfeature_blend', type=float, default=0.2,
                        help='how much complementary non-feature summary is allowed into the center node update')
    parser.add_argument('--edge_feat_mode', type=str, default='mul',
                        choices=['mul', 'diff', 'signed_diff', 'degree',
                                 'mul_diff', 'mul_signed_diff',
                                 'concat', 'concat_diff',
                                 'mul_degree', 'diff_degree',
                                 'mul_diff_degree', 'mul_signed_diff_degree'],
                        help=(
                            'edge feature used by the edge reliability MLP: '
                            'mul=h_src*h_dst, diff=|h_src-h_dst|, '
                            'signed_diff=h_src-h_dst, concat=[h_src,h_dst], '
                            'degree=max normalized log degree, or combinations such as '
                            'mul_diff, mul_signed_diff, concat_diff, diff_degree, '
                            'mul_diff_degree and mul_signed_diff_degree'
                        ))
    parser.add_argument('--edge_gate_mode', type=str, default='vector',
                        choices=['scalar', 'vector'],
                        help='scalar uses one edge gate per neighbor; vector uses one gate per hidden dimension')
    parser.add_argument('--use_layerwise_local_igm', action='store_true',
                        help='apply Local-IGM edge routing after each GNN layer before the next layer')
    parser.add_argument('--layerwise_local_igm_include_last', action='store_false',
                        dest='layerwise_local_igm_skip_last',
                        help='also apply layer-wise Local-IGM after the final GNN layer')
    parser.set_defaults(layerwise_local_igm_skip_last=True)
    parser.add_argument('--disable_layerwise_final_edge_fuse', action='store_false',
                        dest='layerwise_final_edge_fuse',
                        help='keep final edge summary for DAG variables but do not fuse it into z after layer-wise routing')
    parser.set_defaults(layerwise_final_edge_fuse=True)
    parser.add_argument('--layerwise_gate_target', type=float, default=0.5,
                        help='target mean edge gate for optional layer-wise gate budget regularization')
    parser.add_argument('--lambda_layerwise_gate', type=float, default=0.0,
                        help='weight for the optional layer-wise gate budget loss')
    parser.add_argument('--lambda_layer_pred_var', type=float, default=0.0,
                        help='weight for per-layer classifier prediction variance under non-feature complement interventions')
    parser.add_argument('--lambda_layer_pred_cls', type=float, default=0.0,
                        help='weight for auxiliary per-layer classifier supervision')
    parser.add_argument('--lambda_enhance_sem', type=float, default=0.0,
                        help='weight of semantic anchoring between edge-enhanced z and pre-enhancement h')
    parser.add_argument('--enhance_sem_mode', type=str, default='cosine', choices=['cosine', 'mse'],
                        help='semantic anchoring metric for edge-enhanced node representations')
    parser.add_argument('--use_graph_cfam', action='store_true',
                        help='replace local edge enhancement with Graph-CFAM smooth/residual causal decoupling')
    parser.add_argument('--disable_final_graph_cfam', action='store_false',
                        dest='use_final_graph_cfam',
                        help='skip the final post-GNN Graph-CFAM pass while keeping pre/layer-wise CFAM enabled')
    parser.set_defaults(use_final_graph_cfam=True)
    parser.add_argument('--graph_cfam_residual_blend', type=float, default=0.1,
                        help='strength of graph high-pass residual in Graph-CFAM')
    parser.add_argument('--use_pre_gnn_graph_cfam', action='store_true',
                        help='apply Graph-CFAM once after input projection and before the first GNN aggregation')
    parser.add_argument('--pre_graph_cfam_blend', type=float, default=0.1,
                        help='causal-local blend for pre-GNN Graph-CFAM')
    parser.add_argument('--pre_graph_cfam_residual_blend', type=float, default=0.0,
                        help='high-pass residual blend for pre-GNN Graph-CFAM')
    parser.add_argument('--graph_cfam_gate_temp', type=float, default=1.0,
                        help='temperature for dimension-wise Graph-CFAM causal-local gate')
    parser.add_argument('--graph_cfam_gate_target', type=float, default=0.5,
                        help='target mean Graph-CFAM gate for optional balance regularization')
    parser.add_argument('--lambda_graph_cfam_gate', type=float, default=0.0,
                        help='weight of Graph-CFAM gate balance regularization')
    parser.add_argument('--lambda_graph_delf', type=float, default=0.0,
                        help='weight of Graph-DELF ambiguous local shortcut decoupling loss')
    parser.add_argument('--graph_delf_top_frac', type=float, default=0.2,
                        help='fraction of ambiguous train nodes used by Graph-DELF')
    parser.add_argument('--graph_delf_margin', type=float, default=0.2,
                        help='cosine margin for pushing mediator away from shortcut prototypes in Graph-DELF')
    parser.add_argument('--graph_delf_shortcut_weight', type=float, default=0.5,
                        help='weight of shortcut-prototype push term inside Graph-DELF')
    parser.add_argument('--direct_z_spurious_mode', type=str, default='shortcut',
                        choices=['shortcut', 'zero', 'z_adapter','none'],
                        help="'shortcut' uses local shortcut summary, 'zero' uses a zero spurious placeholder, 'z_adapter' derives it from z, 'none' predicts directly from enhanced node z and skips front-door context mixing")
    parser.add_argument('--use_multi_ratio_spurious_fd', action='store_true',
                        help='train front-door predictions from multiple masked ratios of z_spurious')
    parser.add_argument('--multi_ratio_spurious_fd_as_main', action='store_true',
                        help='use per-node multi-ratio z_spurious contexts as the main front-door path and skip global context banks')
    parser.add_argument('--multi_ratio_spurious_mode', type=str, default='sample',
                        choices=['sample', 'ratio'],
                        help="'sample' randomly samples per-node spurious contexts; 'ratio' uses the original ratio-scaled self/shuffle contexts")
    parser.add_argument('--multi_ratio_spurious_source', type=str, default='self',
                        choices=['self', 'shuffle'],
                        help='source of node-level spurious contexts: self uses each node, shuffle uses another shuffled node')
    parser.add_argument('--multi_ratio_spurious_ratios', type=str, default='0,0.33,0.67,1.0',
                        help='comma-separated spurious retention ratios; default keeps four ratios including 0 and 1')
    parser.add_argument('--lambda_multi_ratio_fd', type=float, default=0.5,
                        help='weight of mean supervised loss over multi-ratio spurious front-door predictions')
    parser.add_argument('--lambda_multi_ratio_fd_worst', type=float, default=0.2,
                        help='weight of worst-ratio supervised loss for multi-ratio spurious front-door predictions')
    parser.add_argument('--lambda_multi_ratio_fd_cons', type=float, default=0.1,
                        help='weight of prediction consistency among multi-ratio spurious front-door predictions')
    parser.add_argument('--use_layerwise_spurious_contexts', action='store_true',
                        help='add each GNN layer spurious branch to the front-door environment context bank')
    parser.add_argument('--layerwise_spurious_context_weight', type=float, default=1.0,
                        help='scale of layer-wise spurious environment contexts')
    parser.add_argument('--disable_layerwise_spurious_context_detach', action='store_false',
                        dest='layerwise_spurious_context_detach',
                        help='allow gradients to flow through layer-wise spurious contexts')
    parser.set_defaults(layerwise_spurious_context_detach=True)
    parser.add_argument('--use_nego_prompt', action='store_true',
                        help='enable NeGo-lite negative prompt environment inference loss')
    parser.add_argument('--use_nego_context', action='store_true',
                        help='add NeGo-lite extra-class negative prompt answers to the front-door context bank')
    parser.add_argument('--lambda_nego', type=float, default=0.0,
                        help='weight of NeGo-lite negative prompt loss')
    parser.add_argument('--nego_temp', type=float, default=0.2,
                        help='temperature for NeGo-lite prompt/prototype matching')
    parser.add_argument('--nego_context_weight', type=float, default=1.0,
                        help='scale of NeGo-lite front-door contexts')
    parser.add_argument('--nego_momentum', type=float, default=0.9,
                        help='EMA momentum for NeGo-lite inference-time context bank')
    parser.add_argument('--disable_nego_detach_source', action='store_false',
                        dest='nego_detach_source',
                        help='allow gradients from NeGo prompt loss to flow into the source branch')
    parser.set_defaults(nego_detach_source=True)
    parser.add_argument('--nego_source', type=str, default='spurious', choices=['spurious', 'mediator', 'z'],
                        help='representation source for NeGo-lite negative environment prompts')
    parser.add_argument('--nego_context_mode', type=str, default='class_mean',
                        choices=['class_mean', 'sample_mix'],
                        help='NeGo context construction: global extra-class means or per-sample mixed negative answers')
    parser.add_argument('--nego_mix_k', type=int, default=3,
                        help='number of per-sample mixed NeGo contexts when nego_context_mode=sample_mix')
    parser.add_argument('--nego_mix_alpha', type=float, default=0.5,
                        help='Dirichlet concentration for sample-wise NeGo answer mixing')
    parser.add_argument('--fd_context_source', type=str, default='mixed',
                        choices=['mixed', 'nego_only'],
                        help='front-door context bank source: merge all enabled contexts or use only NeGo contexts')
    parser.add_argument('--lirs_proto_temp', type=float, default=1.0,
                        help='temperature for layer-wise LIRS prototype matching')
    parser.add_argument('--lirs_proto_momentum', type=float, default=0.9,
                        help='EMA momentum for layer-wise LIRS class prototype bank')
    parser.add_argument('--lirs_proto_min_count', type=int, default=1,
                        help='minimum train nodes per class needed to update a layer-wise LIRS prototype')
    parser.add_argument('--lambda_lirs_proto_causal', type=float, default=0.0,
                        help='weight for aligning each Graph-CFAM causal-local branch to same-class layer prototypes')
    parser.add_argument('--lambda_lirs_proto_spurious', type=float, default=0.0,
                        help='weight for making each Graph-CFAM domain branch class-uninformative against layer prototypes')
    parser.add_argument('--use_global_info', action='store_true',
                        help='inject MLEI/AdvDIFFormer-style global information before DAG/front-door splitting')
    parser.add_argument('--disable_global_contexts', action='store_false', dest='use_global_contexts',
                        help='disable MLEI global-attention context prototypes in the front-door context bank')
    parser.set_defaults(use_global_contexts=None)
    parser.add_argument('--global_info_mode', type=str, default='advective', choices=['linear', 'advective'],
                        help="'linear' uses MLEI-style global linear attention; 'advective' mixes it with local topology")
    parser.add_argument('--global_alpha', type=float, default=0.2,
                        help='residual strength of the global information channel')
    parser.add_argument('--global_beta', type=float, default=0.5,
                        help='topology reliance weight beta for advective global-local mixing')
    parser.add_argument('--global_steps', type=int, default=1,
                        help='number of lightweight advective propagation steps')
    parser.add_argument('--global_local_source', type=str, default='gcn', choices=['edge', 'gcn'],
                        help="'edge' uses learned edge-gated aggregation as AdvDIFFormer V; 'gcn' uses raw GCN propagation")
    parser.add_argument('--global_context_weight', type=float, default=1.0,
                        help='scale of global-attention environment prototypes used as front-door contexts')
    parser.add_argument('--lambda_global_env', type=float, default=0.0,
                        help='weight of local/global pseudo-environment consistency')
    parser.add_argument('--lambda_entropy_dro', type=float, default=0.0,
                        help='mix ratio for entropy-smoothed DRO group loss')
    parser.add_argument('--dro_entropy_beta', type=float, default=1.0,
                        help='entropy temperature for smoother worst-group weights')
    parser.add_argument('--dro_num_groups', type=int, default=4,
                        help='number of degree buckets used by entropy-DRO')
    parser.add_argument('--dro_group_by', type=str, default='degree_label',
                        choices=['degree', 'label', 'degree_label', 'none'],
                        help='grouping rule for entropy-DRO nominal/worst-case buckets')
    parser.add_argument('--fd_blend', type=float, default=0.5,
                        help='blend ratio between mediator logits and front-door aggregated logits')
    parser.add_argument('--eval_pred_mode', type=str, default='mediator',
                        choices=['blend', 'mediator', 'frontdoor'],
                        help='prediction path used only during evaluation/inference; default uses enhanced node z directly')
    parser.add_argument('--use_ica_split', action='store_true',
                        help='replace the DAG mediator/spurious split with an end-to-end ICA-like component split over enhanced z')
    parser.add_argument('--use_enhanced_as_causal', action='store_true',
                        help='ablation: bypass DAG/ICA split and use the enhanced node representation z directly as the causal mediator')
    parser.add_argument('--ica_components', type=int, default=16,
                        help='number of ICA-like independent components before causal/spurious grouping')
    parser.add_argument('--ica_gate_temp', type=float, default=1.0,
                        help='temperature for the causal/spurious ICA component gate')
    parser.add_argument('--ica_gate_target', type=float, default=0.5,
                        help='target fraction of ICA components assigned to the causal group')
    parser.add_argument('--lambda_ica_cov', type=float, default=0.0,
                        help='weight for off-diagonal covariance penalty among ICA-like components')
    parser.add_argument('--lambda_ica_ng', type=float, default=0.0,
                        help='weight for non-Gaussianity encouragement among ICA-like components')
    parser.add_argument('--lambda_ica_gate', type=float, default=0.0,
                        help='weight for ICA component gate balance')
    parser.add_argument('--lambda_ica_entropy', type=float, default=0.0,
                        help='weight for low-entropy hard assignment of ICA components to causal/spurious groups')
    parser.add_argument('--proto_aug_k', type=int, default=1,
                        help='deprecated: prototype mixup is replaced by GMM context sampling in the DAG model')
    parser.add_argument('--proto_mix_alpha', type=float, default=1.0,
                        help='deprecated: prototype mixup is replaced by GMM context sampling in the DAG model')
    parser.add_argument('--disable_spu_gmm', action='store_false', dest='use_spu_gmm',
                        help='disable GMM sampling for spurious environment contexts')
    parser.set_defaults(use_spu_gmm=True)
    parser.add_argument('--gmm_sample_k', type=int, default=3,
                        help='number of GMM-sampled spurious contexts; <=0 uses K')
    parser.add_argument('--gmm_min_var', type=float, default=1e-4,
                        help='minimum diagonal variance used by spurious-context GMM')
    parser.add_argument('--gmm_max_std', type=float, default=1.0,
                        help='maximum std for GMM context sampling; <=0 disables clipping')
    parser.add_argument('--disable_dag_mixer', action='store_false', dest='use_dag_mixer',
                        help='disable DAG-masked latent mixing and use the old concat fuser')
    parser.set_defaults(use_dag_mixer=True)
    parser.add_argument('--dag_mixer_heads', type=int, default=1,
                        help='attention heads in the DAG-masked latent mixer')
    parser.add_argument('--dag_mixer_layers', type=int, default=2,
                        help='number of masked latent attention layers')
    parser.add_argument('--disable_cipt_schedule', action='store_false', dest='use_cipt_schedule',
                        help='disable the CIPT-style curriculum that warms up decomposition before full intervention')
    parser.add_argument('--decomp_warmup_epochs', type=int, default=50,
                        help='warmup epochs that emphasize causal decomposition before front-door intervention')
    parser.add_argument('--intervention_ramp_epochs', type=int, default=100,
                        help='epochs used to smoothly ramp front-door and invariance losses after warmup')
    parser.add_argument('--min_intervention_scale', type=float, default=0.0,
                        help='minimum scale applied to intervention-related losses during the warmup stage')
    parser.add_argument('--disable_cosine_lr', action='store_false', dest='use_cosine_lr',
                        help='disable cosine annealing and keep a constant learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='minimum learning rate for cosine annealing')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='gradient clipping norm for stable few-shot front-door training; <= 0 disables it')
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='stop a run if validation does not improve for this many epochs; default 0 disables')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                        help='minimum validation improvement required to reset early stopping')


def sanitize_name(name):
    safe_name = "".join(
        ch if ch.isalnum() or ch in ('-', '_', '.') else '_'
        for ch in str(name).strip()
    ).strip('._')
    return safe_name


def capture_lambda_state(model):
    lambda_names = (
        'lambda_dag',
        'lambda_dag_label',
        'lambda_med',
        'lambda_spu',
        'lambda_role',
        'lambda_fd',
        'lambda_cf',
        'lambda_fd_aug',
        'lambda_var',
        'lambda_ind',
        'lambda_env',
        'lambda_inv',
        'lambda_global_env',
        'lambda_entropy_dro',
        'lambda_multi_ratio_fd',
        'lambda_multi_ratio_fd_worst',
        'lambda_multi_ratio_fd_cons',
        'lambda_layerwise_gate',
        'lambda_graph_cfam_gate',
        'lambda_graph_delf',
        'lambda_enhance_sem',
        'lambda_nego',
        'lambda_lirs_proto_causal',
        'lambda_lirs_proto_spurious',
        'lambda_spu_y',
        'lambda_ica_cov',
        'lambda_ica_ng',
        'lambda_ica_gate',
        'lambda_ica_entropy',
    )
    return {name: float(getattr(model, name)) for name in lambda_names}


def restore_lambda_state(model, lambda_state):
    for name, value in lambda_state.items():
        setattr(model, name, value)


def cosine_rampup(progress):
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * progress)


def apply_cipt_schedule(model, base_lambdas, epoch, args):
    """
    CIPT-inspired curriculum:
    1) stabilize mediator / spurious decomposition first,
    2) then gradually activate intervention-related objectives.
    """
    restore_lambda_state(model, base_lambdas)

    if not args.use_cipt_schedule:
        return {'intervention_scale': 1.0, 'dag_scale': 1.0}

    warmup_epochs = max(0, int(args.decomp_warmup_epochs))
    ramp_epochs = max(0, int(args.intervention_ramp_epochs))
    min_scale = float(args.min_intervention_scale)

    if epoch < warmup_epochs:
        intervention_scale = min_scale
    elif ramp_epochs == 0:
        intervention_scale = 1.0
    else:
        progress = (epoch - warmup_epochs + 1) / float(ramp_epochs)
        intervention_scale = min_scale + (1.0 - min_scale) * cosine_rampup(progress)

    intervention_scale = min(max(intervention_scale, 0.0), 1.0)
    dag_scale = 0.5 + 0.5 * intervention_scale

    for name in (
        'lambda_fd',
        'lambda_cf',
        'lambda_fd_aug',
        'lambda_var',
        'lambda_env',
        'lambda_inv',
        'lambda_role',
        'lambda_entropy_dro',
        'lambda_multi_ratio_fd',
        'lambda_multi_ratio_fd_worst',
        'lambda_multi_ratio_fd_cons',
        'lambda_layerwise_gate',
        'lambda_graph_cfam_gate',
        'lambda_graph_delf',
        'lambda_nego',
        'lambda_lirs_proto_causal',
        'lambda_lirs_proto_spurious',
        'lambda_ica_cov',
        'lambda_ica_ng',
        'lambda_ica_gate',
        'lambda_ica_entropy',
    ):
        setattr(model, name, base_lambdas[name] * intervention_scale)
    setattr(model, 'lambda_dag', base_lambdas['lambda_dag'] * dag_scale)

    return {
        'intervention_scale': intervention_scale,
        'dag_scale': dag_scale,
    }


parser = argparse.ArgumentParser(description='Graph Front-Door DAG-Core Training Pipeline')
parser_add_main_args(parser)
add_frontdoor_dag_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')

if args.dataset == 'twitch':
    dataset = load_twitch_dataset(args.data_dir, train_num=3)
elif args.dataset == 'elliptic':
    dataset = load_elliptic_dataset(args.data_dir, train_num=5)
elif args.dataset == 'arxiv':
    dataset = load_arxiv_dataset(args.data_dir, train_num=3)
elif args.dataset in ('cora', 'citeseer', 'pubmed'):
    dataset = load_synthetic_dataset(
        args.data_dir,
        args.dataset,
        train_num=3,
        combine=args.combine_result,
    )
else:
    raise ValueError('Invalid dataname')

if args.infer_env:
    dataset = infer_pseudo_envs_with_ica(
        dataset,
        env_num=args.infer_env_num,
        n_components=args.infer_env_components,
        num_iters=args.infer_env_iters,
        seed=args.seed,
    )

if len(dataset.y.shape) == 1:
    dataset.y = dataset.y.unsqueeze(1)

args.train_env_num = int(dataset.train_env_num)

c = max(dataset.y.max().item() + 1, dataset.y.shape[1])
d = dataset.x.shape[1]

print(
    f"dataset {args.dataset}: all nodes {dataset.num_nodes} | edges {dataset.edge_index.size(1)} | "
    f"classes {c} | feats {d}"
)
print(
    f"train nodes {dataset.train_idx.shape[0]} | valid nodes {dataset.valid_idx.shape[0]} | "
    f"test in nodes {dataset.test_in_idx.shape[0]}"
)
m = ""
for i in range(len(dataset.test_ood_idx)):
    m += f"test ood{i + 1} nodes {dataset.test_ood_idx[i].shape[0]} "
print(m)
print(f'[INFO] env numbers: {dataset.env_num} train env numbers: {dataset.train_env_num}')

model_cls = GateDirectModel if args.model_variant == 'gate_direct' else FrontDoorNeGoModel
model = model_cls(d, c, args, device).to(device)

if args.dataset in ('elliptic', 'twitch'):
    pos_weight = torch.full((c,), float(args.pos_weight), device=device)
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
else:
    criterion = nn.CrossEntropyLoss(reduction='none')

if args.dataset == 'twitch':
    eval_func = eval_rocauc
elif args.dataset == 'elliptic':
    eval_func = eval_f1
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)
print('MODEL:', model)

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
run_name = sanitize_name(args.result_name)
if not run_name:
    run_name = (
        f"{current_time}_fd_dag_core_d{args.lambda_dag}_dl{args.lambda_dag_label}_fd{args.lambda_fd}"
        f"_warm{args.decomp_warmup_epochs}_ramp{args.intervention_ramp_epochs}"
    )
log_dir = os.path.join('.', 'runs', args.dataset, 'frontdoor_dag_core', run_name)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
print(f"[INFO] TensorBoard logging activated. Logs will be saved to: {log_dir}")
print(
    f"[INFO] Training recipe: {args.model_variant} | CIPT schedule: {args.use_cipt_schedule} | "
    f"cosine lr: {args.use_cosine_lr} | warmup: {args.decomp_warmup_epochs} | "
    f"ramp: {args.intervention_ramp_epochs} | grad clip: {args.grad_clip} | "
    f"DAG mixer: {model.use_dag_mixer} | edge feat: {args.edge_feat_mode} | "
    f"edge gate: {args.edge_gate_mode} | "
    f"node-edge norm: {not args.disable_node_edge_norm} | "
    f"entropy-DRO: {args.lambda_entropy_dro} "
    f"(group={args.dro_group_by}, buckets={args.dro_num_groups}, beta={args.dro_entropy_beta}) | "
    f"CF: {args.lambda_cf} (target={args.cf_target}, mode={args.cf_mode}, "
    f"samples={args.cf_samples}, {args.cf_consistency}) | "
    f"role: {args.lambda_role} (med_y={args.role_med_y_weight}, "
    f"spu_y={args.role_spu_y_weight}, spu_env={args.role_spu_env_weight}, "
    f"med_env={args.role_med_env_weight}) | "
    f"enhanced-as-causal: {model.use_enhanced_as_causal} | "
    f"ICA split: {model.use_ica_split} (components={args.ica_components}, "
    f"cov={args.lambda_ica_cov}, ng={args.lambda_ica_ng}, "
    f"gate={args.lambda_ica_gate}, entropy={args.lambda_ica_entropy}) | "
    f"eval pred: {args.eval_pred_mode} | "
    f"GMM contexts: {args.use_spu_gmm} | "
    f"GMM sample k: {args.gmm_sample_k if args.gmm_sample_k > 0 else args.K} | "
    f"layerwise local IGM: {args.use_layerwise_local_igm} "
    f"(skip_last={args.layerwise_local_igm_skip_last}, final_fuse={args.layerwise_final_edge_fuse}, "
    f"target={args.layerwise_gate_target}, lambda={args.lambda_layerwise_gate}) | "
    f"enhance sem: {args.lambda_enhance_sem} ({args.enhance_sem_mode}) | "
    f"Graph-CFAM: {args.use_graph_cfam} "
    f"(residual={args.graph_cfam_residual_blend}, gate_temp={args.graph_cfam_gate_temp}, "
    f"pre={args.use_pre_gnn_graph_cfam}, pre_blend={args.pre_graph_cfam_blend}, "
    f"final={args.use_final_graph_cfam}, "
    f"gate_lambda={args.lambda_graph_cfam_gate}, delf_lambda={args.lambda_graph_delf}, "
    f"delf_top={args.graph_delf_top_frac}, direct_z_spu={args.direct_z_spurious_mode}) | "
    f"multi-ratio spu FD: {args.use_multi_ratio_spurious_fd} "
    f"(main={args.multi_ratio_spurious_fd_as_main}, "
    f"mode={args.multi_ratio_spurious_mode}, "
    f"source={args.multi_ratio_spurious_source}, "
    f"ratios={args.multi_ratio_spurious_ratios}, "
    f"lambda_mean={args.lambda_multi_ratio_fd}, "
    f"lambda_worst={args.lambda_multi_ratio_fd_worst}, "
    f"lambda_cons={args.lambda_multi_ratio_fd_cons}) | "
    f"layerwise spurious ctx: {args.use_layerwise_spurious_contexts} "
    f"(weight={args.layerwise_spurious_context_weight}, "
    f"detach={args.layerwise_spurious_context_detach}) | "
    f"NeGo-lite: prompt={args.use_nego_prompt}, context={args.use_nego_context}, "
    f"lambda={args.lambda_nego}, source={args.nego_source}, temp={args.nego_temp}, "
    f"mode={args.nego_context_mode}, mix_k={args.nego_mix_k}, mix_alpha={args.nego_mix_alpha}, "
    f"ctx_weight={args.nego_context_weight}, fd_ctx_source={args.fd_context_source} | "
    f"LIRS layer proto: temp={args.lirs_proto_temp}, momentum={args.lirs_proto_momentum}, "
    f"min_count={args.lirs_proto_min_count}, "
    f"lambda_causal={args.lambda_lirs_proto_causal}, "
    f"lambda_spurious={args.lambda_lirs_proto_spurious} | "
    f"global info: {args.use_global_info} ({args.global_info_mode}, "
    f"alpha={args.global_alpha}, beta={args.global_beta}, steps={args.global_steps}, "
    f"local={args.global_local_source}) | "
    f"DAG ablate: label_effect={args.dag_ablate_label_effect}, "
    f"causal_support={args.dag_ablate_causal_support}, "
    f"pollution={args.dag_ablate_pollution}, "
    f"acyclic={args.dag_ablate_acyclic_loss}, "
    f"flow_consistency={args.dag_ablate_flow_consistency}"
)

dataset.x = dataset.x.to(device)
dataset.y = dataset.y.to(device)
dataset.edge_index = dataset.edge_index.to(device)
dataset.env = dataset.env.to(device)

base_lambdas = capture_lambda_state(model)

for run in range(args.runs):
    model.reset_parameters()
    restore_lambda_state(model, base_lambdas)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.use_cosine_lr and args.epochs > 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=min(args.min_lr, args.lr),
        )
    best_valid = float('-inf')
    best_epoch = -1
    stale_epochs = 0

    for epoch in range(args.epochs):
        schedule_state = apply_cipt_schedule(model, base_lambdas, epoch, args)
        model.train()
        optimizer.zero_grad()
        losses = model.compute_losses(dataset, criterion, args, update_state=True)
        losses['total_loss'].backward()

        grad_norm = None
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        model.apply_state_update(losses.get('state_payload'))

        result = evaluate_full(model, dataset, eval_func)
        logger.add_result(run, result)
        current_lr = optimizer.param_groups[0]['lr']

        global_step = run * args.epochs + epoch
        writer.add_scalar('Loss/Total', losses['total_loss'].item(), global_step)
        writer.add_scalar('Loss/Cls', losses['loss_cls'].item(), global_step)
        writer.add_scalar('Loss/ClsMean', losses['loss_cls_mean'].item(), global_step)
        writer.add_scalar('Loss/Med', losses['loss_med'].item(), global_step)
        writer.add_scalar('Loss/FD', losses['loss_fd'].item(), global_step)
        writer.add_scalar('Loss/FDMean', losses['loss_fd_mean'].item(), global_step)
        writer.add_scalar('Loss/CF', (model.lambda_cf * losses['loss_cf']).item(), global_step)
        writer.add_scalar('Loss/FDAug', (model.lambda_fd_aug * losses['loss_fd_aug']).item(), global_step)
        writer.add_scalar('Loss/Ind', (model.lambda_ind * losses['loss_ind']).item(), global_step)
        writer.add_scalar('Loss/DAG', (model.lambda_dag * losses['loss_dag']).item(), global_step)
        writer.add_scalar('Loss/DAGLabel', (model.lambda_dag_label * losses['loss_dag_label']).item(), global_step)
        writer.add_scalar('Loss/ICACov', (model.lambda_ica_cov * losses['loss_ica_cov']).item(), global_step)
        writer.add_scalar('Loss/ICANonGaussian', (model.lambda_ica_ng * losses['loss_ica_ng']).item(), global_step)
        writer.add_scalar('Loss/ICAGate', (model.lambda_ica_gate * losses['loss_ica_gate']).item(), global_step)
        writer.add_scalar('Loss/ICAEntropy', (model.lambda_ica_entropy * losses['loss_ica_entropy']).item(), global_step)
        writer.add_scalar('Loss/Spu', (model.lambda_spu * losses['loss_spu']).item(), global_step)
        writer.add_scalar('Loss/Role', (model.lambda_role * losses['loss_role']).item(), global_step)
        writer.add_scalar('Loss/RoleMedY', losses['loss_role_med_y'].item(), global_step)
        writer.add_scalar('Loss/RoleSpuY', losses['loss_role_spu_y'].item(), global_step)
        writer.add_scalar('Loss/RoleSpuEnv', losses['loss_role_spu_env'].item(), global_step)
        writer.add_scalar('Loss/RoleMedEnv', losses['loss_role_med_env'].item(), global_step)
        writer.add_scalar('Loss/SpuY', (model.lambda_spu_y * losses['loss_spu_y']).item(), global_step)
        writer.add_scalar('Loss/EnvMed', (model.lambda_env * losses['loss_env_med']).item(), global_step)
        writer.add_scalar('Loss/Inv', (model.lambda_inv * losses['loss_inv']).item(), global_step)
        writer.add_scalar('Loss/Var', (model.lambda_var * losses['loss_var']).item(), global_step)
        writer.add_scalar('Loss/GlobalEnv', (model.lambda_global_env * losses['loss_global_env']).item(), global_step)
        writer.add_scalar('Loss/LayerwiseGate', (model.lambda_layerwise_gate * losses['loss_layerwise_gate']).item(), global_step)
        writer.add_scalar('Loss/GraphCFAMGate', (model.lambda_graph_cfam_gate * losses['loss_graph_cfam_gate']).item(), global_step)
        writer.add_scalar('Loss/GraphDELF', (model.lambda_graph_delf * losses['loss_graph_delf']).item(), global_step)
        writer.add_scalar('Loss/EnhanceSem', (model.lambda_enhance_sem * losses['loss_enhance_sem']).item(), global_step)
        writer.add_scalar('Loss/MultiRatioFD', (model.lambda_multi_ratio_fd * losses['loss_multi_ratio_fd']).item(), global_step)
        writer.add_scalar('Loss/MultiRatioFDWorst', (model.lambda_multi_ratio_fd_worst * losses['loss_multi_ratio_fd_worst']).item(), global_step)
        writer.add_scalar('Loss/MultiRatioFDCons', (model.lambda_multi_ratio_fd_cons * losses['loss_multi_ratio_fd_cons']).item(), global_step)
        writer.add_scalar('Loss/NeGo', (model.lambda_nego * losses['loss_nego']).item(), global_step)
        writer.add_scalar('Loss/LIRSProto', losses['loss_lirs_proto'].item(), global_step)
        writer.add_scalar('Loss/LIRSProtoCausal', (model.lambda_lirs_proto_causal * losses['loss_lirs_proto_causal']).item(), global_step)
        writer.add_scalar('Loss/LIRSProtoSpurious', (model.lambda_lirs_proto_spurious * losses['loss_lirs_proto_spurious']).item(), global_step)
        writer.add_scalar('Diag/NeGoExtraScore', losses['nego_extra_score'].item(), global_step)
        writer.add_scalar('Diag/NeGoSelfScore', losses['nego_self_score'].item(), global_step)
        writer.add_scalar('Diag/LIRSProtoCausalAcc', losses['lirs_proto_causal_acc'].item(), global_step)
        writer.add_scalar('Diag/LIRSProtoSpuriousEntropy', losses['lirs_proto_spurious_entropy'].item(), global_step)
        writer.add_scalar('DRO/WeightEntropy', losses['dro_weight_entropy'].item(), global_step)
        writer.add_scalar('DRO/MaxWeight', losses['dro_max_weight'].item(), global_step)
        writer.add_scalar('Graph/LayerwiseGateMean', losses['layerwise_gate_mean'].item(), global_step)
        writer.add_scalar('Graph/LayerwiseGateLayers', losses['layerwise_gate_layers'].item(), global_step)
        writer.add_scalar('Graph/GraphCFAMGateMean', losses['graph_cfam_gate_mean'].item(), global_step)
        writer.add_scalar('Graph/GraphCFAMLayers', losses['graph_cfam_layers'].item(), global_step)
        writer.add_scalar('Graph/MediatorGate', losses['mediator_gate_mean'].item(), global_step)
        writer.add_scalar('Graph/ICAGate', losses['ica_gate_mean'].item(), global_step)
        writer.add_scalar('Graph/CausalScore', losses['causal_score_mean'].item(), global_step)
        writer.add_scalar('Graph/CFShift', losses['cf_pred_shift'].item(), global_step)
        writer.add_scalar('Graph/PollutionScore', losses['pollution_score_mean'].item(), global_step)
        writer.add_scalar('Graph/CounterexamplePenalty', losses['counterexample_penalty_mean'].item(), global_step)
        writer.add_scalar('Graph/CounterexamplePenaltyBatch', losses['counterexample_penalty_batch_mean'].item(), global_step)
        writer.add_scalar('Graph/NumContexts', losses['num_contexts'].item(), global_step)
        writer.add_scalar('Graph/NumMixedContexts', losses['num_mixed_contexts'].item(), global_step)
        writer.add_scalar('Graph/NumGMMContexts', losses['num_gmm_contexts'].item(), global_step)
        writer.add_scalar('Graph/NumGlobalContexts', losses['num_global_contexts'].item(), global_step)
        writer.add_scalar('Graph/NumMultiRatioContexts', losses['num_multi_ratio_contexts'].item(), global_step)
        writer.add_scalar('Graph/NumNeGoContexts', losses['num_nego_contexts'].item(), global_step)
        writer.add_scalar('Graph/NumLIRSLayerProtos', losses['num_lirs_layer_protos'].item(), global_step)
        writer.add_scalar('Schedule/LR', current_lr, global_step)
        writer.add_scalar('Schedule/InterventionScale', schedule_state['intervention_scale'], global_step)
        writer.add_scalar('Schedule/DAGScale', schedule_state['dag_scale'], global_step)
        if grad_norm is not None:
            writer.add_scalar('Grad/Norm', grad_norm.item(), global_step)
        writer.add_scalar('Metrics/1_Train', result[0] * 100, global_step)
        writer.add_scalar('Metrics/2_Valid', result[1] * 100, global_step)
        writer.add_scalar('Metrics/3_Test_In', result[2] * 100, global_step)
        for i in range(len(result) - 3):
            writer.add_scalar(f'Metrics/4_Test_OOD_{i + 1}', result[i + 3] * 100, global_step)

        valid_score = result[1]
        if valid_score > best_valid + args.early_stop_min_delta:
            best_valid = valid_score
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch % args.display_step == 0:
            msg = (
                f"Epoch: {epoch:03d}, Loss: {losses['total_loss'].item():.4f}, "
                f"Cls: {losses['loss_cls'].item():.4f}, "
                f"Med: {(model.lambda_med * losses['loss_med']).item():.4f}, "
                f"FD: {(model.lambda_fd * losses['loss_fd']).item():.4f}, "
                f"CF: {(model.lambda_cf * losses['loss_cf']).item():.4f}, "
                f"FDAug: {(model.lambda_fd_aug * losses['loss_fd_aug']).item():.4f}, "
                f"Ind: {(model.lambda_ind * losses['loss_ind']).item():.4f}, "
                f"DAG: {(model.lambda_dag * losses['loss_dag']).item():.4f}, "
                f"DAGLabel: {(model.lambda_dag_label * losses['loss_dag_label']).item():.4f}, "
                f"ICA: {((model.lambda_ica_cov * losses['loss_ica_cov']) + (model.lambda_ica_ng * losses['loss_ica_ng']) + (model.lambda_ica_gate * losses['loss_ica_gate']) + (model.lambda_ica_entropy * losses['loss_ica_entropy'])).item():.4f}, "
                f"Spu: {(model.lambda_spu * losses['loss_spu']).item():.4f}, "
                f"Role: {(model.lambda_role * losses['loss_role']).item():.4f}, "
                f"SpuY: {(model.lambda_spu_y * losses['loss_spu_y']).item():.4f}, "
                f"EnvMed: {(model.lambda_env * losses['loss_env_med']).item():.4f}, "
                f"Inv: {(model.lambda_inv * losses['loss_inv']).item():.4f}, "
                f"Var: {(model.lambda_var * losses['loss_var']).item():.4f}, "
                f"GlobalEnv: {(model.lambda_global_env * losses['loss_global_env']).item():.4f}, "
                f"LayerGate: {(model.lambda_layerwise_gate * losses['loss_layerwise_gate']).item():.4f}, "
                f"GCFAMGate: {(model.lambda_graph_cfam_gate * losses['loss_graph_cfam_gate']).item():.4f}, "
                f"GDELF: {(model.lambda_graph_delf * losses['loss_graph_delf']).item():.4f}, "
                f"EnhSem: {(model.lambda_enhance_sem * losses['loss_enhance_sem']).item():.4f}, "
                f"MRFD: {((model.lambda_multi_ratio_fd * losses['loss_multi_ratio_fd']) + (model.lambda_multi_ratio_fd_worst * losses['loss_multi_ratio_fd_worst']) + (model.lambda_multi_ratio_fd_cons * losses['loss_multi_ratio_fd_cons'])).item():.4f}, "
                f"LIRSProto: {losses['loss_lirs_proto'].item():.4f}, "
                f"CFShift: {losses['cf_pred_shift'].item():.4f}, "
                f"LayerGateMean: {losses['layerwise_gate_mean'].item():.3f}, "
                f"GraphCFAMGate: {losses['graph_cfam_gate_mean'].item():.3f}, "
                f"ICAGate: {losses['ica_gate_mean'].item():.3f}, "
                f"LayerGateLayers: {int(losses['layerwise_gate_layers'].item())}, "
                f"GraphCFAMLayers: {int(losses['graph_cfam_layers'].item())}, "
                f"DROEnt: {losses['dro_weight_entropy'].item():.3f}, "
                f"DROMax: {losses['dro_max_weight'].item():.3f}, "
                f"LR: {current_lr:.6f}, "
                f"IntScale: {schedule_state['intervention_scale']:.3f}, "
                f"CEPen: {losses['counterexample_penalty_mean'].item():.3f}, "
                f"Ctx: {int(losses['num_contexts'].item())}, "
                f"GMMCtx: {int(losses['num_gmm_contexts'].item())}, "
                f"GlobalCtx: {int(losses['num_global_contexts'].item())}, "
                f"MRCtx: {int(losses['num_multi_ratio_contexts'].item())}, "
                f"LIRSProtoN: {int(losses['num_lirs_layer_protos'].item())}, "
                f"MixCtx: {int(losses['num_mixed_contexts'].item())}, "
                f"Train: {100 * result[0]:.2f}%, Valid: {100 * result[1]:.2f}%, "
                f"Test In: {100 * result[2]:.2f}% "
            )
            for i in range(len(result) - 3):
                msg += f"Test OOD{i + 1}: {100 * result[i + 3]:.2f}% "
            print(msg)

        if scheduler is not None:
            scheduler.step()

        if args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
            print(
                f"[INFO] Early stopping run {run + 1:02d} at epoch {epoch + 1}. "
                f"Best valid {100 * best_valid:.2f}% was reached at epoch {best_epoch + 1}."
            )
            break

    logger.print_statistics(run)

logger.print_statistics()
if args.store_result:
    logger.output(args)

writer.close()
print('[INFO] TensorBoard writer closed.')
