# GCN backbone
python main_gmm3_reviewed1.py --dataset cora --backbone gcn --weight_decay 5e-5 --tau 1 --dropout 0.2 --env_type graph --combine_result --store --edge_feat_mode mul --gmm_sample_k 3 --edge_blend 0.2 --edge_score_temp 5.0 --dag_latent_dim 16 --lambda_dag 0.05 --lambda_dag_label 0.05 --lambda_spu 0.05 --lambda_env 0.05 --lambda_fd 0.5 --fd_blend 0.5
python main_gmm3_reviewed1.py --dataset citeseer --backbone gcn --weight_decay 5e-5 --tau 1 --dropout 0.1 --env_type graph --combine_result --store --edge_feat_mode mul --gmm_sample_k 3 --edge_blend 0.2 --edge_score_temp 5.0 --dag_latent_dim 16 --lambda_dag 0.05 --lambda_dag_label 0.05 --lambda_spu 0.05 --lambda_env 0.05 --lambda_fd 0.5 --fd_blend 0.5 --hidden_channels 32
# Aligned to results/pubmed/gcn/pubmed_gcn_edgectx_fine_spugate_thr0.40_temp8.0_..._residual_gmm2.txt.
# edge_spu_* args from that run are not available in model_gmm3, so this line uses the shared hyperparameters.
python main_gmm3_reviewed1.py --dataset pubmed --backbone gcn --lr 0.005 --weight_decay 5e-5 --tau 2 --K 2 --dropout 0.3 --hidden_channels 64 --num_layers 2 --env_type graph --combine_result --store --edge_feat_mode mul --gmm_sample_k 2 --edge_blend 0.2 --edge_score_temp 5.0 --dag_latent_dim 16 --lambda_dag 0.05 --lambda_dag_label 0.05 --lambda_spu 0.05 --lambda_env 0.05 --lambda_fd 0.5 --fd_blend 0.5 --display_step 10 --runs 5 --epochs 500 --early_stop_patience 80 --early_stop_min_delta 0.0001 --result_name pubmed_gcn_gmm3_from_edgectx_fine_spugate_thr0.40_temp8.0_lr0.005_dp0.3_gmm2
python main_gmm3_reviewed1.py --dataset arxiv --backbone gcn --weight_decay 0.0005 --tau 1 --dropout 0.2 --env_type node --variant --store --edge_feat_mode mul --gmm_sample_k 3 --edge_blend 0.2 --edge_score_temp 5.0 --dag_latent_dim 16 --lambda_dag 0.05 --lambda_dag_label 0.05 --lambda_spu 0.05 --lambda_env 0.05 --lambda_fd 0.5 --fd_blend 0.5
python main_gmm3_reviewed1.py --dataset twitch --backbone gcn --weight_decay 5e-5 --tau 3 --dropout 0 --env_type graph --store --edge_feat_mode mul --gmm_sample_k 3 --edge_blend 0.2 --edge_score_temp 5.0 --dag_latent_dim 16 --lambda_dag 0.05 --lambda_dag_label 0.05 --lambda_spu 0.05 --lambda_env 0.05 --lambda_fd 0.5 --fd_blend 0.5 --epochs 300
python main_gmm3_reviewed1.py --dataset elliptic --backbone gcn --weight_decay 0.001 --tau 1 --K 3 --dropout 0.2 --env_type node --variant --num_layers 3 --hidden_channels 32 --store --edge_feat_mode mul --gmm_sample_k 3 --edge_blend 0.2 --edge_score_temp 5.0 --dag_latent_dim 16 --lambda_dag 0.05 --lambda_dag_label 0.05 --lambda_spu 0.05 --lambda_env 0.05 --lambda_fd 0.5 --fd_blend 0.5

# GAT backbone
python main_gmm3_reviewed1.py --dataset cora --backbone gat --weight_decay 0 --tau 3 --dropout 0.2 --env_type graph --combine_result --store --edge_feat_mode mul --gmm_sample_k 3 --edge_blend 0.2 --edge_score_temp 5.0 --dag_latent_dim 16 --lambda_dag 0.05 --lambda_dag_label 0.05 --lambda_spu 0.05 --lambda_env 0.05 --lambda_fd 0.5 --fd_blend 0.5
python main_gmm3_reviewed1.py --dataset citeseer --backbone gat --weight_decay 0 --tau 3 --dropout 0.2 --env_type graph --combine_result --store --edge_feat_mode mul --gmm_sample_k 3 --edge_blend 0.2 --edge_score_temp 5.0 --dag_latent_dim 16 —lambda_dag 0.05 —lambda_dag_label 0.05 —lambda_spu 0.05 —lambda_env 0.05 —lambda_fd 0.5 —fd_blend 0.5 —hidden_channels 32
python main_gmm3_reviewed1.py -—dataset pubmed —backbone gat —weight_decay 5e-5 —tau 1 —dropout 
python main_gmm3_reviewed1.py --dataset arxiv --backbone gat --weight_decay 5e-5 --tau 2 --dropout 0.2 --env_type graph --store --edge_feat_mode mul --gmm_sample_k 3 --edge_blend 0.2 --edge_score_temp 5.0 --dag_latent_dim 16 --lambda_dag 0.05 --lambda_dag_label 0.05 --lambda_spu 0.05 --lambda_env 0.05 --lambda_fd 0.5 --fd_blend 0.5
python main_gmm3_reviewed1.py --dataset twitch --backbone gat --weight_decay 5e-5 --tau 2 --dropout 0 --env_type graph --store --edge_feat_mode mul --gmm_sample_k 3 --edge_blend 0.2 --edge_score_temp 5.0 --dag_latent_dim 16 --lambda_dag 0.05 --lambda_dag_label 0.05 --lambda_spu 0.05 --lambda_env 0.05 --lambda_fd 0.5 --fd_blend 0.5 --epochs 300
python main_gmm3_reviewed1.py --dataset elliptic --backbone gat --weight_decay 0.0005 --tau 2 --dropout 0.1 --env_type graph --store --edge_feat_mode mul --gmm_sample_k 3 --edge_blend 0.2 --edge_score_temp 5.0 --dag_latent_dim 16 --lambda_dag 0.05 --lambda_dag_label 0.05 --lambda_spu 0.05 --lambda_env 0.05 --lambda_fd 0.5 --fd_blend 0.5


#cora / citeseer / pubmed 训练环境数3，总数量6
#arxiv 训练环境数3，总数量7
#twitch 训练环境数3，总环境数
#elliptic 训练环境数5，总环境数49

#arkiv gcn时用该指令效果较好：python main_gmm3.py --dataset arxiv --backbone gcn --weight_decay 0.0005 --tau 1 --dropout 0.2 --env_type node --variant --store --edge_feat_mode mul --gmm_sample_k 3 --edge_blend 0.2 --edge_score_temp 5.0 --dag_latent_dim 16 --lambda_dag 0.05 --lambda_dag_label 0.05 --lambda_spu 0.05 --lambda_env 0.05 --lambda_fd 0.5 --fd_blend 0.5
#结果为ood test1:64.49+-0.32 ,ood 2:61.44+-0.56,ood3:59.82 +-0.30