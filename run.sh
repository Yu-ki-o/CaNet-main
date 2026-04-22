# GCN backbone
python main.py --dataset cora --backbone gcn --weight_decay 5e-5 --tau 1 --dropout 0.2 --env_type graph --combine_result --store
python main.py --dataset citeseer --backbone gcn --weight_decay 5e-5 --tau 1 --dropout 0.1 --env_type graph --combine_result --store
python main.py --dataset pubmed --backbone gcn --weight_decay 5e-5 --tau 2 --dropout 0.2 --env_type graph --combine_result --store
python main.py --dataset arxiv --backbone gcn --weight_decay 0.0005 --tau 1 --dropout 0.2 --env_type node --variant --store
python main.py --dataset twitch --backbone gcn --weight_decay 5e-5 --tau 3 --dropout 0 --env_type graph --store
python main.py --dataset elliptic --backbone gcn --weight_decay 0.001 --tau 1 --K 3 --dropout 0.2 --env_type node --variant --num_layers 3 --hidden_channels 32 --store

# GAT backbone
python main.py --dataset cora --backbone gat --weight_decay 0 --tau 3 --dropout 0.2 --env_type graph --combine_result --store
python main.py --dataset citeseer --backbone gat --weight_decay 0 --tau 3 --dropout 0.2 --env_type graph --combine_result --store
python main.py --dataset pubmed --backbone gat --weight_decay 5e-5 --tau 1 --dropout 0.2 --env_type graph --combine_result --store
python main.py --dataset arxiv --backbone gat --weight_decay 5e-5 --tau 2 --dropout 0.2 --env_type graph --store
python main.py --dataset twitch --backbone gat --weight_decay 5e-5 --tau 2 --dropout 0 --env_type graph --store
python main.py --dataset elliptic --backbone gat --weight_decay 0.0005 --tau 2 --dropout 0.1 --env_type graph --store  


#cora / citeseer / pubmed 训练环境数3，总数量6
#arxiv 训练环境数3，总数量7
#twitch 训练环境数3，总环境数
#elliptic 训练环境数5，总环境数49