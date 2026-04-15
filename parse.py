def parser_add_main_args(parser):
    # setup and protocol
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--data_dir', type=str,
                        default='./data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=2024) #123#2024#1000#42#3407#777
    parser.add_argument('--runs', type=int, default=5,
                        help='number of distinct runs')
    parser.add_argument('--epochs', type=int, default=500)

    # model network
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')

    # CaNet
    parser.add_argument('--backbone_type', type=str, default='gcn', choices=['gcn', 'gat','transformer'])
    parser.add_argument('--K', type=int, default=3,
                        help='num of domains, each for one graph convolution filter')
    parser.add_argument('--tau', type=float, default=1,
                        help='temperature for Gumbel Softmax')
    parser.add_argument('--env_type', type=str, default='node', choices=['node', 'graph','transformer',"local_global","pure_vn","combined_vn","cross_align"])
    parser.add_argument('--lamda', type=float, default=1.0,
                        help='weight for regularlization')
    parser.add_argument('--variant', action='store_true',help='set to use variant')
    parser.add_argument('--other_env_reduce', type=str, default='sample', choices=['sample', 'env'],
                        help='how to aggregate losses from non-assigned environments')

    # training
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_bn', action='store_true', help='use batch norm')

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--store_result', action='store_true',
                        help='whether to store results')
    parser.add_argument('--combine_result', action='store_true',
                        help='whether to combine all the ood environments')
    
    #自己加的，方便分辨输出result
    parser.add_argument("--result_name",type=str,default='')
    parser.add_argument('--lamda_ciw', type=float, default=1e-4,
                        help='weight for regularlization')
    parser.add_argument('--lambda_l1', type=float, default=0.5,
                        help='weight for regularlization')
    parser.add_argument('--lambda_dag', type=float, default=0.1, help='DAG 无环约束的权重')
    parser.add_argument('--lambda_ind', type=float, default=0.1, help='因果特征独立性约束的权重')
    parser.add_argument('--lambda_cl', type=float, default=0.1, help='对比学习/一致性损失的权重')

    parser.add_argument('--lr_dag', type=float, default=0.001, help='DAG 矩阵 A 的专属学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='EMA 滑动平均的动量系数')
    parser.add_argument('--pos_weight', type=float, default=5.0, help='二分类数据集正样本权重')
