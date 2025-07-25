import torch as th


def add_default_args(parser):
    parser.add_argument('--t_max', type=int, default=40000)
    parser.add_argument('--t_start', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda" if th.cuda.is_available() else "cpu")
    parser.add_argument('--test_nepisode', type=int, default=10)
    parser.add_argument('--test_interval', type=int, default=2000)
    parser.add_argument('--load_models', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--buffers_path', type=str, default='')
    parser.add_argument('--offline_models_path', type=str, default='')
    parser.add_argument('--test_models', type=bool, default=False)
    parser.add_argument('--train_models', type=bool, default=True)
    parser.add_argument('--save_models', type=bool, default=False)
    parser.add_argument('--save_buffers', type=bool, default=False)
    parser.add_argument('--log_tag', type=str, default='default_log_tag')
    parser.add_argument('--folder', type=str, default='default_folder')
    parser.add_argument('--offline', default=False)
    parser.add_argument('--expert', type=bool, default=False)

    return parser
