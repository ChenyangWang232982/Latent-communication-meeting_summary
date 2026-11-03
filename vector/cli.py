"""Shared switches for inference and benchmark entry points."""


def add_arguments(parser):
    parser.add_argument('--vector-routing', action='store_true', help='Enable semantic-assisted speaker block ranking')
    parser.add_argument('--embedding-model', default='models/multilingual-e5-small', help='Local E5 model directory or Hugging Face model ID')
    parser.add_argument('--embedding-device', default='cpu')
    parser.add_argument('--vector-weight', type=float, default=0.3, help='Semantic weight in [0,1]; experimental, not tuned')


def config_kwargs(args):
    return {key: getattr(args, key) for key in
            ('vector_routing', 'embedding_model', 'embedding_device', 'vector_weight')}
