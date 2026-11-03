"""Download embedding weights: python -m vector.download."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='intfloat/multilingual-e5-small')
    parser.add_argument('--output', default='models/multilingual-e5-small')
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    path = snapshot_download(args.model, local_dir=args.output,
                             allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt'],
                             ignore_patterns=['onnx/*', 'openvino/*'])
    print(f'Embedding model: {path}')


if __name__ == '__main__':
    main()
