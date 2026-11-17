"""python -m grounded_specialist_workflow --transcript ... --output ..."""
import argparse
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transcript', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='Qwen/Qwen3-8B')
    parser.add_argument('--embedding-model', default='models/Qwen3-Embedding-0.6B')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--topic-budget-ratio', type=float, default=0.25)
    parser.add_argument('--speaker-neighbor-blocks', type=int, default=1)
    parser.add_argument('--vector-weight', type=float, default=0.3)
    parser.add_argument('--specialist-workers', type=int, default=1,
                        help='Concurrent topic tasks sharing one model; independent KV/RNG per task')
    parser.add_argument('--routing-recency-weight', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--repetition-penalty', type=float, default=1.1)
    args = parser.parse_args()
    from receiver_kv_workflow.engine import ReceiverKVConfig
    from .components.pipeline import GroundedSpecialistRunner, SpecialistConfig
    specialists = SpecialistConfig(args.specialist_workers, args.routing_recency_weight)
    config = ReceiverKVConfig(model=args.model, device=args.device, role_budget_ratio=args.topic_budget_ratio,
        speaker_neighbor_blocks=args.speaker_neighbor_blocks, vector_routing=True,
        embedding_model=args.embedding_model, vector_weight=args.vector_weight, seed=args.seed,
        temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k, repetition_penalty=args.repetition_penalty)
    transcript = args.transcript.read_text(encoding='utf-8-sig')
    def log(message):
        print(f'[{datetime.now():%H:%M:%S}] {message}', flush=True)
    runner = GroundedSpecialistRunner(config, specialists, progress=log)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    artifacts = args.output.parent / (args.output.stem + '.artifacts') / stamp
    log(f'Artifacts: {artifacts}')
    result = runner.run(args.transcript.stem, transcript, artifact_dir=artifacts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result['summary'] + '\n', encoding='utf-8')
    log(f'Summary: {args.output}')


if __name__ == '__main__':
    main()
