"""Start the local meeting-to-AG-UI WebSocket service."""
import argparse
import uvicorn
from receiver_kv_workflow.engine import ReceiverKVConfig
from meeting_stream.service import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--role-budget-ratio", type=float, default=0.25)
    parser.add_argument("--speaker-neighbor-blocks", type=int, default=1)
    parser.add_argument("--vector-routing", action="store_true")
    parser.add_argument("--embedding-model", default="models/Qwen3-Embedding-0.6B")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--review-seed", type=int, default=7)
    parser.add_argument("--review-temperature", type=float, default=0.0)
    parser.add_argument("--review-top-p", type=float, default=0.9)
    args = parser.parse_args()
    config = ReceiverKVConfig(model=args.model, device=args.device, role_budget_ratio=args.role_budget_ratio,
        speaker_neighbor_blocks=args.speaker_neighbor_blocks, vector_routing=args.vector_routing,
        embedding_model=args.embedding_model, seed=args.seed, review_seed=args.review_seed,
        review_temperature=args.review_temperature, review_top_p=args.review_top_p)
    uvicorn.run(create_app(config), host=args.host, port=args.port)

if __name__ == "__main__":
    main()
