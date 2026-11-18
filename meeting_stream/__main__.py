"""Start the local meeting-to-AG-UI WebSocket service."""
import argparse
import uvicorn
from receiver_kv_workflow.engine import ReceiverKVConfig
from meeting_stream.service import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    from receiver_kv_workflow.cli import add_workflow_arguments, model_config
    add_workflow_arguments(parser)
    args = parser.parse_args()
    config = model_config(args)
    uvicorn.run(create_app(config), host=args.host, port=args.port)

if __name__ == "__main__":
    main()
