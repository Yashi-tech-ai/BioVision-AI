#!/usr/bin/env python3
"""Run FastAPI server. Usage: python api/run_api.py --checkpoint checkpoints/classification/best.pt"""

import argparse
import os
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/classification/best.pt")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.checkpoint:
        os.environ["BIOVISION_CHECKPOINT"] = args.checkpoint
    import uvicorn
    from api.app import app
    uvicorn.run(app, host=args.host, port=args.port)
