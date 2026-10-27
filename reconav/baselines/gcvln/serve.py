"""Serve the original GSAM2 with the paper's serialized model-call transport.

Run from GC-VLN's upstream root after installing its ModelServer and weights.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7012)
    args = parser.parse_args(argv)
    from ModelServer import start_server
    from ModelServer.web import server
    from ModelServer.models.GSAM2.GSAM2 import GSAM2

    # One mutable SAM2 predictor / GroundingDINO image cache must not be shared
    # between simultaneous complete calls. Inference logic and weights stay upstream.
    server.thread_pool.shutdown(wait=True)
    server.thread_pool = ThreadPoolExecutor(max_workers=1)
    start_server(GSAM2(), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
