"""Thin wrapper to invoke modal CLI from a sandboxed env using the
HTTP proxy at localhost:3128 for api.modal.com traffic."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._dns_patch import patch
patch()

if __name__ == "__main__":
    # Drop proxy-env so modal's sub-processes don't try to apply them
    # confusingly. (our patch is in-process only.)
    import os
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
              "grpc_proxy", "GRPC_PROXY", "ftp_proxy"):
        os.environ.pop(k, None)

    # Use modal's typer CLI
    from modal.__main__ import entrypoint_cli
    sys.argv = ["modal"] + sys.argv[1:]
    entrypoint_cli()
