"""
Sandbox network shim for Modal:
  - The sandbox blocks direct DNS + direct TCP to public IPs.
  - An HTTP proxy at localhost:3128 is allowed and can CONNECT to
    api.modal.com.
  - Modal's pure-Python gRPC client (grpclib) calls
    ``loop.create_connection(host=..., port=...)`` directly, so we
    monkey-patch that to route *.modal.com traffic through the proxy.
"""

import asyncio
import socket
import ssl as _ssl
from typing import Optional


PROXY_HOST = "localhost"
PROXY_PORT = 3128


def _is_modal_host(host: str) -> bool:
    return isinstance(host, str) and (host == "modal.com" or host.endswith(".modal.com"))


def _proxy_connect_socket(target_host: str, target_port: int) -> socket.socket:
    """Open a TCP socket to localhost:3128, send an HTTP CONNECT request to
    tunnel to target_host:target_port, and return the tunneled socket."""
    s = socket.create_connection((PROXY_HOST, PROXY_PORT), timeout=15)
    req = (
        f"CONNECT {target_host}:{target_port} HTTP/1.1\r\n"
        f"Host: {target_host}:{target_port}\r\n"
        f"Proxy-Connection: Keep-Alive\r\n"
        f"\r\n"
    ).encode()
    s.sendall(req)
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = s.recv(4096)
        if not chunk:
            s.close()
            raise OSError("proxy closed before CONNECT response")
        buf += chunk
    status_line = buf.split(b"\r\n", 1)[0].decode(errors="replace")
    if not status_line.startswith("HTTP/1.1 200") and not status_line.startswith("HTTP/1.0 200"):
        s.close()
        raise OSError(f"proxy CONNECT failed: {status_line!r}")
    s.setblocking(False)
    return s


def patch():
    """Install the shim. Idempotent."""
    loop_cls = asyncio.SelectorEventLoop
    orig_create_connection = loop_cls.create_connection

    async def new_create_connection(self, protocol_factory, host=None, port=None, *, ssl=None, **kwargs):
        if host is not None and _is_modal_host(host):
            # Open proxy-tunneled socket in a thread so we don't block the loop.
            raw = await asyncio.get_running_loop().run_in_executor(
                None, _proxy_connect_socket, host, port
            )
            # Wrap with asyncio using the existing sock; apply TLS if needed.
            tls_hostname = kwargs.pop("server_hostname", None) or host
            conn_kwargs = {k: v for k, v in kwargs.items()
                           if k in ("server_hostname", "ssl_handshake_timeout")}
            return await orig_create_connection(
                self,
                protocol_factory,
                sock=raw,
                ssl=ssl,
                server_hostname=tls_hostname if ssl else None,
                **conn_kwargs,
            )
        return await orig_create_connection(
            self, protocol_factory, host=host, port=port, ssl=ssl, **kwargs
        )

    loop_cls.create_connection = new_create_connection  # type: ignore[assignment]

    # Also patch socket.getaddrinfo to return a dummy so any code that
    # calls it for modal hosts doesn't blow up.
    _orig_getaddrinfo = socket.getaddrinfo
    _orig_gethostbyname = socket.gethostbyname

    def _getaddrinfo(host, port, *a, **kw):
        if _is_modal_host(host):
            p = port if isinstance(port, int) else (443 if port in (None, "https") else 80)
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", p))]
        return _orig_getaddrinfo(host, port, *a, **kw)

    def _gethostbyname(host):
        if _is_modal_host(host):
            return "127.0.0.1"
        return _orig_gethostbyname(host)

    socket.getaddrinfo = _getaddrinfo
    socket.gethostbyname = _gethostbyname
