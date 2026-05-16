from __future__ import annotations

import os


def get_tushare_token(token: str | None = None) -> str:
    token = token or os.getenv("TUSHARE_TOKEN")
    if not token:
        raise ValueError("Missing Tushare token. Set $env:TUSHARE_TOKEN or pass --token.")
    return token


def init_tushare(token: str | None = None, http_url: str | None = None):
    """Initialize the unified Tushare client used by all project data fetchers.

    Equivalent calling style:

        import tushare as ts
        pro = ts.pro_api(token)
        pro._DataApi__http_url = "http://your-proxy:port/"  # optional

    Token is intentionally read from TUSHARE_TOKEN by default instead of being
    stored in source code.
    """
    try:
        import tushare as ts
    except ImportError as exc:
        raise ImportError("tushare is not installed. Run: pip install tushare") from exc

    pro = ts.pro_api(get_tushare_token(token))
    resolved_http_url = http_url or os.getenv("TUSHARE_HTTP_URL")
    if resolved_http_url:
        pro._DataApi__http_url = resolved_http_url
    return pro


def pro_bar(*args, token: str | None = None, http_url: str | None = None, api=None, **kwargs):
    try:
        import tushare as ts
    except ImportError as exc:
        raise ImportError("tushare is not installed. Run: pip install tushare") from exc

    api = api or init_tushare(token=token, http_url=http_url)
    return ts.pro_bar(*args, api=api, **kwargs)

