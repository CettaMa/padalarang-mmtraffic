import re
import secrets
from urllib.parse import urlsplit, urlunsplit

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from cctv_scraper.dashboard.settings import DashboardConfig

# Camera URLs are operator credentials (rotating ``memfs`` tokens), so nothing
# that leaves this process may carry one verbatim -- not the status rows, and
# not the FFmpeg or requests error strings that quote the URL back at us.
MASK = "****"
MIN_TOKEN_LENGTH = 12

_URL_PATTERN = re.compile(r"https?://[^\s\"'<>()\[\],;]+", re.IGNORECASE)
_PATH_PATTERN = re.compile(r"(?<![\w.~%-])(?:/[A-Za-z0-9._~%-]+)+")

_basic_scheme = HTTPBasic(auto_error=False)


def _mask_segment(segment: str) -> str:
    """Mask a path segment that is long enough to be a token, keeping its shape."""
    stem, dot, extension = segment.partition(".")
    if len(stem) < MIN_TOKEN_LENGTH:
        return segment
    return f"{MASK}{stem[-4:]}{dot}{extension}"


def _mask_path(path: str) -> str:
    return "/".join(_mask_segment(part) for part in path.split("/"))


def mask_url(url: str) -> str:
    """Keep scheme, host, and path shape; drop the token and the whole query."""
    text = (url or "").strip()
    if not text:
        return ""
    try:
        parts = urlsplit(text)
    except ValueError:
        return MASK
    if not parts.scheme or not parts.netloc:
        return _mask_path(text)
    query = MASK if parts.query else ""
    return urlunsplit((parts.scheme, parts.netloc, _mask_path(parts.path), query, ""))


def scrub_text(text: str) -> str:
    """Remove camera tokens from free-form text such as error reasons and logs."""
    if not text:
        return ""
    scrubbed = _URL_PATTERN.sub(lambda match: mask_url(match.group(0)), text)
    return _PATH_PATTERN.sub(lambda match: _mask_path(match.group(0)), scrubbed)


def build_auth_dependency(config: DashboardConfig):
    """Return a FastAPI dependency enforcing Basic Auth, or a no-op when disabled."""
    if not config.auth_enabled:

        async def allow_all() -> None:
            return None

        return allow_all

    async def verify(
        credentials: HTTPBasicCredentials | None = Depends(_basic_scheme),  # noqa: B008
    ) -> None:
        unauthorized = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": 'Basic realm="cctv-dashboard"'},
        )
        if credentials is None:
            raise unauthorized
        # Compare both fields unconditionally so the response time does not
        # reveal whether the username alone was correct.
        username_ok = secrets.compare_digest(
            credentials.username.encode("utf-8"), config.username.encode("utf-8")
        )
        password_ok = secrets.compare_digest(
            credentials.password.encode("utf-8"), config.password.encode("utf-8")
        )
        if not (username_ok and password_ok):
            raise unauthorized
        return None

    return verify
