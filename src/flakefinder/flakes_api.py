"""Client helpers for flakes.sharpelab.science."""

import gzip
import json
import os
from dataclasses import dataclass
from pathlib import Path

import requests

BASE_URL = "https://flakes.sharpelab.science"


@dataclass
class FlakesAuth:
    user: str
    password: str

    def as_tuple(self) -> tuple[str, str]:
        return (self.user, self.password)


def get_auth() -> FlakesAuth:
    """Load flakes.sharpelab.science credentials from environment.

    Reads FLAKES_USER and FLAKES_PASSWORD. Raises RuntimeError if either
    is missing — set them in .env or your shell environment.
    """
    user = os.environ.get("FLAKES_USER")
    password = os.environ.get("FLAKES_PASSWORD")
    if not user or not password:
        raise RuntimeError(
            "FLAKES_USER and FLAKES_PASSWORD must be set. Copy .env.example to .env and fill in credentials."
        )
    return FlakesAuth(user=user, password=password)


def api_get(path: str, params: dict | None = None) -> list | dict:
    """GET from the API, handling gzip-compressed JSON responses."""
    auth = get_auth().as_tuple()
    r = requests.get(f"{BASE_URL}/api/{path}", params=params, auth=auth)
    r.raise_for_status()
    try:
        data = gzip.decompress(r.content)
        return json.loads(data)
    except gzip.BadGzipFile:
        return r.json()


def download_image(url: str, dest: Path) -> int | None:
    """Download a single image. Returns size in bytes, or None if 404."""
    auth = get_auth().as_tuple()
    r = requests.get(url, auth=auth, stream=True)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest.stat().st_size
