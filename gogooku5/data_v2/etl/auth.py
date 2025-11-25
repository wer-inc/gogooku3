"""J-Quants authentication utilities."""

from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

from .config import AUTH_URL, PROJECT_ROOT, REFRESH_URL


def load_env() -> None:
    """Load .env from repo root and data_v2."""

    load_dotenv(dotenv_path=PROJECT_ROOT.parent / ".env")
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


def get_id_token() -> str:
    """Obtain idToken using refresh token or email/password."""

    load_env()
    mail = os.environ.get("JQUANTS_AUTH_EMAIL")
    password = os.environ.get("JQUANTS_AUTH_PASSWORD")
    refresh_token_env = os.environ.get("JQUANTS_REFRESH_TOKEN")

    if refresh_token_env:
        refresh_token = refresh_token_env
    else:
        if not mail or not password:
            raise RuntimeError("JQUANTS_AUTH_EMAIL and JQUANTS_AUTH_PASSWORD are required")
        resp = requests.post(AUTH_URL, json={"mailaddress": mail, "password": password}, timeout=30)
        resp.raise_for_status()
        refresh_token = resp.json().get("refreshToken")
        if not refresh_token:
            raise RuntimeError("Failed to obtain refreshToken from auth_user response")

    resp = requests.post(REFRESH_URL, params={"refreshtoken": refresh_token}, timeout=30)
    if resp.status_code == 403:
        raise RuntimeError("refreshToken is invalid or expired (403). Please refresh it.")
    resp.raise_for_status()
    id_token = resp.json().get("idToken")
    if not id_token:
        raise RuntimeError("Failed to obtain idToken from auth_refresh response")
    return id_token
