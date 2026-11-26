"""Fail-closed local approval authentication using configured role tokens and TOTP MFA."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import struct
import time


class AuthorizationError(ValueError):
    """Approval authentication or role authorization failed."""


def _role_tokens() -> dict:
    raw = os.environ.get("BLOCKCHAIN_RAG_ROLE_TOKENS", "")
    if not raw:
        raise AuthorizationError(
            "Human approvals are disabled. Configure BLOCKCHAIN_RAG_ROLE_TOKENS and "
            "BLOCKCHAIN_RAG_MFA_SECRET before using approval actions."
        )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AuthorizationError("BLOCKCHAIN_RAG_ROLE_TOKENS must be a JSON object.") from exc
    if not isinstance(data, dict) or not data or any(not isinstance(v, str) or len(v) < 24 for v in data.values()):
        raise AuthorizationError("BLOCKCHAIN_RAG_ROLE_TOKENS must map role names to tokens of at least 24 characters.")
    return {str(role).strip().lower(): token for role, token in data.items()}


def _totp(secret: str, timestamp: int) -> str:
    normalized = secret.replace(" ", "").upper()
    normalized += "=" * ((8 - len(normalized) % 8) % 8)
    try:
        key = base64.b32decode(normalized, casefold=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise AuthorizationError("BLOCKCHAIN_RAG_MFA_SECRET is not valid Base32.") from exc
    if len(key) < 20:
        raise AuthorizationError("BLOCKCHAIN_RAG_MFA_SECRET must decode to at least 20 bytes.")
    counter = int(timestamp // 30)
    digest = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    number = (struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF) % 1_000_000
    return f"{number:06d}"


def authenticate_approver(token: str, mfa_code: str) -> str:
    tokens = _role_tokens()
    matched_role = None
    for role, expected in tokens.items():
        if hmac.compare_digest(token or "", expected):
            matched_role = role
    if not matched_role:
        raise AuthorizationError("Invalid reviewer credential.")
    secret = os.environ.get("BLOCKCHAIN_RAG_MFA_SECRET", "")
    if not secret:
        raise AuthorizationError("Approvals require BLOCKCHAIN_RAG_MFA_SECRET to be configured.")
    now = time.time()
    if not any(hmac.compare_digest(str(mfa_code or ""), _totp(secret, now + offset * 30)) for offset in (-1, 0, 1)):
        raise AuthorizationError("Invalid or expired MFA code.")
    return matched_role


APPROVAL_ROLES = {
    "requirement": {"business_analyst", "product_owner", "software_architect"},
    "security": {"security_officer"},
    "compliance": {"compliance_officer"},
    "sdlc": {"project_manager", "software_architect", "security_officer", "compliance_officer"},
    "production": {"release_manager", "security_officer", "compliance_officer"},
}


def authorize_approval(token: str, mfa_code: str, approval_type: str) -> str:
    role = authenticate_approver(token, mfa_code)
    allowed = APPROVAL_ROLES.get(approval_type)
    if not allowed:
        raise AuthorizationError("Unknown approval type.")
    if role not in allowed:
        raise AuthorizationError(f"Role '{role}' is not authorized for {approval_type} approval.")
    return role
