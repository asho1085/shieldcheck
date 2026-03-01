from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse, urlunparse

SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd", "ow.ly", "buff.ly",
    "cutt.ly", "rb.gy", "tiny.cc", "soo.gd"
}


def normalize_url(raw: str) -> str:
    """
    Normalize user input into a safe, valid URL.
    Handles common mistakes like:
    - "https:youtube.com" -> "https://youtube.com"
    - "youtube.com" -> "https://youtube.com"
    Returns "" if URL is invalid.
    """
    raw = (raw or "").strip()
    if not raw:
        return ""

    # Fix: "https:example.com" or "http:example.com"
    raw = re.sub(r"^(https?):(?!//)", r"\1://", raw, flags=re.I)

    # Remove spaces inside
    raw = re.sub(r"\s+", "", raw)

    # Add scheme if missing
    if not re.match(r"^https?://", raw, re.I):
        raw = "https://" + raw

    # Parse safely (port parsing can throw ValueError)
    try:
        p = urlparse(raw)
        _ = p.port  # may raise ValueError if malformed
    except Exception:
        return ""

    if not p.hostname:
        return ""

    # Keep original path/query/fragment
    return urlunparse((p.scheme, p.netloc, p.path, p.params, p.query, p.fragment))


def get_host(url: str) -> str:
    try:
        p = urlparse(normalize_url(url))
        return (p.hostname or "").lower()
    except Exception:
        return ""


def is_ip_host(host: str) -> bool:
    if not host:
        return False
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def is_ip_url(url: str) -> bool:
    return is_ip_host(get_host(url))


def has_at_symbol(url: str) -> bool:
    return "@" in (url or "")


def is_shortener(url: str) -> bool:
    return get_host(url) in SHORTENERS


def extract_features(url: str, columns: list[str]) -> list[int]:
    """
    Return features in EXACT training column order.
    Values follow dataset style: 1 (legit), 0 (suspicious), -1 (phishing)
    """
    url_norm = normalize_url(url)
    if not url_norm:
        # If invalid URL, return neutral features (or you can raise)
        return [0 for _ in columns]

    parsed = urlparse(url_norm)
    host = (parsed.hostname or "").lower()

    # everything after scheme://
    after_proto = url_norm.split("://", 1)[-1]

    feats: dict[str, int] = {}

    # IP address in hostname
    feats["having_IPhaving_IP_Address"] = -1 if is_ip_host(host) else 1

    # URL length
    L = len(url_norm)
    if L < 54:
        feats["URLURL_Length"] = 1
    elif L <= 75:
        feats["URLURL_Length"] = 0
    else:
        feats["URLURL_Length"] = -1

    # Shortener
    feats["Shortining_Service"] = -1 if host in SHORTENERS else 1

    # @ symbol
    feats["having_At_Symbol"] = -1 if "@" in url_norm else 1

    # double slash redirecting (only AFTER domain, not normal "https://")
    # Example phishing pattern: https://site.com//redirect/...
    feats["double_slash_redirecting"] = -1 if "//" in parsed.path else 1

    # Prefix-Suffix in domain (hyphen)
    feats["Prefix_Suffix"] = -1 if "-" in host else 1

    # subdomain count
    if is_ip_host(host):
        feats["having_Sub_Domain"] = 1
    else:
        dot_count = host.count(".")
        feats["having_Sub_Domain"] = 1 if dot_count <= 1 else (0 if dot_count == 2 else -1)

    # SSL final state
    feats["SSLfinal_State"] = 1 if parsed.scheme == "https" else -1

    # not implemented: neutral
    feats["Domain_registeration_length"] = 0
    feats["Favicon"] = 0

    # Port feature (safe)
    if parsed.port is None:
        feats["port"] = 1
    else:
        is_standard = (parsed.scheme == "http" and parsed.port == 80) or (parsed.scheme == "https" and parsed.port == 443)
        feats["port"] = 1 if is_standard else -1

    # HTTPS token trick: "https" inside domain/path while scheme is not https
    feats["HTTPS_token"] = -1 if ("https" in after_proto.lower() and parsed.scheme != "https") else 1

    html_based = [
        "Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH",
        "Submitting_to_email", "Abnormal_URL", "Redirect",
        "on_mouseover", "RightClick", "popUpWidnow", "Iframe"
    ]
    for k in html_based:
        feats[k] = 0

    external = [
        "age_of_domain", "DNSRecord", "web_traffic", "Page_Rank",
        "Google_Index", "Links_pointing_to_page", "Statistical_report"
    ]
    for k in external:
        feats[k] = 0

    # Output in training column order
    out: list[int] = []
    for col in columns:
        key = col.strip()
        out.append(int(feats.get(key, 0)))

    return out