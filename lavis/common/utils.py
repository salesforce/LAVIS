from urllib.parse import urlparse


def now():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d%H%M")


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")
