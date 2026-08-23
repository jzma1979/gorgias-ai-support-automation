from __future__ import annotations

import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def env_list(name: str, default: list[str] | None = None) -> list[str]:
    raw_value = os.getenv(name)
    if not raw_value:
        return default or []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ImproperlyConfigured(f"{name} must be a number.") from exc


DJANGO_ENV = os.getenv("DJANGO_ENV", "development")
DEBUG = env_bool("DJANGO_DEBUG", default=False)

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")
if not SECRET_KEY:
    if DJANGO_ENV == "production":
        raise ImproperlyConfigured("DJANGO_SECRET_KEY is required in production.")
    SECRET_KEY = "django-insecure-local-development-only-change-me"

ALLOWED_HOSTS = env_list(
    "DJANGO_ALLOWED_HOSTS",
    default=["localhost", "127.0.0.1", "testserver"],
)

INSTALLED_APPS = [
    "support",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "gorgias-ai-support-automation",
    }
}

GORGIAS_BASE_URL = os.getenv("GORGIAS_BASE_URL", "")
GORGIAS_USERNAME = os.getenv("GORGIAS_USERNAME", "")
GORGIAS_API_KEY = os.getenv("GORGIAS_API_KEY", "")

AI_PROVIDER = os.getenv("AI_PROVIDER", "openrouter")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
)

EXTERNAL_REQUEST_TIMEOUT_SECONDS = env_float("EXTERNAL_REQUEST_TIMEOUT_SECONDS", 10.0)
WEBHOOK_IDEMPOTENCY_TTL_SECONDS = int(
    os.getenv("WEBHOOK_IDEMPOTENCY_TTL_SECONDS", "3600")
)
INTEGRATION_NAME = os.getenv("INTEGRATION_NAME", "Gorgias AI Support Automation")
GORGIAS_INTEGRATION_EMAIL = os.getenv("GORGIAS_INTEGRATION_EMAIL", "")
GORGIAS_INTEGRATION_USER_ID = os.getenv("GORGIAS_INTEGRATION_USER_ID", "")
