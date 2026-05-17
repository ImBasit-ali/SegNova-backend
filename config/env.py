"""
Resolve environment variables from a single .env file.

Set APP_ENV=local or APP_ENV=production. Only LOCAL_* or PROD_* keys for the
active mode are used; the other prefix is ignored. Unprefixed keys still work
for Render/Vercel dashboard overrides.
"""

import os

APP_ENV = os.environ.get('APP_ENV', 'local').strip().lower()
_IS_LOCAL = APP_ENV in ('local', 'development', 'dev')


def env(name: str, default: str = '') -> str:
    prefix = 'LOCAL_' if _IS_LOCAL else 'PROD_'
    prefixed = os.environ.get(f'{prefix}{name}')
    if prefixed is not None and str(prefixed).strip() != '':
        return str(prefixed).strip()
    fallback = os.environ.get(name)
    if fallback is not None and str(fallback).strip() != '':
        return str(fallback).strip()
    return default
