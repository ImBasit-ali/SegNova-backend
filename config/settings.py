"""
Django settings for BraTS AI backend.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from config.env import env


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

# Model assets (always local in the backend repo on Render)
MODEL_KERAS_PATH = Path(
    os.environ.get('MODEL_KERAS_PATH', str(BASE_DIR / 'model' / 'model.keras'))
)
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.keras')

SECRET_KEY = env('DJANGO_SECRET_KEY', 'dev-secret-key-change-in-production-!@#$%')

DEBUG = env('DEBUG', 'True').lower() in ('1', 'true', 'yes')

_allowed = env('ALLOWED_HOSTS', 'localhost,127.0.0.1,.onrender.com')
ALLOWED_HOSTS = [h.strip() for h in _allowed.split(',') if h.strip()]

_frontend_origins = env(
    'FRONTEND_URL',
    'http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001',
)
CORS_ALLOWED_ORIGINS = [o.strip() for o in _frontend_origins.split(',') if o.strip()]
CORS_ALLOW_ALL_ORIGINS = env('CORS_ALLOW_ALL_ORIGINS', 'False').lower() in ('1', 'true', 'yes')

CSRF_TRUSTED_ORIGINS = CORS_ALLOWED_ORIGINS.copy()

# Storage: Supabase in production, local media/ in development
USE_SUPABASE_STORAGE = env('USE_SUPABASE_STORAGE', 'False').lower() in ('1', 'true', 'yes')

SUPABASE_URL = env('SUPABASE_URL', '')
SUPABASE_SERVICE_ROLE_KEY = env('SUPABASE_SERVICE_ROLE_KEY', '')
SUPABASE_BUCKET = env('SUPABASE_BUCKET', 'brain-mri')
SUPABASE_PUBLIC_URL = env('SUPABASE_PUBLIC_URL', '')

# Optional absolute base for /media URLs when serving locally behind a public host
PUBLIC_MEDIA_BASE_URL = env('PUBLIC_MEDIA_BASE_URL', '')

# Scratch space for downloads + model inference (ephemeral on Render)
TEMP_MEDIA_ROOT = Path(os.environ.get('TEMP_MEDIA_ROOT', str(BASE_DIR / 'tmp' / 'sessions')))
TEMP_MEDIA_ROOT.mkdir(parents=True, exist_ok=True)


def _ensure_dir(path_value):
    Path(path_value).mkdir(parents=True, exist_ok=True)
    return Path(path_value)


# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'segmentation',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

MEDIA_URL = '/media/'
MEDIA_ROOT = _ensure_dir(BASE_DIR / 'media')

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.MultiPartParser',
        'rest_framework.parsers.FormParser',
    ],
}

DATA_UPLOAD_MAX_MEMORY_SIZE = 524288000
FILE_UPLOAD_MAX_MEMORY_SIZE = 524288000

FILE_UPLOAD_HANDLERS = [
    'django.core.files.uploadhandler.TemporaryFileUploadHandler',
    'django.core.files.uploadhandler.MemoryFileUploadHandler',
]

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '[{asctime}] {levelname} {name}: {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'segmentation': {
            'handlers': ['console'],
            'level': 'INFO',
        },
    },
}
