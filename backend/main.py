from __future__ import annotations

"""
FastAPI Backend for the Hybrid Recommender System — v3 (Supabase).
Integrates PostgreSQL full-text search, Supabase auth, and the improved hybrid model.
"""
import os
import re
import sys
import io
import time
import logging
import math
import secrets

try:
    import bleach
except ModuleNotFoundError:
    class bleach:
        @staticmethod
        def clean(value, strip=True):
            if not strip:
                return str(value)
            return re.sub(r"<[^>]*>", "", str(value))

from collections import deque, Counter
from threading import Lock
from datetime import datetime, timezone, timedelta

from collections import defaultdict

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import (
    FastAPI,
    Depends,
    Header,
    UploadFile,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)

from celery.result import AsyncResult
from celery_app import celery_app
from tasks import compute_recommendations


# backend/main.py — corrected imports
from src.data.db import get_supabase, get_supabase_admin
from src.data.data_adapter import adapt_data, read_file
from src.model.nlp_engine import batch_analyze, aggregate_sentiment_by_item
from src.model.content_model import ContentRecommender
from src.model.collaborative_model import CollaborativeRecommender
from src.model.hybrid_model import HybridRecommender
from src.model.issue_triage import triage_issue
from src.model.federated_learning import train_federated_collaborative_model

from functools import lru_cache

from backend.csrf import CSRFMiddleware, generate_csrf_token, set_csrf_cookie, CSRFTokenResponse


# ── OpenAPI CSRF header dependency ────────────────────────────────────
async def csrf_header_dep(
    x_csrf_token: str = Header(
        ...,
        alias="X-CSRF-Token",
        description=(
            "CSRF token obtained from **GET /api/csrf-token**. "
            "Required on all state-mutating requests (POST / PUT / PATCH / DELETE). "
            "Must match the value stored in the `csrftoken` cookie."
        ),
    ),
) -> None:
    """Declares X-CSRF-Token in OpenAPI. Enforcement is done by CSRFMiddleware."""
    pass

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="Hybrid Recommender API", version="3.0")

@app.on_event("startup")
def download_nltk_assets():
    """
    Ensures NLTK VADER assets are downloaded safely at startup
    to prevent multi-worker download race conditions.
    """
    try:
        SentimentIntensityAnalyzer()
        logger.info("NLTK VADER lexicon verified successfully.")
    except LookupError:
        logger.info("VADER lexicon missing. Downloading safely at startup...")
        nltk.download('vader_lexicon', quiet=True)
        logger.info("NLTK VADER lexicon downloaded successfully.")


RESPONSE_TIME_HEADER = "X-Response-Time-ms"
DEFAULT_SLOW_RESPONSE_THRESHOLD_MS = 1000.0
CACHE_TTL_SECONDS = 300
CACHE_CONTROL_VALUE = f"public, max-age={CACHE_TTL_SECONDS}"
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
MAX_SEARCH_QUERY_LENGTH = 120
_response_cache: dict = {}
ADMIN_API_TOKEN_ENV = "ADMIN_API_TOKEN"
_rate_limit_buckets: dict = {}
_rate_limit_lock = Lock()
_cache_lock = Lock()

MOCK_PRODUCTS = [
    {
        "id": 1,
        "title": "Acoustic Noise-Cancelling Headphones",
        "description": "Premium over-ear headphones with active noise cancellation.",
        "category": "Electronics",
        "rating": 4.8,
        "avg_sentiment": 0.85,
        "review_count": 245,
        "price": 1299,
    },
    {
        "id": 2,
        "title": "Ergonomic Mechanical Keyboard",
        "description": "Tactile switches, RGB backlighting, and a comfortable wrist rest.",
        "category": "Electronics",
        "rating": 4.5,
        "avg_sentiment": 0.65,
        "review_count": 189,
        "price": 799,
    },
    {
        "id": 3,
        "title": "Portable Fitness Tracker",
        "description": "Track heart rate, sleep, and workouts from your wrist.",
        "category": "Health",
        "rating": 4.2,
        "avg_sentiment": 0.42,
        "review_count": 128,
        "price": 499,
    },
]


def _get_slow_response_threshold_ms() -> float:
    try:
        return float(os.environ.get("RESPONSE_TIME_SLOW_MS", DEFAULT_SLOW_RESPONSE_THRESHOLD_MS))
    except ValueError:
        return DEFAULT_SLOW_RESPONSE_THRESHOLD_MS


def _cache_key(*parts: Any) -> str:
    return ":".join(str(part).strip().lower() for part in parts)


def _get_cached_response(key: str):
    with _cache_lock:
        cached = _response_cache.get(key)
        if not cached:
            return None
        expires_at, value = cached
        if expires_at <= time.time():
            _response_cache.pop(key, None)
            return None
        return value


def _set_cached_response(key: str, value: Any) -> None:
    with _cache_lock:
        _response_cache[key] = (time.time() + CACHE_TTL_SECONDS, value)


def _clear_response_cache() -> None:
    with _cache_lock:
        _response_cache.clear()


def _normalize_search_query(query: str) -> str:
    normalized = " ".join((query or "").split())
    if len(normalized) > MAX_SEARCH_QUERY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Search query must be {MAX_SEARCH_QUERY_LENGTH} characters or fewer.",
        )
    return normalized


def _escape_like_pattern(value: str) -> str:
    """Escape special LIKE metacharacters to prevent pattern injection."""
    return (
        value
        .replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


_USER_ID_RE = re.compile(r"^[a-zA-Z0-9_\-\.@]{1,128}$")


def _validate_user_id(user_id: str) -> str:
    """Allowlist-validate user_id to block injection via path parameters."""
    if not _USER_ID_RE.match(user_id):
        raise HTTPException(status_code=400, detail="Invalid user_id format.")
    return user_id


def _set_cache_headers(response: Response, status: str) -> None:
    response.headers["Cache-Control"] = CACHE_CONTROL_VALUE
    response.headers["X-Cache"] = status


def _get_rate_limit(limit_env: str, default_limit: int) -> int:
    try:
        limit = int(os.environ.get(limit_env, str(default_limit)))
    except ValueError:
        return default_limit
    return max(1, limit)


def _rate_limit_exceeded_response(rate_limit: int, reset_time: int) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "message": "Too many requests. Please try again later.",
        },
        headers={
            "x-ratelimit-limit": str(rate_limit),
            "x-ratelimit-remaining": "0",
            "x-ratelimit-reset": str(reset_time),
        },
    )


def _apply_rate_limit(
    request: Request,
    response: Response,
    scope: str,
    limit_env: str,
    default_limit: int,
) -> JSONResponse | None:
    rate_limit = _get_rate_limit(limit_env, default_limit)
    client_ip = request.client.host if request.client else "127.0.0.1"
    bucket_key = (scope, client_ip)
    now = time.time()

    with _rate_limit_lock:
        timestamps = _rate_limit_buckets.setdefault(bucket_key, [])
        timestamps[:] = [timestamp for timestamp in timestamps if now - timestamp < 60]

        reset_time = int(60 - (now - timestamps[0])) if timestamps else 60
        reset_time = max(0, reset_time)

        if len(timestamps) >= rate_limit:
            return _rate_limit_exceeded_response(rate_limit, reset_time)

        timestamps.append(now)
        remaining = rate_limit - len(timestamps)
        reset_time = int(60 - (now - timestamps[0])) if timestamps else 60
        reset_time = max(0, reset_time)

    response.headers["x-ratelimit-limit"] = str(rate_limit)
    response.headers["x-ratelimit-remaining"] = str(remaining)
    response.headers["x-ratelimit-reset"] = str(reset_time)
    return None


def _extract_bearer_token(value: str | None) -> str:
    if not value:
        return ""
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()


def _require_admin_access(request: Request) -> None:
    expected_token = os.environ.get(ADMIN_API_TOKEN_ENV, "").strip()
    if not expected_token:
        return

    provided_token = (
        request.headers.get("x-admin-token", "").strip()
        or _extract_bearer_token(request.headers.get("authorization"))
    )
    if not provided_token or not secrets.compare_digest(provided_token, expected_token):
        raise HTTPException(status_code=401, detail="Admin token required.")


# CORS
allowed_origins_env = os.environ.get("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "X-CSRF-Token"],
)

app.add_middleware(CSRFMiddleware)

# ── Response Time Monitoring ─────────────────────────────────────────
SLOW_RESPONSE_THRESHOLD_MS = 500.0
METRICS_SAMPLE_SIZE = 1000
response_time_samples = deque(maxlen=METRICS_SAMPLE_SIZE)
METRICS_WINDOW_SECONDS = 600
response_metrics = {
    "total_requests": 0,
    "error_requests": 0,
}
response_metrics_lock = Lock()


def _percentile(values, percentile):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = math.ceil((percentile / 100) * len(sorted_values)) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def record_response_metric(endpoint, method, status_code, response_time_ms):
    with response_metrics_lock:
        response_metrics["total_requests"] += 1
        if status_code >= 400:
            response_metrics["error_requests"] += 1
        response_time_samples.append(
          (time.time(), response_time_ms)
        )

        current_time = time.time()

        while (
          response_time_samples
          and current_time - response_time_samples[0][0] > METRICS_WINDOW_SECONDS
        ):
          response_time_samples.popleft()
    log_level = logging.WARNING if response_time_ms > SLOW_RESPONSE_THRESHOLD_MS else logging.INFO
    if log_level == logging.WARNING:
        logger.warning("API request slow endpoint=%s method=%s status=%s time=%.2fms response_time_ms=%.2f endpoint=%s",
                       endpoint, method, status_code, response_time_ms, response_time_ms, endpoint)
    else:
        logger.info("API request endpoint=%s method=%s status=%s time=%.2fms",
                    endpoint, method, status_code, response_time_ms)


def reset_response_metrics():
    with response_metrics_lock:
        response_metrics["total_requests"] = 0
        response_metrics["error_requests"] = 0
        response_time_samples.clear()


def get_response_metrics_snapshot():
    with response_metrics_lock:
        samples = [value for _, value in response_time_samples]
        total_requests = response_metrics["total_requests"]
        error_requests = response_metrics["error_requests"]
    avg_response_time = sum(samples) / len(samples) if samples else 0.0
    error_rate = (error_requests / total_requests) * 100 if total_requests else 0.0
    return {
        "avg_response_time": round(avg_response_time, 2),
        "p95_response_time": round(_percentile(samples, 95), 2),
        "total_requests": total_requests,
        "error_rate": round(error_rate, 2),
    }


@app.middleware("http")
async def response_time_middleware(request, call_next):
    start_time = time.perf_counter()
    response = None
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        response_time_ms = (time.perf_counter() - start_time) * 1000
        if response is not None:
            response.headers["X-Response-Time"] = f"{response_time_ms:.2f}ms"
        record_response_metric(request.url.path, request.method, status_code, response_time_ms)


# ── State ─────────────────────────────────────────────────────────────
models = {
    "content": None,
    "collab": None,
    "hybrid": None,
    "ready": False,
    "item_df": None,
    "build_time": None,
    "last_trained_at": None,
}

MODEL_REGISTRY = {}
ACTIVE_MODEL_VERSION = None
SHADOW_MODEL_VERSION = None
STAGING_MODEL_VERSION = None

SHADOW_LOGS = []

def generate_model_version():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"1.0.0-{timestamp}"


class RealtimeConnectionHub:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        for connection in disconnected:
            self.disconnect(connection)

realtime_hub = RealtimeConnectionHub()


class WeightsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alpha: float = 0.4
    beta: float = 0.35
    gamma: float = 0.25


class PurchaseCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_\-\.@]+$")
    product_id: int = Field(..., gt=0)
    rating: float = Field(0.0, ge=0.0, le=5.0)
    review_text: str = Field("", max_length=1000)


class FeedbackCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_\-\.@]+$")
    item: str = Field(..., min_length=1, max_length=500)
    feedback: str = Field(..., min_length=1, max_length=2000)


class RealtimeRecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_title: str
    top_n: int = 10
    explain: bool = False
    target_catalog: Optional[str] = None


# ── CSRF Token ───────────────────────────────────────────────────────
@app.get(
    "/api/csrf-token",
    response_model=CSRFTokenResponse,
    summary="Issue a CSRF token",
    tags=["Security"],
)
def get_csrf_token(response: Response):
    token = generate_csrf_token()
    set_csrf_cookie(response, token)
    return CSRFTokenResponse(csrfToken=token)


class FederatedTrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_factors: int = 20
    epochs: int = 5
    lr: float = 0.05
    reg: float = 0.05


# ── Health ────────────────────────────────────────────────────────────
@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_loaded": models["ready"],
    }


# ── API Metrics ───────────────────────────────────────────────────────
@app.get("/api/version")
def get_version():
    return {
        "version": app.version,
        "service": app.title,
        "status": "running",
    }


@app.get("/api/metrics")
def get_api_metrics():
    return get_response_metrics_snapshot()


# ── Config ────────────────────────────────────────────────────────────
@app.get("/api/config")
def get_config():
    return {
        "supabase_url": os.environ.get("SUPABASE_URL", ""),
    }


# ── Status ────────────────────────────────────────────────────────────
@app.get("/api/status")
def status():
    return {
        "status": "healthy",
        "model_ready": models["ready"],
        "message": "Hybrid Recommender API running",
    }


# ── Dashboard ─────────────────────────────────────────────────────────
@app.get("/api/dashboard")
def dashboard(request: Request):
    _require_admin_access(request)
    sb = get_supabase()
    try:
        product_count = sb.table('products').select('id', count='exact').limit(0).execute().count or 0
    except Exception as e:
        logger.warning("Dashboard: product count failed: %s", e)
        product_count = 0

    try:
        interaction_count = sb.table('purchases').select('id', count='exact').limit(0).execute().count or 0
    except Exception as e:
        logger.warning("Dashboard: interaction count failed: %s", e)
        interaction_count = 0

    total_users = 0
    purchase_counts = Counter()

    try:
        user_result = sb.rpc('get_total_users').execute()
        total_users = user_result.data or 0

        top_products_result = sb.rpc('get_top_product_counts').execute()
        purchase_counts = Counter({
            row['product_id']: row['interaction_count']
            for row in (top_products_result.data or [])
        })
    except Exception as e:
        logger.warning("Dashboard error: %s", e)

    avg_recommendation_score = 0.0
    avg_sentiment_score = 0.0
    try:
        prod_stats = sb.table('products').select('rating, avg_sentiment').limit(50000).execute().data or []
        ratings = [float(p['rating']) for p in prod_stats if p.get('rating') not in (None, 0)]
        sentiments = [float(p['avg_sentiment']) for p in prod_stats if p.get('avg_sentiment') is not None]
        if ratings:
            avg_recommendation_score = round(sum(ratings) / len(ratings), 4)
        if sentiments:
            avg_sentiment_score = round(sum(sentiments) / len(sentiments), 4)
    except Exception as e:
        logger.warning("Dashboard: averages query failed: %s", e)

    top_products = []
    try:
        if purchase_counts:
            top_ids = [pid for pid, _ in purchase_counts.most_common(5)]
            prod_result = sb.table('products').select('id, title, category, rating').in_('id', top_ids).execute().data or []
            prod_map = {p['id']: p for p in prod_result}
            for pid in top_ids:
                p = prod_map.get(pid)
                if p:
                    top_products.append({
                        'id': p['id'], 'title': p.get('title', ''),
                        'category': p.get('category', ''),
                        'rating': round(float(p.get('rating', 0) or 0), 2),
                        'interactions': purchase_counts[pid],
                    })
        if not top_products:
            fallback = sb.table('products').select('id, title, category, rating').order('rating', desc=True).limit(5).execute().data or []
            for p in fallback:
                top_products.append({
                    'id': p['id'], 'title': p.get('title', ''),
                    'category': p.get('category', ''),
                    'rating': round(float(p.get('rating', 0) or 0), 2),
                    'interactions': 0,
                })
    except Exception as e:
        logger.warning("Dashboard: top products query failed: %s", e)

    return {
        "total_products": product_count,
        "total_users": total_users,
        "total_interactions": interaction_count,
        "avg_recommendation_score": avg_recommendation_score,
        "avg_sentiment_score": avg_sentiment_score,
        "top_5_recommended_products": top_products,
        "model_last_trained": models.get("last_trained_at"),
    }


# ── Search ────────────────────────────────────────────────────────────
@app.get("/api/search")
def search_items(
    request: Request,
    response: Response,
    q: str = "",
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0, le=10000),
    sort: str = Query(
        "relevance",
        pattern="^(relevance|price-low|price-high|rating)$",
    ),
):
    query = _normalize_search_query(q)
    try:
        rate_limit = int(os.environ.get("RATE_LIMIT_SEARCH_PER_MIN", "60"))
    except ValueError:
        rate_limit = 60

    client_ip = request.client.host if request.client else "127.0.0.1"
    now = time.time()

    with _rate_limit_lock:
        bucket = _rate_limit_buckets.setdefault(client_ip, {"timestamps": []})
        bucket["timestamps"] = [t for t in bucket["timestamps"] if now - t < 60]

        if len(bucket["timestamps"]) >= rate_limit:
            reset_time = int(60 - (now - bucket["timestamps"][0])) if bucket["timestamps"] else 60
            reset_time = max(0, reset_time)
            response.status_code = 429
            response.headers["x-ratelimit-limit"] = str(rate_limit)
            response.headers["x-ratelimit-remaining"] = "0"
            response.headers["x-ratelimit-reset"] = str(reset_time)
            return {
                "error": "Rate limit exceeded",
                "message": "Too many requests. Please try again later.",
            }

        bucket["timestamps"].append(now)
        remaining = rate_limit - len(bucket["timestamps"])
        reset_time = int(60 - (now - bucket["timestamps"][0])) if bucket["timestamps"] else 60
        reset_time = max(0, reset_time)
        response.headers["x-ratelimit-limit"] = str(rate_limit)
        response.headers["x-ratelimit-remaining"] = str(remaining)
        response.headers["x-ratelimit-reset"] = str(reset_time)

    cache_key = _cache_key("search", query, limit, offset, sort)
    cached = _get_cached_response(cache_key)
    if cached is not None:
        _set_cache_headers(response, "HIT")
        return cached

    is_fuzzy_fallback = False
    try:
        sb = get_supabase()

        if query:
            # 1. Attempt standard Full-Text Search (FTS) first
            result = sb.rpc('search_products', {
                'query_text': query, 'match_count': limit, 'offset_val': offset,
            }).execute()
            products = result.data or []
            
            # 2. Task 4 Fallback: If FTS returns fewer than 3 matches, trigger fuzzy search
            if len(products) < 3:
                is_fuzzy_fallback = True
                fuzzy_res = sb.rpc('fuzzy_search_products', {
                    'q': query, 
                    'threshold': 0.3
                }).execute()
                products = fuzzy_res.data or []
        else:
            query_builder = sb.table('products').select('id, title, description, category, rating, avg_sentiment, review_count, metadata')

            if sort == "rating":
                query_builder = query_builder.order('rating', desc=True)
            else:
                query_builder = query_builder.order('rating', desc=True).order('review_count', desc=True)

            result = query_builder.limit(limit).offset(offset).execute()
            products = result.data or []
    except Exception as e:
        logger.warning("Search fallback to mock products: %s", e)
        products = MOCK_PRODUCTS

        if query:
            query_lower = query.lower()
            products = [
                p for p in products
                if query_lower in str(p.get('title', '')).lower()
                or query_lower in str(p.get('description', '')).lower()
                or query_lower in str(p.get('category', '')).lower()
            ]
            for p in products:
                p['rank'] = 0.0

        products = products[offset:offset + limit]

    def _product_price(product):
        metadata = product.get('metadata') or {}
        raw_price = (
            product.get('price')
            if product.get('price') is not None
            else metadata.get('price')
        )
        try:
            return float(raw_price or 0)
        except (TypeError, ValueError):
            return 0.0

    if sort == "price-low":
        products = sorted(products, key=_product_price)
    elif sort == "price-high":
        products = sorted(products, key=_product_price, reverse=True)
    elif sort == "rating":
        products = sorted(products, key=lambda p: float(p.get('rating') or 0), reverse=True)

    results = []
    for p in products:
        price = _product_price(p)
        results.append({
            'id': p.get('id'), 'title': p.get('title', ''),
            'description': str(p.get('description', ''))[:200],
            'category': p.get('category', ''), 'rating': p.get('rating', 0.0),
            'price': price,
            'avg_sentiment': p.get('avg_sentiment', 0.0),
            'review_count': p.get('review_count', 0), 'rank': p.get('rank', 0.0),
        })

    result_count = len(results)
    payload = {
        "results": results,
        "count": result_count,
        "total": result_count,
        "query": query,
        "sort": sort,
        "is_fallback": not query or is_fuzzy_fallback,
    }
    _set_cached_response(cache_key, payload)
    _set_cache_headers(response, "MISS")
    return payload


# ── Autocomplete ──────────────────────────────────────────────────────
@app.get("/api/autocomplete")
def autocomplete_products(
    q: str = Query("", min_length=1),
    limit: int = Query(5, ge=1, le=10),
):
    sb = get_supabase()
    query = _normalize_search_query(q)
    if not query:
        return {"suggestions": []}
    try:
        escaped_query = _escape_like_pattern(query)
        result = (
            sb.table('products')
            .select('title')
            .ilike('title', f'%{escaped_query}%')
            .limit(limit)
            .execute()
        )
        suggestions = []
        seen = set()
        for item in result.data or []:
            title = item.get('title', '').strip()
            if title and title.lower() not in seen:
                suggestions.append(title)
                seen.add(title.lower())
        return {"suggestions": suggestions[:limit]}
    except Exception as e:
        logger.error("Autocomplete error: %s", e)
        raise HTTPException(status_code=500, detail="Autocomplete failed")


# ── Fuzzy Search Endpoint ─────────────────────────────────────────────
@app.get("/api/search/fuzzy")
def fuzzy_search_items(
    request: Request,
    response: Response,
    q: str = "",
    threshold: float = Query(0.3, ge=0.0, le=1.0),
):
    """
    Task 3: Executes a typo-tolerant string similarity lookup 
    using the PostgreSQL pg_trgm extension via Supabase RPC.
    """
    query = _normalize_search_query(q)
    if not query:
        return {"results": [], "count": 0, "query": query}

    try:
        sb = get_supabase()
        result = sb.rpc('fuzzy_search_products', {
            'q': query, 
            'threshold': threshold
        }).execute()
        
        products = result.data or []
        
        results = []
        for p in products:
            metadata = p.get('metadata') or {}
            price = float(p.get('price') if p.get('price') is not None else metadata.get('price', 0.0))
            results.append({
                'id': p.get('id'), 
                'title': p.get('title', ''),
                'description': str(p.get('description', ''))[:200],
                'category': p.get('category', ''), 
                'rating': p.get('rating', 0.0),
                'price': price,
                'avg_sentiment': p.get('avg_sentiment', 0.0),
                'review_count': p.get('review_count', 0), 
                'rank': p.get('rank', 0.0),
            })
            
        return {
            "results": results,
            "count": len(results),
            "query": query,
            "threshold": threshold
        }
    except Exception as e:
        logger.error("Fuzzy search pipeline exception: %s", e)
        raise HTTPException(status_code=500, detail="Fuzzy search failed")


def _validate_upload_bytes(filename: str, ext: str, contents: bytes) -> None:
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if b'\x00' in contents:
        raise HTTPException(status_code=400, detail="Uploaded file appears to be binary.")
    try:
        decoded = contents.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Uploaded file appears to be binary.")
    
    stripped = decoded.strip()
    if ext == ".csv":
        if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
            raise HTTPException(status_code=400, detail="CSV uploads must contain CSV content.")
    elif ext == ".json":
        if not (stripped.startswith("{") or stripped.startswith("[")):
            raise HTTPException(status_code=400, detail="JSON uploads must contain JSON content.")