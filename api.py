"""
Flatprint.news API Backend
FastAPI server for multi-dimensional financial news rankings
"""
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict
from datetime import datetime, date, timedelta, timezone
from pydantic import BaseModel
from collections import defaultdict
import logging
import os
import time

# Import your existing storage module
from r2_storage import get_r2_storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== SERVER-SIDE CACHING ====================
_cache = {}
CACHE_TTL = 300  # 5 minutes in seconds

def get_cached(key: str, loader_func):
    """Cache responses in server memory to reduce R2 calls"""
    now = time.time()
    
    # Check if we have cached data
    if key in _cache:
        data, timestamp = _cache[key]
        # Return cached data if still fresh
        if now - timestamp < CACHE_TTL:
            logger.info(f"✅ Cache HIT: {key}")
            return data
        else:
            logger.info(f"⏰ Cache EXPIRED: {key}")
    
    # Cache miss or expired - load fresh data
    logger.info(f"⬇️  Cache MISS: {key} - loading from R2")
    data = loader_func()
    _cache[key] = (data, now)
    return data

app = FastAPI(
    title="Flatprint.news API",
    description="Multi-dimensional AI-ranked financial news aggregator",
    version="1.0.0"
)

# CORS configuration - locked to production domain - dont init until production
# AFTER (uncomment and update):
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://flatprint.news", "http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_cache_headers(request: Request, call_next):
    """Add cache headers to tell browsers to cache responses"""
    response = await call_next(request)
    
    # Tell browsers to cache GET requests for 5 minutes
    if request.url.path.startswith("/api/v1/") and request.method == "GET":
        response.headers["Cache-Control"] = "public, max-age=300"
    
    return response

# ==================== RATE LIMITING ====================

# In-memory rate limiter (use Redis for production with multiple servers)
rate_limit_storage = defaultdict(list)
RATE_LIMIT = 60  # requests per minute
RATE_WINDOW = 60  # seconds


def check_rate_limit(ip: str) -> bool:
    """Check if IP is within rate limit"""
    now = time.time()
    
    # Clean old entries
    rate_limit_storage[ip] = [
        timestamp for timestamp in rate_limit_storage[ip]
        if now - timestamp < RATE_WINDOW
    ]
    
    # Check limit
    if len(rate_limit_storage[ip]) >= RATE_LIMIT:
        return False
    
    # Add new request
    rate_limit_storage[ip].append(now)
    return True


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    # Skip rate limiting for health check
    if request.url.path == "/health":
        return await call_next(request)
    
    # Get client IP
    client_ip = request.client.host
    
    # Check rate limit
    if not check_rate_limit(client_ip):
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": f"Maximum {RATE_LIMIT} requests per minute allowed",
                "retry_after": 60
            }
        )
    
    return await call_next(request)


# Initialize R2 storage
try:
    storage = get_r2_storage()
    logger.info("✅ R2 Storage initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize R2: {e}")
    storage = None


# ==================== MODELS ====================

class Article(BaseModel):
    """Article response model"""
    title: str
    source: str
    url: str
    overall_score: float
    macro_score: float
    equities_score: float
    sector_score: float
    sentiment: float
    dimension_primary: str
    description: Optional[str] = None
    publish_date: Optional[str] = None


class PaginationMeta(BaseModel):
    """Pagination metadata"""
    page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_previous: bool


class ArticlesResponse(BaseModel):
    """API response wrapper with pagination"""
    date: str
    last_updated: str
    run_number: int
    total_articles: int
    graded_articles: int
    metadata_coverage_pct: float
    dimension_stats: Dict
    dimension_distribution: Dict
    articles: List[Article]
    pagination: Optional[PaginationMeta] = None


class StockMover(BaseModel):
    """Stock mover model"""
    ticker: str
    change: float
    news: List[Dict]


class IndustryMover(BaseModel):
    """Industry mover model"""
    industry: str
    change: float
    url: str
    num_stocks: int
    stocks: List[StockMover]


class MoversResponse(BaseModel):
    """Movers response model"""
    date: str
    timestamp: str
    run_number: int
    data_source: str
    positive_industries_count: int
    negative_industries_count: int
    total_news_articles: int
    positive_industries: Dict[str, IndustryMover]
    negative_industries: Dict[str, IndustryMover]


# ==================== HELPER FUNCTIONS ====================

def get_date_est() -> date:
    """Get current date in EST"""
    est = timezone(timedelta(hours=-5))
    return datetime.now(est).date()


def paginate_results(items: List, page: int, page_size: int) -> tuple:
    """Paginate a list of items and return (items, metadata)"""
    total_items = len(items)
    total_pages = (total_items + page_size - 1) // page_size
    
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    paginated_items = items[start_idx:end_idx]
    
    metadata = PaginationMeta(
        page=page,
        page_size=page_size,
        total_items=total_items,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_previous=page > 1
    )
    
    return paginated_items, metadata


def load_articles_for_date(target_date: str) -> Optional[Dict]:
    """Load articles for a specific date from R2"""
    if not storage:
        raise HTTPException(status_code=503, detail="Storage not available")
    
    try:
        key = f"articles/{target_date}.json"
        response = storage.s3_client.get_object(
            Bucket=storage.bucket_name, 
            Key=key
        )
        import json
        data = json.loads(response['Body'].read().decode('utf-8'))
        return data
    except storage.s3_client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        logger.error(f"Error loading articles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load articles: {str(e)}")


def load_movers_for_date(target_date: str) -> Optional[Dict]:
    """Load sector movers for a specific date from R2"""
    if not storage:
        raise HTTPException(status_code=503, detail="Storage not available")
    
    try:
        key = f"movers/{target_date}.json"
        response = storage.s3_client.get_object(
            Bucket=storage.bucket_name, 
            Key=key
        )
        import json
        data = json.loads(response['Body'].read().decode('utf-8'))
        return data
    except storage.s3_client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        logger.error(f"Error loading movers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load movers: {str(e)}")


def filter_articles(
    articles: List[Dict],
    dimension: Optional[str] = None,
    min_score: float = 0.0,
    sentiment_min: Optional[float] = None,
    sentiment_max: Optional[float] = None,
    source: Optional[str] = None
) -> List[Dict]:
    """Filter articles based on criteria"""
    filtered = articles.copy()
    
    # Filter by dimension score
    if dimension and dimension in ['macro', 'equities', 'sector']:
        score_key = f"{dimension}_score"
        filtered = [a for a in filtered if a.get(score_key, 0) >= min_score]
        # Sort by dimension score
        filtered.sort(key=lambda x: x.get(score_key, 0), reverse=True)
    elif min_score > 0:
        # Filter by overall score
        filtered = [a for a in filtered if a.get('overall_score', 0) >= min_score]
    
    # Filter by sentiment range
    if sentiment_min is not None:
        filtered = [a for a in filtered if a.get('sentiment', 0) >= sentiment_min]
    if sentiment_max is not None:
        filtered = [a for a in filtered if a.get('sentiment', 0) <= sentiment_max]
    
    # Filter by source
    if source:
        filtered = [a for a in filtered if a.get('source', '').lower() == source.lower()]
    
    return filtered


# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Flatprint.news API",
        "version": "1.0.0",
        "status": "operational",
        "rate_limit": f"{RATE_LIMIT} requests per minute",
        "endpoints": {
            "articles": {
                "today": "/api/v1/today",
                "date": "/api/v1/articles/{date}",
                "top": "/api/v1/top",
                "macro": "/api/v1/macro",
                "equities": "/api/v1/equities",
                "sector": "/api/v1/sector",
                "bullish": "/api/v1/bullish",
                "bearish": "/api/v1/bearish",
                "search": "/api/v1/search",
                "stats": "/api/v1/stats"
            },
            "movers": {
                "today": "/api/v1/movers/today",
                "date": "/api/v1/movers/{date}",
                "gainers": "/api/v1/movers/gainers",
                "losers": "/api/v1/movers/losers",
                "industry": "/api/v1/movers/industry/{industry_name}",
                "stock": "/api/v1/movers/stock/{ticker}",
                "stats": "/api/v1/movers/stats"
            },
            "indices": {
                "prices": "/api/v1/prices",
            },
            "utility": {
                "dates": "/api/v1/dates",
                "health": "/health"
            }
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "storage": "connected" if storage else "disconnected",
        "timestamp": datetime.now(timezone(timedelta(hours=-5))).isoformat()
    }


@app.get("/api/v1/prices")
async def get_latest_prices():
    """Get latest index prices"""
    if not storage:
        raise HTTPException(status_code=503, detail="Storage not available")
    
    cache_key = "prices:latest"
    
    def load_data():
        try:
            response = storage.s3_client.get_object(
                Bucket=storage.bucket_name, 
                Key='prices/latest.json'
            )
            import json
            return json.loads(response['Body'].read())
        except storage.s3_client.exceptions.NoSuchKey:
            raise HTTPException(status_code=404, detail="Price data not found")
        except Exception as e:
            logger.error(f"Error loading prices: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load prices: {str(e)}")
    
    return get_cached(cache_key, load_data)

@app.get("/api/v1/today", response_model=ArticlesResponse)
async def get_today_articles(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(25, ge=1, le=100, description="Items per page"),
    min_score: float = Query(0.0, ge=0.0, le=10.0),
    source: Optional[str] = Query(None)
):
    """Get ALL articles for today with pagination"""
    today = get_date_est().isoformat()
    
    # Create unique cache key including all parameters
    cache_key = f"today:{today}:{page}:{page_size}:{min_score}:{source}"
    
    def load_data():
        data = load_articles_for_date(today)
        
        if not data:
            raise HTTPException(status_code=404, detail="No articles available")
        
        # Filter articles
        filtered = filter_articles(
            data['articles'],
            min_score=min_score,
            source=source
        )
        
        # Paginate
        paginated_articles, pagination = paginate_results(filtered, page, page_size)
        
        data['articles'] = paginated_articles
        data['total_articles'] = pagination.total_items
        data['pagination'] = pagination
        
        return data
    
    return get_cached(cache_key, load_data)


@app.get("/api/v1/articles/{target_date}", response_model=ArticlesResponse)
async def get_articles_by_date(
    target_date: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(25, ge=1, le=100, description="Items per page"),
    min_score: float = Query(0.0, ge=0.0, le=10.0)
):
    """Get articles for a specific date with pagination (format: YYYY-MM-DD)"""
    try:
        datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    data = load_articles_for_date(target_date)
    
    if not data:
        raise HTTPException(status_code=404, detail=f"No articles found for {target_date}")
    
    # Filter
    filtered = filter_articles(data['articles'], min_score=min_score)
    
    # Paginate
    paginated_articles, pagination = paginate_results(filtered, page, page_size)
    
    data['articles'] = paginated_articles
    data['total_articles'] = pagination.total_items
    data['pagination'] = pagination
    
    return data


@app.get("/api/v1/top", response_model=ArticlesResponse)
async def get_top_articles(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(25, ge=1, le=100, description="Items per page"),
    min_score: float = Query(7.0, ge=0.0, le=10.0, description="Minimum overall score threshold")
):
    """Get top-ranked articles with pagination"""
    today = get_date_est().isoformat()
    data = load_articles_for_date(today)
    
    if not data:
        raise HTTPException(status_code=404, detail="No articles available for today")
    
    # Filter for high-scoring articles
    filtered = [a for a in data['articles'] if a.get('overall_score', 0) >= min_score]
    
    # Paginate
    paginated_articles, pagination = paginate_results(filtered, page, page_size)
    
    data['articles'] = paginated_articles
    data['total_articles'] = pagination.total_items
    data['pagination'] = pagination
    
    return data


@app.get("/api/v1/macro")
async def get_macro_articles(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(25, ge=1, le=100, description="Items per page"),
    min_score: float = Query(6.0, ge=0.0, le=10.0, description="Minimum macro score")
):
    """Get articles with high macro/policy relevance with pagination"""
    today = get_date_est().isoformat()
    data = load_articles_for_date(today)
    
    if not data:
        raise HTTPException(status_code=404, detail="No articles available")
    
    filtered = filter_articles(data['articles'], dimension='macro', min_score=min_score)
    
    # Paginate
    paginated_articles, pagination = paginate_results(filtered, page, page_size)
    
    return {
        **data,
        "articles": paginated_articles,
        "total_articles": pagination.total_items,
        "pagination": pagination.dict(),
        "filter": {"dimension": "macro", "min_score": min_score}
    }


@app.get("/api/v1/equities")
async def get_equities_articles(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(25, ge=1, le=100, description="Items per page"),
    min_score: float = Query(6.0, ge=0.0, le=10.0, description="Minimum equities score")
):
    """Get articles about individual stocks with pagination"""
    today = get_date_est().isoformat()
    data = load_articles_for_date(today)
    
    if not data:
        raise HTTPException(status_code=404, detail="No articles available")
    
    filtered = filter_articles(data['articles'], dimension='equities', min_score=min_score)
    
    # Paginate
    paginated_articles, pagination = paginate_results(filtered, page, page_size)
    
    return {
        **data,
        "articles": paginated_articles,
        "total_articles": pagination.total_items,
        "pagination": pagination.dict(),
        "filter": {"dimension": "equities", "min_score": min_score}
    }


@app.get("/api/v1/sector")
async def get_sector_articles(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(25, ge=1, le=100, description="Items per page"),
    min_score: float = Query(6.0, ge=0.0, le=10.0, description="Minimum sector score")
):
    """Get articles about sector-wide trends with pagination"""
    today = get_date_est().isoformat()
    data = load_articles_for_date(today)
    
    if not data:
        raise HTTPException(status_code=404, detail="No articles available")
    
    filtered = filter_articles(data['articles'], dimension='sector', min_score=min_score)
    
    # Paginate
    paginated_articles, pagination = paginate_results(filtered, page, page_size)
    
    return {
        **data,
        "articles": paginated_articles,
        "total_articles": pagination.total_items,
        "pagination": pagination.dict(),
        "filter": {"dimension": "sector", "min_score": min_score}
    }


@app.get("/api/v1/bullish")
async def get_bullish_articles(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(25, ge=1, le=100, description="Items per page"),
    min_sentiment: float = Query(2.0, ge=0.0, le=5.0, description="Minimum positive sentiment")
):
    """Get bullish/positive sentiment articles with pagination"""
    today = get_date_est().isoformat()
    data = load_articles_for_date(today)
    
    if not data:
        raise HTTPException(status_code=404, detail="No articles available")
    
    filtered = filter_articles(
        data['articles'], 
        sentiment_min=min_sentiment
    )
    
    # Sort by sentiment (most bullish first)
    filtered.sort(key=lambda x: x.get('sentiment', 0), reverse=True)
    
    # Paginate
    paginated_articles, pagination = paginate_results(filtered, page, page_size)
    
    return {
        **data,
        "articles": paginated_articles,
        "total_articles": pagination.total_items,
        "pagination": pagination.dict(),
        "filter": {"sentiment": "bullish", "min_sentiment": min_sentiment}
    }


@app.get("/api/v1/bearish")
async def get_bearish_articles(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(25, ge=1, le=100, description="Items per page"),
    max_sentiment: float = Query(-2.0, ge=-5.0, le=0.0, description="Maximum negative sentiment")
):
    """Get bearish/negative sentiment articles with pagination"""
    today = get_date_est().isoformat()
    data = load_articles_for_date(today)
    
    if not data:
        raise HTTPException(status_code=404, detail="No articles available")
    
    filtered = filter_articles(
        data['articles'],
        sentiment_max=max_sentiment
    )
    
    # Sort by sentiment (most bearish first)
    filtered.sort(key=lambda x: x.get('sentiment', 0))
    
    # Paginate
    paginated_articles, pagination = paginate_results(filtered, page, page_size)
    
    return {
        **data,
        "articles": paginated_articles,
        "total_articles": pagination.total_items,
        "pagination": pagination.dict(),
        "filter": {"sentiment": "bearish", "max_sentiment": max_sentiment}
    }


@app.get("/api/v1/dates")
async def get_available_dates(days: int = Query(7, ge=1, le=30)):
    """Get list of dates with available data"""
    if not storage:
        raise HTTPException(status_code=503, detail="Storage not available")
    
    try:
        files = storage.list_recent_files(days=days)
        
        dates = []
        for key in files:
            try:
                date_str = key.split('/')[1].split('.')[0]
                dates.append(date_str)
            except:
                continue
        
        return {
            "available_dates": dates,
            "count": len(dates)
        }
    except Exception as e:
        logger.error(f"Error listing dates: {e}")
        raise HTTPException(status_code=500, detail="Failed to list available dates")


@app.get("/api/v1/search")
async def search_articles(
    q: str = Query(..., min_length=3, description="Search query"),
    limit: int = Query(25, ge=1, le=100)
):
    """Search articles by title or description"""
    today = get_date_est().isoformat()
    
    try:
        data = load_articles_for_date(today)
        
        if not data or 'articles' not in data:
            raise HTTPException(status_code=404, detail="No articles available")
        
        query = q.lower()
        results = []
        
        for article in data.get('articles', []):
            try:
                title = str(article.get('title', '')).lower()
                desc = str(article.get('description', '')).lower()
                
                if query in title or query in desc:
                    results.append(article)
                
                if len(results) >= limit:
                    break
            except Exception as e:
                logger.warning(f"Skipping article in search: {e}")
                continue
        
        return {
            "query": q,
            "results": results,
            "total_found": len(results),
            "date": today
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/api/v1/stats")
async def get_statistics():
    """Get statistics about today's articles"""
    today = get_date_est().isoformat()
    data = load_articles_for_date(today)
    
    if not data:
        raise HTTPException(status_code=404, detail="No articles available")
    
    articles = data['articles']
    graded = [a for a in articles if 'overall_score' in a]
    
    return {
        "date": today,
        "last_updated": data.get('last_updated'),
        "run_number": data.get('run_number'),
        "total_articles": len(articles),
        "graded_articles": len(graded),
        "metadata_coverage": data.get('metadata_coverage_pct', 0),
        "dimension_stats": data.get('dimension_stats', {}),
        "dimension_distribution": data.get('dimension_distribution', {}),
        "score_ranges": {
            "overall": {
                "min": min(a.get('overall_score', 0) for a in graded) if graded else 0,
                "max": max(a.get('overall_score', 0) for a in graded) if graded else 0,
                "avg": sum(a.get('overall_score', 0) for a in graded) / len(graded) if graded else 0
            },
            "sentiment": {
                "min": min(a.get('sentiment', 0) for a in graded) if graded else 0,
                "max": max(a.get('sentiment', 0) for a in graded) if graded else 0,
                "avg": sum(a.get('sentiment', 0) for a in graded) / len(graded) if graded else 0
            }
        }
    }


# ==================== MOVERS ENDPOINTS ====================

@app.get("/api/v1/movers/today")
async def get_today_movers():
    """Get today's sector movers"""
    today = get_date_est().isoformat()
    
    cache_key = f"movers:today:{today}"
    
    def load_data():
        data = load_movers_for_date(today)
        
        if not data:
            raise HTTPException(status_code=404, detail="No movers data available for today")
        
        return {
            "date": data.get("date"),
            "timestamp": data.get("timestamp"),
            "run_number": data.get("run_number"),
            "data_source": data.get("data_source"),
            "total_news_articles": data.get("total_news_articles", 0),
            "positive_industries_count": data.get("positive_industries_count", 0),
            "negative_industries_count": data.get("negative_industries_count", 0),
            "positive_industries": data.get("movers", {}).get("Positive Industries", {}),
            "negative_industries": data.get("movers", {}).get("Negative Industries", {})
        }
    
    return get_cached(cache_key, load_data)


@app.get("/api/v1/movers/gainers")
async def get_top_gainers(
    limit: int = Query(5, ge=1, le=10, description="Number of top gaining industries")
):
    """Get top gaining industries"""
    today = get_date_est().isoformat()
    data = load_movers_for_date(today)
    
    if not data:
        raise HTTPException(status_code=404, detail="No movers data available")
    
    positive_industries = data.get("movers", {}).get("Positive Industries", {})
    
    sorted_industries = sorted(
        positive_industries.items(),
        key=lambda x: x[1].get("change", 0),
        reverse=True
    )[:limit]
    
    return {
        "date": data.get("date"),
        "industries": dict(sorted_industries),
        "count": len(sorted_industries)
    }


@app.get("/api/v1/movers/losers")
async def get_top_losers(
    limit: int = Query(5, ge=1, le=10, description="Number of top losing industries")
):
    """Get top losing industries"""
    today = get_date_est().isoformat()
    data = load_movers_for_date(today)
    
    if not data:
        raise HTTPException(status_code=404, detail="No movers data available")
    
    negative_industries = data.get("movers", {}).get("Negative Industries", {})
    
    sorted_industries = sorted(
        negative_industries.items(),
        key=lambda x: x[1].get("change", 0)
    )[:limit]
    
    return {
        "date": data.get("date"),
        "industries": dict(sorted_industries),
        "count": len(sorted_industries)
    }


@app.get("/api/v1/movers/stats")
async def get_movers_stats():
    """Get statistics about today's movers"""
    today = get_date_est().isoformat()
    data = load_movers_for_date(today)
    
    if not data:
        raise HTTPException(status_code=404, detail="No movers data available")
    
    movers = data.get("movers", {})
    positive = movers.get("Positive Industries", {})
    negative = movers.get("Negative Industries", {})
    
    total_stocks = 0
    total_news = 0
    
    for industry_data in list(positive.values()) + list(negative.values()):
        stocks = industry_data.get("stocks", [])
        total_stocks += len(stocks)
        for stock in stocks:
            total_news += len(stock.get("news", []))
    
    return {
        "date": data.get("date"),
        "timestamp": data.get("timestamp"),
        "run_number": data.get("run_number"),
        "positive_industries": len(positive),
        "negative_industries": len(negative),
        "total_industries": len(positive) + len(negative),
        "total_stocks_tracked": total_stocks,
        "total_news_articles": total_news,
        "data_source": data.get("data_source")
    }


@app.get("/api/v1/movers/{target_date}")
async def get_movers_by_date(target_date: str):
    """Get sector movers for a specific date"""
    try:
        datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    data = load_movers_for_date(target_date)
    
    if not data:
        raise HTTPException(status_code=404, detail=f"No movers data found for {target_date}")
    
    return {
        "date": data.get("date"),
        "timestamp": data.get("timestamp"),
        "run_number": data.get("run_number"),
        "data_source": data.get("data_source"),
        "total_news_articles": data.get("total_news_articles", 0),
        "positive_industries_count": data.get("positive_industries_count", 0),
        "negative_industries_count": data.get("negative_industries_count", 0),
        "positive_industries": data.get("movers", {}).get("Positive Industries", {}),
        "negative_industries": data.get("movers", {}).get("Negative Industries", {})
    }


@app.get("/api/v1/movers/industry/{industry_name}")
async def get_industry_details(industry_name: str):
    """Get detailed information about a specific industry"""
    today = get_date_est().isoformat()
    data = load_movers_for_date(today)
    
    if not data:
        raise HTTPException(status_code=404, detail="No movers data available")
    
    movers = data.get("movers", {})
    
    industry_data = None
    category = None
    
    if industry_name in movers.get("Positive Industries", {}):
        industry_data = movers["Positive Industries"][industry_name]
        category = "gainer"
    elif industry_name in movers.get("Negative Industries", {}):
        industry_data = movers["Negative Industries"][industry_name]
        category = "loser"
    
    if not industry_data:
        raise HTTPException(status_code=404, detail=f"Industry '{industry_name}' not found")
    
    return {
        "date": data.get("date"),
        "industry": industry_name,
        "category": category,
        "change": industry_data.get("change"),
        "url": industry_data.get("url"),
        "num_stocks": industry_data.get("num_stocks"),
        "stocks": industry_data.get("stocks", [])
    }


@app.get("/api/v1/movers/stock/{ticker}")
async def get_stock_news(ticker: str):
    """Get news for a specific stock ticker from movers data"""
    today = get_date_est().isoformat()
    data = load_movers_for_date(today)
    
    if not data:
        raise HTTPException(status_code=404, detail="No movers data available")
    
    ticker = ticker.upper()
    movers = data.get("movers", {})
    
    # Search for ticker in all industries
    for category in ["Positive Industries", "Negative Industries"]:
        for industry_name, industry_data in movers.get(category, {}).items():
            for stock in industry_data.get("stocks", []):
                if stock.get("ticker") == ticker:
                    return {
                        "date": data.get("date"),
                        "ticker": ticker,
                        "change": stock.get("change"),
                        "industry": industry_name,
                        "category": "gainer" if category == "Positive Industries" else "loser",
                        "news": stock.get("news", []),
                        "news_count": len(stock.get("news", []))
                    }
    
    raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found in movers data")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)