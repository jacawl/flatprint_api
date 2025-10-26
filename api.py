"""
Flatprint.news API Backend
FastAPI server for multi-dimensional financial news rankings
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict
from datetime import datetime, date, timedelta, timezone
from pydantic import BaseModel
import logging
import os

# Import your existing storage module
from r2_storage import get_r2_storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Flatprint.news API",
    description="Multi-dimensional AI-ranked financial news aggregator",
    version="1.0.0"
)

# CORS configuration - adjust origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


class ArticlesResponse(BaseModel):
    """API response wrapper"""
    date: str
    last_updated: str
    run_number: int
    total_articles: int
    graded_articles: int
    metadata_coverage_pct: float
    dimension_stats: Dict
    dimension_distribution: Dict
    articles: List[Article]


# ==================== HELPER FUNCTIONS ====================

def get_date_est() -> date:
    """Get current date in EST"""
    est = timezone(timedelta(hours=-5))
    return datetime.now(est).date()


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
        "endpoints": {
            "today": "/api/v1/today",
            "date": "/api/v1/articles/{date}",
            "top": "/api/v1/top",
            "macro": "/api/v1/macro",
            "equities": "/api/v1/equities",
            "sector": "/api/v1/sector",
            "bullish": "/api/v1/bullish",
            "bearish": "/api/v1/bearish",
            "recent_dates": "/api/v1/dates"
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


@app.get("/api/v1/today", response_model=ArticlesResponse)
async def get_today_articles(
    limit: int = Query(50, ge=1, le=500, description="Number of articles to return"),
    min_score: float = Query(0.0, ge=0.0, le=10.0, description="Minimum overall score"),
    source: Optional[str] = Query(None, description="Filter by source (e.g., 'bloomberg', 'reuters')")
):
    """Get today's articles"""
    today = get_date_est().isoformat()
    data = load_articles_for_date(today)
    
    if not data:
        raise HTTPException(
            status_code=404, 
            detail=f"No articles found for today ({today}). Check back during market hours (9am, 12pm, or 4pm EST)."
        )
    
    # Filter and limit
    articles = filter_articles(
        data['articles'],
        min_score=min_score,
        source=source
    )[:limit]
    
    data['articles'] = articles
    data['total_articles'] = len(articles)
    
    return data


@app.get("/api/v1/articles/{target_date}", response_model=ArticlesResponse)
async def get_articles_by_date(
    target_date: str,
    limit: int = Query(50, ge=1, le=500),
    min_score: float = Query(0.0, ge=0.0, le=10.0)
):
    """Get articles for a specific date (format: YYYY-MM-DD)"""
    # Validate date format
    try:
        datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    data = load_articles_for_date(target_date)
    
    if not data:
        raise HTTPException(status_code=404, detail=f"No articles found for {target_date}")
    
    # Filter and limit
    articles = filter_articles(data['articles'], min_score=min_score)[:limit]
    data['articles'] = articles
    data['total_articles'] = len(articles)
    
    return data


@app.get("/api/v1/top", response_model=ArticlesResponse)
async def get_top_articles(
    limit: int = Query(25, ge=1, le=100, description="Number of top articles"),
    min_score: float = Query(7.0, ge=0.0, le=10.0, description="Minimum overall score threshold")
):
    """Get top-ranked articles (high importance across all dimensions)"""
    today = get_date_est().isoformat()
    data = load_articles_for_date(today)
    
    if not data:
        raise HTTPException(status_code=404, detail="No articles available for today")
    
    # Filter for high-scoring articles
    articles = [a for a in data['articles'] if a.get('overall_score', 0) >= min_score]
    articles = articles[:limit]
    
    data['articles'] = articles
    data['total_articles'] = len(articles)
    
    return data


@app.get("/api/v1/macro")
async def get_macro_articles(
    limit: int = Query(25, ge=1, le=100),
    min_score: float = Query(6.0, ge=0.0, le=10.0, description="Minimum macro score")
):
    """Get articles with high macro/policy relevance (Fed, inflation, GDP, etc.)"""
    today = get_date_est().isoformat()
    data = load_articles_for_date(today)
    
    if not data:
        raise HTTPException(status_code=404, detail="No articles available")
    
    articles = filter_articles(data['articles'], dimension='macro', min_score=min_score)[:limit]
    
    return {
        **data,
        "articles": articles,
        "total_articles": len(articles),
        "filter": {"dimension": "macro", "min_score": min_score}
    }


@app.get("/api/v1/equities")
async def get_equities_articles(
    limit: int = Query(25, ge=1, le=100),
    min_score: float = Query(6.0, ge=0.0, le=10.0, description="Minimum equities score")
):
    """Get articles about individual stocks and companies"""
    today = get_date_est().isoformat()
    data = load_articles_for_date(today)
    
    if not data:
        raise HTTPException(status_code=404, detail="No articles available")
    
    articles = filter_articles(data['articles'], dimension='equities', min_score=min_score)[:limit]
    
    return {
        **data,
        "articles": articles,
        "total_articles": len(articles),
        "filter": {"dimension": "equities", "min_score": min_score}
    }


@app.get("/api/v1/sector")
async def get_sector_articles(
    limit: int = Query(25, ge=1, le=100),
    min_score: float = Query(6.0, ge=0.0, le=10.0, description="Minimum sector score")
):
    """Get articles about sector-wide trends and industry news"""
    today = get_date_est().isoformat()
    data = load_articles_for_date(today)
    
    if not data:
        raise HTTPException(status_code=404, detail="No articles available")
    
    articles = filter_articles(data['articles'], dimension='sector', min_score=min_score)[:limit]
    
    return {
        **data,
        "articles": articles,
        "total_articles": len(articles),
        "filter": {"dimension": "sector", "min_score": min_score}
    }


@app.get("/api/v1/bullish")
async def get_bullish_articles(
    limit: int = Query(25, ge=1, le=100),
    min_sentiment: float = Query(2.0, ge=0.0, le=5.0, description="Minimum positive sentiment")
):
    """Get bullish/positive sentiment articles"""
    today = get_date_est().isoformat()
    data = load_articles_for_date(today)
    
    if not data:
        raise HTTPException(status_code=404, detail="No articles available")
    
    articles = filter_articles(
        data['articles'], 
        sentiment_min=min_sentiment
    )[:limit]
    
    # Sort by sentiment (most bullish first)
    articles.sort(key=lambda x: x.get('sentiment', 0), reverse=True)
    
    return {
        **data,
        "articles": articles,
        "total_articles": len(articles),
        "filter": {"sentiment": "bullish", "min_sentiment": min_sentiment}
    }


@app.get("/api/v1/bearish")
async def get_bearish_articles(
    limit: int = Query(25, ge=1, le=100),
    max_sentiment: float = Query(-2.0, ge=-5.0, le=0.0, description="Maximum negative sentiment")
):
    """Get bearish/negative sentiment articles"""
    today = get_date_est().isoformat()
    data = load_articles_for_date(today)
    
    if not data:
        raise HTTPException(status_code=404, detail="No articles available")
    
    articles = filter_articles(
        data['articles'],
        sentiment_max=max_sentiment
    )[:limit]
    
    # Sort by sentiment (most bearish first)
    articles.sort(key=lambda x: x.get('sentiment', 0))
    
    return {
        **data,
        "articles": articles,
        "total_articles": len(articles),
        "filter": {"sentiment": "bearish", "max_sentiment": max_sentiment}
    }


@app.get("/api/v1/dates")
async def get_available_dates(days: int = Query(7, ge=1, le=30)):
    """Get list of dates with available data"""
    if not storage:
        raise HTTPException(status_code=503, detail="Storage not available")
    
    try:
        files = storage.list_recent_files(days=days)
        
        # Extract dates from filenames
        dates = []
        for key in files:
            try:
                # Extract "2025-10-25" from "articles/2025-10-25.json"
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
    data = load_articles_for_date(today)
    
    if not data:
        raise HTTPException(status_code=404, detail="No articles available")
    
    # Simple case-insensitive search
    query = q.lower()
    results = []
    
    for article in data['articles']:
        title = article.get('title', '').lower()
        desc = article.get('description', '').lower()
        
        if query in title or query in desc:
            results.append(article)
        
        if len(results) >= limit:
            break
    
    return {
        "query": q,
        "results": results,
        "total_found": len(results),
        "date": today
    }


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


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)