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


@app.get("/api/v1/today", response_model=ArticlesResponse)
async def get_today_articles(
    min_score: float = Query(0.0, ge=0.0, le=10.0),
    source: Optional[str] = Query(None)
):
    """Get ALL articles for today"""
    today = get_date_est().isoformat()
    data = load_articles_for_date(today)
    
    if not data:
        raise HTTPException(status_code=404, detail="No articles available")
    
    articles = filter_articles(
        data['articles'],
        min_score=min_score,
        source=source
    )
    
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

    # ==================== NEW MODELS ====================
# Add these new models after the existing Article and ArticlesResponse models:

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


# Add this helper function after load_articles_for_date():

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


# Add these endpoints before if __name__ == "__main__":

@app.get("/api/v1/movers/today")
async def get_today_movers():
    """Get today's sector movers (gainers and losers)"""
    today = get_date_est().isoformat()
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


@app.get("/api/v1/movers/{target_date}")
async def get_movers_by_date(target_date: str):
    """Get sector movers for a specific date (format: YYYY-MM-DD)"""
    # Validate date format
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


@app.get("/api/v1/movers/gainers")
async def get_top_gainers(
    limit: int = Query(5, ge=1, le=10, description="Number of top gaining industries")
):
    """Get top gaining industries with their top stocks"""
    today = get_date_est().isoformat()
    data = load_movers_for_date(today)
    
    if not data:
        raise HTTPException(status_code=404, detail="No movers data available")
    
    positive_industries = data.get("movers", {}).get("Positive Industries", {})
    
    # Sort by change percentage
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
    """Get top losing industries with their top stocks"""
    today = get_date_est().isoformat()
    data = load_movers_for_date(today)
    
    if not data:
        raise HTTPException(status_code=404, detail="No movers data available")
    
    negative_industries = data.get("movers", {}).get("Negative Industries", {})
    
    # Sort by change percentage (most negative first)
    sorted_industries = sorted(
        negative_industries.items(),
        key=lambda x: x[1].get("change", 0)
    )[:limit]
    
    return {
        "date": data.get("date"),
        "industries": dict(sorted_industries),
        "count": len(sorted_industries)
    }


@app.get("/api/v1/movers/industry/{industry_name}")
async def get_industry_details(industry_name: str):
    """Get detailed information about a specific industry including stocks and news"""
    today = get_date_est().isoformat()
    data = load_movers_for_date(today)
    
    if not data:
        raise HTTPException(status_code=404, detail="No movers data available")
    
    movers = data.get("movers", {})
    
    # Search in both positive and negative industries
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
    
    # Calculate stats
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


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)