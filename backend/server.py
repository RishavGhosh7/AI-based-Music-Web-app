from fastapi import FastAPI, APIRouter, HTTPException, Depends, Query
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import asyncio
import json
from emergentintegrations.llm.chat import LlmChat, UserMessage
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import httpx

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Initialize FastAPI
app = FastAPI(
    title="Music Recommender AI",
    description="AI-powered music recommendation system with personalized suggestions",
    version="1.0.0"
)

# Create API router
api_router = APIRouter(prefix="/api")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic Models
class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: str
    favorite_genres: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Song(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    spotify_id: str
    name: str
    artist: str
    album: str
    genres: List[str] = Field(default_factory=list)
    duration_ms: int
    popularity: int
    audio_features: Optional[Dict[str, float]] = None
    preview_url: Optional[str] = None
    external_url: str
    image_url: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Rating(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    song_id: str
    rating: int = Field(ge=1, le=5)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Favorite(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    song_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Recommendation(BaseModel):
    song_id: str
    score: float
    reason: str
    song_details: Song

class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    confidence: float
    sources: List[str] = Field(default_factory=list)

# Request/Response Models
class UserCreate(BaseModel):
    name: str
    email: str
    favorite_genres: List[str] = Field(default_factory=list)

class SongSearch(BaseModel):
    query: str
    limit: int = 20

class RatingCreate(BaseModel):
    user_id: str
    song_id: str
    rating: int = Field(ge=1, le=5)

class FavoriteCreate(BaseModel):
    user_id: str
    song_id: str

# Initialize AI Services
class AIService:
    def __init__(self):
        self.emergent_key = os.environ.get('EMERGENT_LLM_KEY')
        if not self.emergent_key:
            raise ValueError("EMERGENT_LLM_KEY not found in environment")
        
        # Initialize Gemini for recommendations
        self.gemini_chat = LlmChat(
            api_key=self.emergent_key,
            session_id="recommendation-engine",
            system_message="You are a music recommendation expert. Analyze user preferences and suggest similar songs based on musical characteristics like genre, tempo, mood, and style."
        ).with_model("gemini", "gemini-2.0-flash")
        
        # Initialize Perplexity for explanations (using httpx for direct API calls)
        self.perplexity_api_key = None  # Will implement if user provides their own key
    
    async def generate_recommendations(self, user_preferences: Dict, available_songs: List[Dict]) -> List[Dict]:
        """Generate AI-powered music recommendations using Gemini"""
        try:
            # Prepare context for AI
            context = {
                "favorite_genres": user_preferences.get("favorite_genres", []),
                "recent_ratings": user_preferences.get("recent_ratings", []),
                "favorite_songs": user_preferences.get("favorite_songs", []),
                "available_songs": available_songs[:50]  # Limit for token efficiency
            }
            
            prompt = f"""
            Based on the user's music preferences, recommend 10 songs from the available catalog.
            
            User Preferences:
            - Favorite Genres: {', '.join(context['favorite_genres'])}
            - Recent High Ratings: {context['recent_ratings']}
            - Favorite Songs: {context['favorite_songs']}
            
            Available Songs (sample): {json.dumps(context['available_songs'][:10])}
            
            Please provide recommendations in this JSON format:
            {{
                "recommendations": [
                    {{
                        "song_id": "spotify_id",
                        "score": 0.95,
                        "reason": "Short explanation why this song matches user preferences"
                    }}
                ]
            }}
            
            Focus on musical similarity, genre matching, and user's demonstrated preferences.
            """
            
            user_message = UserMessage(text=prompt)
            response = await self.gemini_chat.send_message(user_message)
            
            # Parse AI response
            try:
                # Extract JSON from response
                response_text = response.strip()
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end]
                elif "{" in response_text:
                    json_start = response_text.find("{")
                    json_end = response_text.rfind("}") + 1
                    response_text = response_text[json_start:json_end]
                
                ai_recommendations = json.loads(response_text)
                return ai_recommendations.get("recommendations", [])
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to parse AI recommendations: {e}")
                return []
                
        except Exception as e:
            logger.error(f"Error generating AI recommendations: {e}")
            return []
    
    async def explain_recommendation(self, question: str, context: Dict) -> ChatResponse:
        """Explain music recommendations using Perplexity AI or Gemini as fallback"""
        try:
            # Use Gemini for explanations as well
            system_message = """
            You are a music expert who explains why songs are recommended to users.
            Focus on musical elements like genre, tempo, instrumentation, mood, and lyrical themes.
            Provide clear, engaging explanations that help users understand their musical preferences.
            Always cite specific musical characteristics and avoid overly technical jargon.
            """
            
            explanation_chat = LlmChat(
                api_key=self.emergent_key,
                session_id="music-explanation",
                system_message=system_message
            ).with_model("gemini", "gemini-2.0-flash")
            
            # Build context-aware prompt
            prompt = f"""
            User question: {question}
            
            Context:
            - User's favorite genres: {context.get('favorite_genres', [])}
            - Recently played: {context.get('recent_songs', [])}
            - User's mood: {context.get('mood', 'not specified')}
            - Recommended song details: {context.get('song_details', {})}
            
            Please provide a detailed, engaging explanation focusing on musical characteristics.
            """
            
            user_message = UserMessage(text=prompt)
            response = await explanation_chat.send_message(user_message)
            
            return ChatResponse(
                response=response,
                confidence=0.9,
                sources=["Gemini AI Music Analysis"]
            )
            
        except Exception as e:
            logger.error(f"Error explaining recommendation: {e}")
            return ChatResponse(
                response="I'm having trouble generating an explanation right now. Please try again.",
                confidence=0.1,
                sources=[]
            )

# Initialize services
ai_service = AIService()

# Real Spotify API Integration Service
class SpotifyService:
    def __init__(self):
        self.client_id = os.environ.get('SPOTIFY_CLIENT_ID')
        self.client_secret = os.environ.get('SPOTIFY_CLIENT_SECRET')
        self.access_token = None
        self.token_expires_at = 0
        
        # Initialize spotipy client if credentials are provided
        if self.client_id and self.client_secret:
            try:
                self.sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
                    client_id=self.client_id,
                    client_secret=self.client_secret
                ))
                logger.info("Spotify API initialized successfully")
                self.use_real_spotify = True
            except Exception as e:
                logger.error(f"Failed to initialize Spotify API: {e}")
                self.use_real_spotify = False
                self.sp = None
        else:
            logger.warning("Spotify credentials not provided, using mock data")
            self.use_real_spotify = False
            self.sp = None
        
        # Fallback sample songs database (enhanced with real metadata)
        self.sample_songs = [
            # Global Pop Hits
            {
                "spotify_id": "4iV5W9uYEdYUVa79Axb7Rh",
                "name": "Bohemian Rhapsody",
                "artist": "Queen",
                "album": "A Night at the Opera",
                "genres": ["rock", "progressive rock", "classic rock"],
                "duration_ms": 355000,
                "popularity": 85,
                "external_url": "https://open.spotify.com/track/4iV5W9uYEdYUVa79Axb7Rh",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273ce4f1737bc8a646c8c4bd25a"
            },
            {
                "spotify_id": "7qiZfU4dY1lWllzX7mkmht",
                "name": "Shape of You",
                "artist": "Ed Sheeran",
                "album": "÷ (Divide)",
                "genres": ["pop", "dance pop"],
                "duration_ms": 233000,
                "popularity": 90,
                "external_url": "https://open.spotify.com/track/7qiZfU4dY1lWllzX7mkmht",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273ba5db46f4b838ef6027e6f96"
            },
            {
                "spotify_id": "4u7EnebtmKWzUH433cf5Qv",
                "name": "Blinding Lights",
                "artist": "The Weeknd",
                "album": "After Hours",
                "genres": ["pop", "synth-pop", "electronic"],
                "duration_ms": 200000,
                "popularity": 95,
                "external_url": "https://open.spotify.com/track/4u7EnebtmKWzUH433cf5Qv",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273ef6e3e2ec0b7958bbc8c6c4b"
            },
            {
                "spotify_id": "1mea3bSkSGXuIRvnydlB5b",
                "name": "Uptown Funk",
                "artist": "Mark Ronson ft. Bruno Mars",
                "album": "Uptown Special",
                "genres": ["funk", "pop", "dance"],
                "duration_ms": 270000,
                "popularity": 89,
                "external_url": "https://open.spotify.com/track/1mea3bSkSGXuIRvnydlB5b",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273dbb3dd82da45b7d7f31b1b42"
            },
            {
                "spotify_id": "4VqPOruhp5EdPBeR92t6lQ",
                "name": "Unstoppable",
                "artist": "Sia",
                "album": "This Is Acting",
                "genres": ["pop", "electropop"],
                "duration_ms": 217000,
                "popularity": 85,
                "external_url": "https://open.spotify.com/track/4VqPOruhp5EdPBeR92t6lQ",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273f926c9f9e8022fa0dfea0c6d"
            },
            {
                "spotify_id": "0G3fbPbE1vGeABXWSGKzKi",
                "name": "Someone Like You",
                "artist": "Adele",
                "album": "21",
                "genres": ["pop", "soul", "ballad"],
                "duration_ms": 285000,
                "popularity": 82,
                "external_url": "https://open.spotify.com/track/0G3fbPbE1vGeABXWSGKzKi",
                "image_url": "https://i.scdn.co/image/ab67616d0000b2735c9890c0456a3719eeecd8aa"
            },
            {
                "spotify_id": "2YJDmXz0gM9YhNlGjXTyNT",
                "name": "Billie Jean",
                "artist": "Michael Jackson",
                "album": "Thriller",
                "genres": ["pop", "funk", "r&b"],
                "duration_ms": 295000,
                "popularity": 88,
                "external_url": "https://open.spotify.com/track/2YJDmXz0gM9YhNlGjXTyNT",
                "image_url": "https://i.scdn.co/image/ab67616d0000b2734121faee8df82c526cbab2be"
            },
            {
                "spotify_id": "3LdW1z4YolBuThfJ4I9Cno",
                "name": "Bad Guy",
                "artist": "Billie Eilish",
                "album": "When We All Fall Asleep, Where Do We Go?",
                "genres": ["pop", "alternative pop", "electropop"],
                "duration_ms": 194000,
                "popularity": 92,
                "external_url": "https://open.spotify.com/track/3LdW1z4YolBuThfJ4I9Cno",
                "image_url": "https://i.scdn.co/image/ab67616d0000b27350a3147b4edd7701a876c6ce"
            },
            {
                "spotify_id": "7KXjTSCq5nL1LoYtL7XAwS",
                "name": "Watermelon Sugar",
                "artist": "Harry Styles",
                "album": "Fine Line",
                "genres": ["pop", "rock", "indie pop"],
                "duration_ms": 174000,
                "popularity": 86,
                "external_url": "https://open.spotify.com/track/7KXjTSCq5nL1LoYtL7XAwS",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273adce4d10b1e6f9e8fb9b42d6"
            },
            {
                "spotify_id": "6habFhsOp2NvshLv26DqMb",
                "name": "Levitating",
                "artist": "Dua Lipa",
                "album": "Future Nostalgia",
                "genres": ["pop", "dance pop", "disco"],
                "duration_ms": 203000,
                "popularity": 88,
                "external_url": "https://open.spotify.com/track/6habFhsOp2NvshLv26DqMb",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273ef6e3e2ec0b7958bbc8c6c4b"
            },
            
            # Rock Classics
            {
                "spotify_id": "1dfeR4HaWDbWqFHLkxsg1d",
                "name": "Hotel California",
                "artist": "Eagles",
                "album": "Hotel California",
                "genres": ["rock", "classic rock"],
                "duration_ms": 391000,
                "popularity": 80,
                "external_url": "https://open.spotify.com/track/1dfeR4HaWDbWqFHLkxsg1d",
                "image_url": "https://i.scdn.co/image/ab67616d0000b2734637341b9f507521afa9a778"
            },
            {
                "spotify_id": "3n3Ppam7vgaVa1iaRUc9Lp",
                "name": "Mr. Brightside",
                "artist": "The Killers",
                "album": "Hot Fuss",
                "genres": ["alternative rock", "indie rock"],
                "duration_ms": 222000,
                "popularity": 83,
                "external_url": "https://open.spotify.com/track/3n3Ppam7vgaVa1iaRUc9Lp",
                "image_url": "https://i.scdn.co/image/ab67616d0000b2734b2fb6b6ca8f0b0d24fd14e7"
            },
            {
                "spotify_id": "5uQ0vKy2973Y9IUCd1wMEF",
                "name": "Sweet Child O' Mine",
                "artist": "Guns N' Roses",
                "album": "Appetite for Destruction",
                "genres": ["hard rock", "classic rock"],
                "duration_ms": 356000,
                "popularity": 85,
                "external_url": "https://open.spotify.com/track/5uQ0vKy2973Y9IUCd1wMEF",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273af07dc851962508661bbcfce"
            },
            {
                "spotify_id": "4gMgiXfqyzZLMhsksGmbQV",
                "name": "Another Brick in the Wall",
                "artist": "Pink Floyd",
                "album": "The Wall",
                "genres": ["progressive rock", "psychedelic rock"],
                "duration_ms": 238000,
                "popularity": 84,
                "external_url": "https://open.spotify.com/track/4gMgiXfqyzZLMhsksGmbQV",
                "image_url": "https://i.scdn.co/image/ab67616d0000b2735d48e2f56d691f9a4e4b0baf"
            },
            {
                "spotify_id": "2UkMeKrQe0UYsRLpCXL4xp",
                "name": "Thunderstruck",
                "artist": "AC/DC",
                "album": "The Razors Edge",
                "genres": ["hard rock", "classic rock"],
                "duration_ms": 292000,
                "popularity": 82,
                "external_url": "https://open.spotify.com/track/2UkMeKrQe0UYsRLpCXL4xp",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273b9c8bf86c4d4f3a1f8e8e5ad"
            },
            {
                "spotify_id": "3RiPr603aXAMBnaqUOL2yf",
                "name": "Stairway to Heaven",
                "artist": "Led Zeppelin",
                "album": "Led Zeppelin IV",
                "genres": ["rock", "hard rock", "classic rock"],
                "duration_ms": 482000,
                "popularity": 87,
                "external_url": "https://open.spotify.com/track/3RiPr603aXAMBnaqUOL2yf",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273c8a11e48c91a982d086afc69"
            },
            {
                "spotify_id": "6mFkJmJqdDVQ1REhVfGgd1",
                "name": "We Will Rock You",
                "artist": "Queen",
                "album": "News of the World",
                "genres": ["rock", "arena rock", "classic rock"],
                "duration_ms": 122000,
                "popularity": 89,
                "external_url": "https://open.spotify.com/track/6mFkJmJqdDVQ1REhVfGgd1",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273e4e0d9c9a447b6c4c91f8acc"
            },
            
            # Hip-Hop & Rap
            {
                "spotify_id": "5ChkMS8OtdzJeqyybCc9R5",
                "name": "Lose Yourself",
                "artist": "Eminem",
                "album": "8 Mile Soundtrack",
                "genres": ["hip-hop", "rap"],
                "duration_ms": 326000,
                "popularity": 87,
                "external_url": "https://open.spotify.com/track/5ChkMS8OtdzJeqyybCc9R5",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273076b6cd6e2dfb7b1977bff9e"
            },
            {
                "spotify_id": "6kex4EcnDkEacGU8NId0AC",
                "name": "Old Town Road",
                "artist": "Lil Nas X ft. Billy Ray Cyrus",
                "album": "7 EP",
                "genres": ["country rap", "hip-hop"],
                "duration_ms": 157000,
                "popularity": 91,
                "external_url": "https://open.spotify.com/track/6kex4EcnDkEacGU8NId0AC",
                "image_url": "https://i.scdn.co/image/ab67616d0000b2739b6ca98409164d5c48f9a9e6"
            },
            {
                "spotify_id": "4LRPiXqCikLlN15c3yImP7",
                "name": "HUMBLE.",
                "artist": "Kendrick Lamar",
                "album": "DAMN.",
                "genres": ["hip-hop", "rap", "conscious rap"],
                "duration_ms": 177000,
                "popularity": 85,
                "external_url": "https://open.spotify.com/track/4LRPiXqCikLlN15c3yImP7",
                "image_url": "https://i.scdn.co/image/ab67616d0000b2738b52c6b9bc4e0e0e9d0c0f8d"
            },
            {
                "spotify_id": "2YpeDb67231RjR0MgVLzsG",
                "name": "God's Plan",
                "artist": "Drake",
                "album": "Scorpion",
                "genres": ["hip-hop", "rap", "pop rap"],
                "duration_ms": 198000,
                "popularity": 88,
                "external_url": "https://open.spotify.com/track/2YpeDb67231RjR0MgVLzsG",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273f907de96b9a4fbc04accc0d5"
            },
            {
                "spotify_id": "3DK6m7It6Pw857FcQftMds",
                "name": "Stronger",
                "artist": "Kanye West",
                "album": "Graduation",
                "genres": ["hip-hop", "rap", "electronic rap"],
                "duration_ms": 312000,
                "popularity": 84,
                "external_url": "https://open.spotify.com/track/3DK6m7It6Pw857FcQftMds",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273926f158b4c47f22dca3e7b42"
            },
            {
                "spotify_id": "7ouMYWpwJ422jRcDASZB7P",
                "name": "Crazy in Love",
                "artist": "Beyoncé ft. Jay-Z",
                "album": "Dangerously in Love",
                "genres": ["r&b", "hip-hop", "pop"],
                "duration_ms": 236000,
                "popularity": 86,
                "external_url": "https://open.spotify.com/track/7ouMYWpwJ422jRcDASZB7P",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273ff5429b5ec2e26e54345d4e8"
            },
            
            # Electronic & Dance
            {
                "spotify_id": "7BKLCZ1jbUBVqRi2FVlTVw",
                "name": "Closer",
                "artist": "The Chainsmokers ft. Halsey",
                "album": "Collage EP",
                "genres": ["electronic", "pop", "dance"],
                "duration_ms": 244000,
                "popularity": 86,
                "external_url": "https://open.spotify.com/track/7BKLCZ1jbUBVqRi2FVlTVw",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273495ce6da9aeb159e94eaa453"
            },
            {
                "spotify_id": "4aVuWgvD0X63hcOCnZtNFA",
                "name": "Wake Me Up",
                "artist": "Avicii",
                "album": "True",
                "genres": ["electronic", "edm", "progressive house"],
                "duration_ms": 247000,
                "popularity": 87,
                "external_url": "https://open.spotify.com/track/4aVuWgvD0X63hcOCnZtNFA",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273e14f11f796cef9f9a82691a7"
            },
            {
                "spotify_id": "1uNFoZAHBGtllmzznpCI3s",
                "name": "Titanium",
                "artist": "David Guetta ft. Sia",
                "album": "Nothing but the Beat",
                "genres": ["electronic", "dance", "edm"],
                "duration_ms": 245000,
                "popularity": 85,
                "external_url": "https://open.spotify.com/track/1uNFoZAHBGtllmzznpCI3s",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273ee5fe0e3e3bf87e6dcd4e6e9"
            },
            {
                "spotify_id": "5YJ9c9kmKJdQKqKLtxzKsX",
                "name": "Bangarang",
                "artist": "Skrillex ft. Sirah",
                "album": "Bangarang EP",
                "genres": ["dubstep", "electronic", "bass"],
                "duration_ms": 215000,
                "popularity": 82,
                "external_url": "https://open.spotify.com/track/5YJ9c9kmKJdQKqKLtxzKsX",
                "image_url": "https://i.scdn.co/image/ab67616d0000b2737e2b0c8a0d5d2e3e1d2e6e7f"
            },
            {
                "spotify_id": "0VjIjW4GlUZAMYd2vXMi3b",
                "name": "Levels",
                "artist": "Avicii",
                "album": "Levels",
                "genres": ["electronic", "progressive house", "edm"],
                "duration_ms": 202000,
                "popularity": 88,
                "external_url": "https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273e14f11f796cef9f9a82691a7"
            },
            {
                "spotify_id": "2dpaYNEQHiRxtZbfNsse99",
                "name": "Animals",
                "artist": "Martin Garrix",
                "album": "Animals",
                "genres": ["electronic", "big room", "edm"],
                "duration_ms": 302000,
                "popularity": 84,
                "external_url": "https://open.spotify.com/track/2dpaYNEQHiRxtZbfNsse99",
                "image_url": "https://i.scdn.co/image/ab67616d0000b2733e6e5f4e8a8e4e3e2e3e4e5f"
            },
            
            # Jazz & Blues
            {
                "spotify_id": "5ZrrZ5loApAEvFwjCDbtX7",
                "name": "What a Wonderful World",
                "artist": "Louis Armstrong",
                "album": "What a Wonderful World",
                "genres": ["jazz", "traditional pop", "blues"],
                "duration_ms": 137000,
                "popularity": 78,
                "external_url": "https://open.spotify.com/track/5ZrrZ5loApAEvFwjCDbtX7",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273b8c8b8c8b8c8b8c8b8c8b8c8"
            },
            {
                "spotify_id": "6JV2JOEocTTpE5M1xgBvvL",
                "name": "Fly Me to the Moon",
                "artist": "Frank Sinatra",
                "album": "It Might as Well Be Swing",
                "genres": ["jazz", "swing", "traditional pop"],
                "duration_ms": 148000,
                "popularity": 80,
                "external_url": "https://open.spotify.com/track/6JV2JOEocTTpE5M1xgBvvL",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273a9a9a9a9a9a9a9a9a9a9a9a9"
            },
            {
                "spotify_id": "1rqAAGrswSKoRGgjO2hv1d",
                "name": "Take Five",
                "artist": "Dave Brubeck Quartet",
                "album": "Time Out",
                "genres": ["jazz", "cool jazz", "bebop"],
                "duration_ms": 324000,
                "popularity": 75,
                "external_url": "https://open.spotify.com/track/1rqAAGrswSKoRGgjO2hv1d",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273c1c1c1c1c1c1c1c1c1c1c1c1"
            },
            {
                "spotify_id": "3WMj8moIAXJhHUUnsfw0Ni",
                "name": "The Thrill Is Gone",
                "artist": "B.B. King",
                "album": "Completely Well",
                "genres": ["blues", "electric blues", "chicago blues"],
                "duration_ms": 308000,
                "popularity": 72,
                "external_url": "https://open.spotify.com/track/3WMj8moIAXJhHUUnsfw0Ni",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273d2d2d2d2d2d2d2d2d2d2d2d2"
            },
            
            # Country
            {
                "spotify_id": "4zAydkypV3IlMOOeINnhSP",
                "name": "Friends in Low Places",
                "artist": "Garth Brooks",
                "album": "No Fences",
                "genres": ["country", "country pop"],
                "duration_ms": 260000,
                "popularity": 79,
                "external_url": "https://open.spotify.com/track/4zAydkypV3IlMOOeINnhSP",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273e3e3e3e3e3e3e3e3e3e3e3e3"
            },
            {
                "spotify_id": "0lPkmfk6fR2vOONI2sSCx6",
                "name": "Sweet Caroline",
                "artist": "Neil Diamond",
                "album": "Brother Love's Travelling Salvation Show",
                "genres": ["country", "folk", "soft rock"],
                "duration_ms": 201000,
                "popularity": 81,
                "external_url": "https://open.spotify.com/track/0lPkmfk6fR2vOONI2sSCx6",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273f4f4f4f4f4f4f4f4f4f4f4f4"
            },
            {
                "spotify_id": "5KKMoVgEJGk8KW2f9KdKxC",
                "name": "Jolene",
                "artist": "Dolly Parton",
                "album": "Jolene",
                "genres": ["country", "country pop", "folk"],
                "duration_ms": 178000,
                "popularity": 77,
                "external_url": "https://open.spotify.com/track/5KKMoVgEJGk8KW2f9KdKxC",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273f5f5f5f5f5f5f5f5f5f5f5f5"
            },
            
            # R&B & Soul
            {
                "spotify_id": "7Lt5ZXbJJsq4uFDMi1Vsr3",
                "name": "Respect",
                "artist": "Aretha Franklin",
                "album": "I Never Loved a Man the Way I Love You",
                "genres": ["soul", "r&b", "funk"],
                "duration_ms": 147000,
                "popularity": 76,
                "external_url": "https://open.spotify.com/track/7Lt5ZXbJJsq4uFDMi1Vsr3",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273f6f6f6f6f6f6f6f6f6f6f6f6"
            },
            {
                "spotify_id": "3Y3G2M8d4Dh4k7QxYHQC6t",
                "name": "Superstition",
                "artist": "Stevie Wonder",
                "album": "Talking Book",
                "genres": ["soul", "r&b", "funk"],
                "duration_ms": 245000,
                "popularity": 78,
                "external_url": "https://open.spotify.com/track/3Y3G2M8d4Dh4k7QxYHQC6t",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273f7f7f7f7f7f7f7f7f7f7f7f7"
            },
            {
                "spotify_id": "5ChkMS8OtdzJeqyybCc9R6",
                "name": "I Will Always Love You",
                "artist": "Whitney Houston",
                "album": "The Bodyguard Soundtrack",
                "genres": ["r&b", "soul", "pop"],
                "duration_ms": 273000,
                "popularity": 83,
                "external_url": "https://open.spotify.com/track/5ChkMS8OtdzJeqyybCc9R6",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273f8f8f8f8f8f8f8f8f8f8f8f8"
            },
            
            # Alternative & Indie
            {
                "spotify_id": "7xzU7hKDOEGjE3pE2ygU3c",
                "name": "Creep",
                "artist": "Radiohead",
                "album": "Pablo Honey",
                "genres": ["alternative rock", "grunge", "britpop"],
                "duration_ms": 238000,
                "popularity": 82,
                "external_url": "https://open.spotify.com/track/7xzU7hKDOEGjE3pE2ygU3c",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273f9f9f9f9f9f9f9f9f9f9f9f9"
            },
            {
                "spotify_id": "4Dvkj6JhhA12EX05fT7y2e",
                "name": "Smells Like Teen Spirit",
                "artist": "Nirvana",
                "album": "Nevermind",
                "genres": ["grunge", "alternative rock"],
                "duration_ms": 301000,
                "popularity": 84,
                "external_url": "https://open.spotify.com/track/4Dvkj6JhhA12EX05fT7y2e",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273fafafafafafafafafafafafafa"
            },
            {
                "spotify_id": "5uQ0vKy2973Y9IUCd1wMEG",
                "name": "Wonderwall",
                "artist": "Oasis",
                "album": "(What's the Story) Morning Glory?",
                "genres": ["britpop", "alternative rock"],
                "duration_ms": 258000,
                "popularity": 85,
                "external_url": "https://open.spotify.com/track/5uQ0vKy2973Y9IUCd1wMEG",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273fbfbfbfbfbfbfbfbfbfbfbfb"
            },
            
            # Classical
            {
                "spotify_id": "2takcwOaAZWiXQijPHIx7B",
                "name": "Canon in D",
                "artist": "Johann Pachelbel",
                "album": "Classical Masterpieces",
                "genres": ["classical", "baroque"],
                "duration_ms": 355000,
                "popularity": 68,
                "external_url": "https://open.spotify.com/track/2takcwOaAZWiXQijPHIx7B",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273fcfcfcfcfcfcfcfcfcfcfcfc"
            },
            {
                "spotify_id": "4VxoN8jOQ5d3ZFrEhiIYqL",
                "name": "Moonlight Sonata",
                "artist": "Ludwig van Beethoven",
                "album": "Beethoven Piano Sonatas",
                "genres": ["classical", "romantic"],
                "duration_ms": 897000,
                "popularity": 71,
                "external_url": "https://open.spotify.com/track/4VxoN8jOQ5d3ZFrEhiIYqL",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273fdfdfdfdfdfdfdfdfdfdfdfd"
            },
            {
                "spotify_id": "6oMLxXMKvmJG5iRw5Jf6MO",
                "name": "The Four Seasons - Spring",
                "artist": "Antonio Vivaldi",
                "album": "The Four Seasons",
                "genres": ["classical", "baroque"],
                "duration_ms": 623000,
                "popularity": 69,
                "external_url": "https://open.spotify.com/track/6oMLxXMKvmJG5iRw5Jf6MO",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273fefefefefefefefefefefefefe"
            },
            
            # Reggae
            {
                "spotify_id": "6JV2JOEocTTpE5M1xgBvvM",
                "name": "No Woman No Cry",
                "artist": "Bob Marley & The Wailers",
                "album": "Natty Dread",
                "genres": ["reggae", "roots reggae"],
                "duration_ms": 423000,
                "popularity": 80,
                "external_url": "https://open.spotify.com/track/6JV2JOEocTTpE5M1xgBvvM",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273ffffffffffffffffffffffff"
            },
            {
                "spotify_id": "3tbwb5pUbJU6V6RGqvUgpl",
                "name": "Three Little Birds",
                "artist": "Bob Marley & The Wailers",
                "album": "Exodus",
                "genres": ["reggae", "roots reggae"],
                "duration_ms": 180000,
                "popularity": 78,
                "external_url": "https://open.spotify.com/track/3tbwb5pUbJU6V6RGqvUgpl",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273000000000000000000000000"
            },
            
            # Bollywood & Hindi Songs
            {
                "spotify_id": "1A2B3C4D5E6F7G8H9I0J1K",
                "name": "Kesariya",
                "artist": "Arijit Singh",
                "album": "Brahmastra",
                "genres": ["bollywood", "romantic", "hindi"],
                "duration_ms": 269000,
                "popularity": 95,
                "external_url": "https://open.spotify.com/track/1A2B3C4D5E6F7G8H9I0J1K",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273kesariya123456789"
            },
            {
                "spotify_id": "2B3C4D5E6F7G8H9I0J1K2L",
                "name": "Jai Ho",
                "artist": "A.R. Rahman",
                "album": "Slumdog Millionaire",
                "genres": ["bollywood", "soundtrack", "hindi"],
                "duration_ms": 342000,
                "popularity": 88,
                "external_url": "https://open.spotify.com/track/2B3C4D5E6F7G8H9I0J1K2L",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273jaiho123456789"
            },
            {
                "spotify_id": "3C4D5E6F7G8H9I0J1K2L3M",
                "name": "Raataan Lambiyan",
                "artist": "Tanishk Bagchi, Jubin Nautiyal & Asees Kaur",
                "album": "Shershaah",
                "genres": ["bollywood", "romantic", "hindi"],
                "duration_ms": 298000,
                "popularity": 92,
                "external_url": "https://open.spotify.com/track/3C4D5E6F7G8H9I0J1K2L3M",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273raatan123456789"
            },
            {
                "spotify_id": "4D5E6F7G8H9I0J1K2L3M4N",
                "name": "Apna Time Aayega",
                "artist": "Ranveer Singh, DIVINE & Dub Sharma",
                "album": "Gully Boy",
                "genres": ["bollywood", "hip-hop", "hindi rap"],
                "duration_ms": 178000,
                "popularity": 89,
                "external_url": "https://open.spotify.com/track/4D5E6F7G8H9I0J1K2L3M4N",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273apnatime123456789"
            },
            {
                "spotify_id": "5E6F7G8H9I0J1K2L3M4N5O",
                "name": "Tum Hi Ho",
                "artist": "Arijit Singh",
                "album": "Aashiqui 2",
                "genres": ["bollywood", "romantic", "hindi"],
                "duration_ms": 262000,
                "popularity": 94,
                "external_url": "https://open.spotify.com/track/5E6F7G8H9I0J1K2L3M4N5O",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273tumhiho123456789"
            },
            {
                "spotify_id": "6F7G8H9I0J1K2L3M4N5O6P",
                "name": "Gulabi Aankhen",
                "artist": "Mohammed Rafi",
                "album": "The Train",
                "genres": ["bollywood", "classic", "hindi"],
                "duration_ms": 234000,
                "popularity": 85,
                "external_url": "https://open.spotify.com/track/6F7G8H9I0J1K2L3M4N5O6P",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273gulabi123456789"
            },
            {
                "spotify_id": "7G8H9I0J1K2L3M4N5O6P7Q",
                "name": "Dil Chahta Hai",
                "artist": "Shankar-Ehsaan-Loy",
                "album": "Dil Chahta Hai",
                "genres": ["bollywood", "pop", "hindi"],
                "duration_ms": 275000,
                "popularity": 87,
                "external_url": "https://open.spotify.com/track/7G8H9I0J1K2L3M4N5O6P7Q",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273dilchahta123456789"
            },
            {
                "spotify_id": "8H9I0J1K2L3M4N5O6P7Q8R",
                "name": "Ve Maahi",
                "artist": "Arijit Singh & Asees Kaur",
                "album": "Kesari",
                "genres": ["bollywood", "romantic", "hindi"],
                "duration_ms": 243000,
                "popularity": 88,
                "external_url": "https://open.spotify.com/track/8H9I0J1K2L3M4N5O6P7Q8R",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273vemaahi123456789"
            },
            
            # Tamil Songs
            {
                "spotify_id": "9I0J1K2L3M4N5O6P7Q8R9S",
                "name": "Oo Antava",
                "artist": "Indravathi Chauhan",
                "album": "Pushpa",
                "genres": ["tollywood", "dance", "tamil"],
                "duration_ms": 198000,
                "popularity": 93,
                "external_url": "https://open.spotify.com/track/9I0J1K2L3M4N5O6P7Q8R9S",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273ooantava123456789"
            },
            {
                "spotify_id": "0J1K2L3M4N5O6P7Q8R9S0T",
                "name": "Naatu Naatu",
                "artist": "Kala Bhairava, Rahul Sipligunj",
                "album": "RRR",
                "genres": ["tollywood", "dance", "telugu"],
                "duration_ms": 287000,
                "popularity": 96,
                "external_url": "https://open.spotify.com/track/0J1K2L3M4N5O6P7Q8R9S0T",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273naatu123456789"
            },
            {
                "spotify_id": "1K2L3M4N5O6P7Q8R9S0T1U",
                "name": "Rowdy Baby",
                "artist": "Dhanush, Dhee",
                "album": "Maari 2",
                "genres": ["kollywood", "dance", "tamil"],
                "duration_ms": 213000,
                "popularity": 89,
                "external_url": "https://open.spotify.com/track/1K2L3M4N5O6P7Q8R9S0T1U",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273rowdy123456789"
            },
            {
                "spotify_id": "2L3M4N5O6P7Q8R9S0T1U2V",
                "name": "Vaathi Coming",
                "artist": "Anirudh Ravichander",
                "album": "Master",
                "genres": ["kollywood", "mass", "tamil"],
                "duration_ms": 256000,
                "popularity": 91,
                "external_url": "https://open.spotify.com/track/2L3M4N5O6P7Q8R9S0T1U2V",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273vaathi123456789"
            },
            {
                "spotify_id": "3M4N5O6P7Q8R9S0T1U2V3W",
                "name": "Sulthan",
                "artist": "Anirudh Ravichander",
                "album": "Master",
                "genres": ["kollywood", "action", "tamil"],
                "duration_ms": 298000,
                "popularity": 85,
                "external_url": "https://open.spotify.com/track/3M4N5O6P7Q8R9S0T1U2V3W",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273sulthan123456789"
            },
            
            # Punjabi Songs
            {
                "spotify_id": "4N5O6P7Q8R9S0T1U2V3W4X",
                "name": "295",
                "artist": "Sidhu Moose Wala",
                "album": "PBX 1",
                "genres": ["punjabi", "hip-hop", "desi"],
                "duration_ms": 234000,
                "popularity": 92,
                "external_url": "https://open.spotify.com/track/4N5O6P7Q8R9S0T1U2V3W4X",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273295song123456789"
            },
            {
                "spotify_id": "5O6P7Q8R9S0T1U2V3W4X5Y",
                "name": "Laembadgini",
                "artist": "Diljit Dosanjh",
                "album": "Back 2 Basics",
                "genres": ["punjabi", "pop", "desi"],
                "duration_ms": 198000,
                "popularity": 88,
                "external_url": "https://open.spotify.com/track/5O6P7Q8R9S0T1U2V3W4X5Y",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273lambo123456789"
            },
            {
                "spotify_id": "6P7Q8R9S0T1U2V3W4X5Y6Z",
                "name": "Goosebumps",
                "artist": "Honey Singh",
                "album": "Glory",
                "genres": ["punjabi", "hip-hop", "desi"],
                "duration_ms": 267000,
                "popularity": 90,
                "external_url": "https://open.spotify.com/track/6P7Q8R9S0T1U2V3W4X5Y6Z",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273goose123456789"
            },
            {
                "spotify_id": "7Q8R9S0T1U2V3W4X5Y6Z7A",
                "name": "Brown Munde",
                "artist": "AP Dhillon, Gurinder Gill, Shinda Kahlon",
                "album": "Brown Munde",
                "genres": ["punjabi", "drill", "desi hip-hop"],
                "duration_ms": 189000,
                "popularity": 94,
                "external_url": "https://open.spotify.com/track/7Q8R9S0T1U2V3W4X5Y6Z7A",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273brown123456789"
            },
            
            # Malayalam Songs
            {
                "spotify_id": "8R9S0T1U2V3W4X5Y6Z7A8B",
                "name": "Parayuvaan",
                "artist": "Vineeth Sreenivasan",
                "album": "Hridayam",
                "genres": ["mollywood", "romantic", "malayalam"],
                "duration_ms": 278000,
                "popularity": 87,
                "external_url": "https://open.spotify.com/track/8R9S0T1U2V3W4X5Y6Z7A8B",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273parayuvaan123456789"
            },
            {
                "spotify_id": "9S0T1U2V3W4X5Y6Z7A8B9C",
                "name": "Darshana",
                "artist": "Vineeth Sreenivasan, Anne Amie",
                "album": "Hridayam",
                "genres": ["mollywood", "romantic", "malayalam"],
                "duration_ms": 245000,
                "popularity": 84,
                "external_url": "https://open.spotify.com/track/9S0T1U2V3W4X5Y6Z7A8B9C",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273darshana123456789"
            },
            
            # Marathi Songs
            {
                "spotify_id": "0T1U2V3W4X5Y6Z7A8B9C0D",
                "name": "Zingaat",
                "artist": "Ajay Gogavale",
                "album": "Sairat",
                "genres": ["marathi", "folk", "dance"],
                "duration_ms": 267000,
                "popularity": 86,
                "external_url": "https://open.spotify.com/track/0T1U2V3W4X5Y6Z7A8B9C0D",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273zingaat123456789"
            },
            
            # Bengali Songs
            {
                "spotify_id": "1U2V3W4X5Y6Z7A8B9C0D1E",
                "name": "Kahaani Suno 2.0",
                "artist": "Kaifi Khalil",
                "album": "Kahaani Suno",
                "genres": ["bengali", "folk", "modern"],
                "duration_ms": 198000,
                "popularity": 83,
                "external_url": "https://open.spotify.com/track/1U2V3W4X5Y6Z7A8B9C0D1E",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273kahaani123456789"
            },
            
            # Gujarati Songs
            {
                "spotify_id": "2V3W4X5Y6Z7A8B9C0D1E2F",
                "name": "Chogada",
                "artist": "Darshan Raval, Asees Kaur",
                "album": "Loveyatri",
                "genres": ["gujarati", "folk", "dance"],
                "duration_ms": 234000,
                "popularity": 85,
                "external_url": "https://open.spotify.com/track/2V3W4X5Y6Z7A8B9C0D1E2F",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273chogada123456789"
            },
            
            # More Recent Bollywood Hits
            {
                "spotify_id": "3W4X5Y6Z7A8B9C0D1E2F3G",
                "name": "Malang Sajna",
                "artist": "Sachet Tandon, Parampara Thakur",
                "album": "Malang",
                "genres": ["bollywood", "romantic", "hindi"],
                "duration_ms": 256000,
                "popularity": 89,
                "external_url": "https://open.spotify.com/track/3W4X5Y6Z7A8B9C0D1E2F3G",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273malang123456789"
            },
            {
                "spotify_id": "4X5Y6Z7A8B9C0D1E2F3G4H",
                "name": "Shayad",
                "artist": "Arijit Singh",
                "album": "Love Aaj Kal",
                "genres": ["bollywood", "romantic", "hindi"],
                "duration_ms": 298000,
                "popularity": 91,
                "external_url": "https://open.spotify.com/track/4X5Y6Z7A8B9C0D1E2F3G4H",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273shayad123456789"
            },
            {
                "spotify_id": "5Y6Z7A8B9C0D1E2F3G4H5I",
                "name": "Bekhayali",
                "artist": "Sachet Tandon",
                "album": "Kabir Singh",
                "genres": ["bollywood", "romantic", "hindi"],
                "duration_ms": 387000,
                "popularity": 93,
                "external_url": "https://open.spotify.com/track/5Y6Z7A8B9C0D1E2F3G4H5I",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273bekhayali123456789"
            },
            {
                "spotify_id": "6Z7A8B9C0D1E2F3G4H5I6J",
                "name": "Ghungroo",
                "artist": "Arijit Singh, Shilpa Rao",
                "album": "War",
                "genres": ["bollywood", "dance", "hindi"],
                "duration_ms": 298000,
                "popularity": 88,
                "external_url": "https://open.spotify.com/track/6Z7A8B9C0D1E2F3G4H5I6J",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273ghungroo123456789"
            },
            
            # Indian Classical & Folk
            {
                "spotify_id": "7A8B9C0D1E2F3G4H5I6J7K",
                "name": "Raag Yaman",
                "artist": "Pandit Jasraj",
                "album": "Classical Ragas",
                "genres": ["indian classical", "hindustani", "vocal"],
                "duration_ms": 1248000,
                "popularity": 72,
                "external_url": "https://open.spotify.com/track/7A8B9C0D1E2F3G4H5I6J7K",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273raagyaman123456789"
            },
            {
                "spotify_id": "8B9C0D1E2F3G4H5I6J7K8L",
                "name": "Vande Mataram",
                "artist": "A.R. Rahman",
                "album": "Vande Mataram",
                "genres": ["patriotic", "indian classical", "hindi"],
                "duration_ms": 425000,
                "popularity": 86,
                "external_url": "https://open.spotify.com/track/8B9C0D1E2F3G4H5I6J7K8L",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273vande123456789"
            },
            
            # Regional Language Hits
            {
                "spotify_id": "9C0D1E2F3G4H5I6J7K8L9M",
                "name": "Butta Bomma",
                "artist": "Armaan Malik",
                "album": "Ala Vaikunthapurramuloo",
                "genres": ["tollywood", "romantic", "telugu"],
                "duration_ms": 267000,
                "popularity": 90,
                "external_url": "https://open.spotify.com/track/9C0D1E2F3G4H5I6J7K8L9M",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273butta123456789"
            },
            {
                "spotify_id": "0D1E2F3G4H5I6J7K8L9M0N",
                "name": "Inkem Inkem",
                "artist": "Sid Sriram",
                "album": "Geetha Govindam",
                "genres": ["tollywood", "romantic", "telugu"],
                "duration_ms": 234000,
                "popularity": 87,
                "external_url": "https://open.spotify.com/track/0D1E2F3G4H5I6J7K8L9M0N",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273inkem123456789"
            },
            
            # International Pop & Recent Hits
            {
                "spotify_id": "1E2F3G4H5I6J7K8L9M0N1O",
                "name": "Anti-Hero",
                "artist": "Taylor Swift",
                "album": "Midnights",
                "genres": ["pop", "indie pop"],
                "duration_ms": 200000,
                "popularity": 96,
                "external_url": "https://open.spotify.com/track/1E2F3G4H5I6J7K8L9M0N1O",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273antihero123456789"
            },
            {
                "spotify_id": "2F3G4H5I6J7K8L9M0N1O2P",
                "name": "As It Was",
                "artist": "Harry Styles",
                "album": "Harry's House",
                "genres": ["pop", "rock"],
                "duration_ms": 167000,
                "popularity": 94,
                "external_url": "https://open.spotify.com/track/2F3G4H5I6J7K8L9M0N1O2P",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273asitwas123456789"
            },
            {
                "spotify_id": "3G4H5I6J7K8L9M0N1O2P3Q",
                "name": "Unholy",
                "artist": "Sam Smith ft. Kim Petras",
                "album": "Gloria",
                "genres": ["pop", "dance pop"],
                "duration_ms": 156000,
                "popularity": 93,
                "external_url": "https://open.spotify.com/track/3G4H5I6J7K8L9M0N1O2P3Q",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273unholy123456789"
            },
            {
                "spotify_id": "4H5I6J7K8L9M0N1O2P3Q4R",
                "name": "Flowers",
                "artist": "Miley Cyrus",
                "album": "Endless Summer Vacation",
                "genres": ["pop", "dance pop"],
                "duration_ms": 200000,
                "popularity": 95,
                "external_url": "https://open.spotify.com/track/4H5I6J7K8L9M0N1O2P3Q4R",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273flowers123456789"
            },
            
            # K-Pop Hits
            {
                "spotify_id": "5I6J7K8L9M0N1O2P3Q4R5S",
                "name": "Dynamite",
                "artist": "BTS",
                "album": "BE",
                "genres": ["k-pop", "pop", "dance pop"],
                "duration_ms": 199000,
                "popularity": 92,
                "external_url": "https://open.spotify.com/track/5I6J7K8L9M0N1O2P3Q4R5S",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273dynamite123456789"
            },
            {
                "spotify_id": "6J7K8L9M0N1O2P3Q4R5S6T",
                "name": "Gangnam Style",
                "artist": "PSY",
                "album": "PSY 6 (Six Rules), Part 1",
                "genres": ["k-pop", "dance", "electronic"],
                "duration_ms": 219000,
                "popularity": 85,
                "external_url": "https://open.spotify.com/track/6J7K8L9M0N1O2P3Q4R5S6T",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273gangnam123456789"
            },
            
            # Latin Hits
            {
                "spotify_id": "11dFlYqUUdU0I4xBWcpRKY",
                "name": "Despacito",
                "artist": "Luis Fonsi ft. Daddy Yankee",
                "album": "Vida",
                "genres": ["latin", "reggaeton", "latin pop"],
                "duration_ms": 229000,
                "popularity": 89,
                "external_url": "https://open.spotify.com/track/11dFlYqUUdU0I4xBWcpRKY",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273111111111111111111111111"
            },
            {
                "spotify_id": "6kTgHN8NQVfGp6niBiLQ6t",
                "name": "Macarena",
                "artist": "Los Del Rio",
                "album": "A mí me gusta",
                "genres": ["latin", "dance", "latin pop"],
                "duration_ms": 253000,
                "popularity": 76,
                "external_url": "https://open.spotify.com/track/6kTgHN8NQVfGp6niBiLQ6t",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273222222222222222222222222"
            },
            {
                "spotify_id": "7K8L9M0N1O2P3Q4R5S6T7U",
                "name": "Con Altura",
                "artist": "Rosalía & J Balvin ft. El Guincho",
                "album": "Con Altura",
                "genres": ["latin", "reggaeton", "spanish pop"],
                "duration_ms": 162000,
                "popularity": 88,
                "external_url": "https://open.spotify.com/track/7K8L9M0N1O2P3Q4R5S6T7U",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273conaltura123456789"
            },
            
            # More Hip-Hop
            {
                "spotify_id": "8L9M0N1O2P3Q4R5S6T7U8V",
                "name": "Industry Baby",
                "artist": "Lil Nas X & Jack Harlow",
                "album": "MONTERO",
                "genres": ["hip-hop", "rap", "pop rap"],
                "duration_ms": 212000,
                "popularity": 90,
                "external_url": "https://open.spotify.com/track/8L9M0N1O2P3Q4R5S6T7U8V",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273industry123456789"
            },
            {
                "spotify_id": "9M0N1O2P3Q4R5S6T7U8V9W",
                "name": "STAY",
                "artist": "The Kid LAROI & Justin Bieber",
                "album": "F*CK LOVE 3: OVER YOU",
                "genres": ["pop", "hip-hop", "pop rap"],
                "duration_ms": 141000,
                "popularity": 93,
                "external_url": "https://open.spotify.com/track/9M0N1O2P3Q4R5S6T7U8V9W",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273stay123456789"
            }
        ]
        
        # Create comprehensive playlists including Indian music
        self.playlists = {
            "Top 50 Global": {
                "id": "playlist_global_top50",
                "name": "Top 50 Global",
                "description": "The most played songs worldwide",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273e3e3e3e3e3e3e3e3e3e3e3e3",
                "songs": self.get_top_songs_by_popularity(50)
            },
            "Bollywood Hits": {
                "id": "playlist_bollywood_hits",
                "name": "Bollywood Hits",
                "description": "The biggest Bollywood and Hindi songs",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273bollywood123456789",
                "songs": self.get_songs_by_genre("bollywood") + self.get_songs_by_genre("hindi")
            },
            "South Indian Hits": {
                "id": "playlist_south_indian",
                "name": "South Indian Hits",
                "description": "Popular Tamil, Telugu, Malayalam songs",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273south123456789",
                "songs": self.get_songs_by_genre("tamil") + self.get_songs_by_genre("telugu") + self.get_songs_by_genre("malayalam") + self.get_songs_by_genre("kollywood") + self.get_songs_by_genre("tollywood")
            },
            "Punjabi Hits": {
                "id": "playlist_punjabi_hits",
                "name": "Punjabi Hits",
                "description": "Best of Punjabi music and desi hip-hop",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273punjabi123456789",
                "songs": self.get_songs_by_genre("punjabi") + self.get_songs_by_genre("desi")
            },
            "Indian Classical": {
                "id": "playlist_indian_classical",
                "name": "Indian Classical",
                "description": "Traditional Indian classical and folk music",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273classical123456789",
                "songs": self.get_songs_by_genre("indian classical") + self.get_songs_by_genre("hindustani") + self.get_songs_by_genre("folk")
            },
            "Regional India": {
                "id": "playlist_regional_india",
                "name": "Regional India",
                "description": "Marathi, Bengali, Gujarati and other regional hits",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273regional123456789",
                "songs": self.get_songs_by_genre("marathi") + self.get_songs_by_genre("bengali") + self.get_songs_by_genre("gujarati")
            },
            "Pop Hits": {
                "id": "playlist_pop_hits",
                "name": "Pop Hits",
                "description": "The biggest pop songs of all time",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273f4f4f4f4f4f4f4f4f4f4f4f4",
                "songs": self.get_songs_by_genre("pop")
            },
            "Rock Classics": {
                "id": "playlist_rock_classics",
                "name": "Rock Classics",
                "description": "Timeless rock anthems",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273f5f5f5f5f5f5f5f5f5f5f5f5",
                "songs": self.get_songs_by_genre("rock")
            },
            "Hip-Hop Central": {
                "id": "playlist_hiphop_central",
                "name": "Hip-Hop Central",
                "description": "The best in hip-hop and rap",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273f6f6f6f6f6f6f6f6f6f6f6f6",
                "songs": self.get_songs_by_genre("hip-hop")
            },
            "Electronic Vibes": {
                "id": "playlist_electronic_vibes",
                "name": "Electronic Vibes",
                "description": "Electronic and dance music",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273f7f7f7f7f7f7f7f7f7f7f7f7",
                "songs": self.get_songs_by_genre("electronic")
            },
            "Jazz & Blues": {
                "id": "playlist_jazz_blues",
                "name": "Jazz & Blues",
                "description": "Smooth jazz and soulful blues",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273f8f8f8f8f8f8f8f8f8f8f8f8",
                "songs": self.get_songs_by_genre("jazz") + self.get_songs_by_genre("blues")
            },
            "R&B Soul": {
                "id": "playlist_rnb_soul",
                "name": "R&B Soul",
                "description": "Classic R&B and soul music",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273f9f9f9f9f9f9f9f9f9f9f9f9",
                "songs": self.get_songs_by_genre("r&b") + self.get_songs_by_genre("soul")
            },
            "Country Roads": {
                "id": "playlist_country_roads",
                "name": "Country Roads",
                "description": "Classic country favorites",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273fafafafafafafafafafafafafa",
                "songs": self.get_songs_by_genre("country")
            },
            "Classical Masterpieces": {
                "id": "playlist_classical",
                "name": "Classical Masterpieces",
                "description": "Timeless classical compositions",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273fbfbfbfbfbfbfbfbfbfbfbfb",
                "songs": self.get_songs_by_genre("classical")
            },
            "Reggae Vibes": {
                "id": "playlist_reggae",
                "name": "Reggae Vibes",
                "description": "Laid-back reggae classics",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273fcfcfcfcfcfcfcfcfcfcfcfc",
                "songs": self.get_songs_by_genre("reggae")
            },
            "Latin Heat": {
                "id": "playlist_latin",
                "name": "Latin Heat",
                "description": "Hot Latin and reggaeton hits",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273fdfdfdfdfdfdfdfdfdfdfdfd",
                "songs": self.get_songs_by_genre("latin")
            },
            "Alternative Rock": {
                "id": "playlist_alternative",
                "name": "Alternative Rock",
                "description": "Alternative and indie rock favorites",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273fefefefefefefefefefefefefe",
                "songs": self.get_songs_by_genre("alternative rock") + self.get_songs_by_genre("grunge")
            },
            "Workout Mix": {
                "id": "playlist_workout",
                "name": "Workout Mix",
                "description": "High-energy songs for your workout",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273ffffffffffffffffffffffff",
                "songs": self.get_high_energy_songs()
            },
            "Chill Vibes": {
                "id": "playlist_chill",
                "name": "Chill Vibes",
                "description": "Relaxing songs for a laid-back mood",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273000000000000000000000000",
                "songs": self.get_chill_songs()
            },
            "Party Hits": {
                "id": "playlist_party",
                "name": "Party Hits",
                "description": "Dance floor favorites and party anthems",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273111111111111111111111111",
                "songs": self.get_party_songs()
            },
            "Love Songs": {
                "id": "playlist_love",
                "name": "Love Songs",
                "description": "Romantic ballads and love songs",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273222222222222222222222222",
                "songs": self.get_love_songs()
            },
            "Road Trip": {
                "id": "playlist_roadtrip",
                "name": "Road Trip",
                "description": "Perfect songs for your next adventure",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273333333333333333333333333",
                "songs": self.get_roadtrip_songs()
            },
            "Focus & Study": {
                "id": "playlist_focus",
                "name": "Focus & Study",
                "description": "Instrumental and ambient music for concentration",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273444444444444444444444444",
                "songs": self.get_focus_songs()
            },
            "Throwback Hits": {
                "id": "playlist_throwback",
                "name": "Throwback Hits",
                "description": "Classic hits that never get old",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273555555555555555555555555",
                "songs": self.get_throwback_songs()
            },
            "New Releases": {
                "id": "playlist_new_releases",
                "name": "New Releases",
                "description": "The latest and greatest new music",
                "image_url": "https://i.scdn.co/image/ab67616d0000b273666666666666666666666666",
                "songs": self.get_new_releases()
            }
        }
    
    def get_songs_by_genre(self, genre: str) -> List[Dict]:
        """Get songs by specific genre"""
        return [song for song in self.sample_songs 
                if any(genre.lower() in g.lower() for g in song["genres"])]
    
    def get_top_songs_by_popularity(self, limit: int = 50) -> List[Dict]:
        """Get top songs by popularity score"""
        sorted_songs = sorted(self.sample_songs, key=lambda x: x["popularity"], reverse=True)
        return sorted_songs[:limit]
    
    def get_high_energy_songs(self) -> List[Dict]:
        """Get high-energy songs for workouts"""
        high_energy_genres = ["electronic", "dance", "edm", "hip-hop", "rock", "pop"]
        return [song for song in self.sample_songs 
                if any(genre in high_energy_genres for genre in song["genres"]) 
                and song["popularity"] >= 75]
    
    def get_chill_songs(self) -> List[Dict]:
        """Get chill/relaxing songs"""
        chill_genres = ["jazz", "soul", "r&b", "classical", "ballad"]
        chill_songs = []
        for song in self.sample_songs:
            if any(genre in chill_genres for genre in song["genres"]):
                chill_songs.append(song)
            elif "ballad" in song["name"].lower() or any(word in song["name"].lower() 
                for word in ["love", "wonderful", "moon", "dream"]):
                chill_songs.append(song)
        return chill_songs
    
    def get_party_songs(self) -> List[Dict]:
        """Get party/dance songs"""
        party_genres = ["dance", "pop", "funk", "electronic", "disco"]
        party_keywords = ["party", "dance", "funk", "uptown", "crazy", "closer"]
        party_songs = []
        for song in self.sample_songs:
            if any(genre in party_genres for genre in song["genres"]):
                party_songs.append(song)
            elif any(keyword in song["name"].lower() for keyword in party_keywords):
                party_songs.append(song)
        return party_songs
    
    def get_love_songs(self) -> List[Dict]:
        """Get romantic/love songs"""
        love_keywords = ["love", "heart", "someone", "crazy", "wonderful", "fly me"]
        love_genres = ["ballad", "r&b", "soul"]
        love_songs = []
        for song in self.sample_songs:
            if any(keyword in song["name"].lower() for keyword in love_keywords):
                love_songs.append(song)
            elif any(genre in love_genres for genre in song["genres"]):
                love_songs.append(song)
        return love_songs
    
    def get_roadtrip_songs(self) -> List[Dict]:
        """Get good road trip songs"""
        roadtrip_keywords = ["road", "california", "country", "rock", "brightside", "sweet"]
        classic_genres = ["rock", "classic rock", "country", "alternative rock"]
        roadtrip_songs = []
        for song in self.sample_songs:
            if any(keyword in song["name"].lower() for keyword in roadtrip_keywords):
                roadtrip_songs.append(song)
            elif any(genre in classic_genres for genre in song["genres"]) and song["popularity"] >= 75:
                roadtrip_songs.append(song)
        return roadtrip_songs
    
    def get_focus_songs(self) -> List[Dict]:
        """Get focus/study songs"""
        focus_genres = ["classical", "jazz", "instrumental"]
        return [song for song in self.sample_songs 
                if any(genre in focus_genres for genre in song["genres"])]
    
    def get_throwback_songs(self) -> List[Dict]:
        """Get throwback/classic songs"""
        throwback_artists = ["Queen", "Michael Jackson", "Eagles", "Led Zeppelin", 
                            "AC/DC", "Pink Floyd", "Aretha Franklin", "Stevie Wonder",
                            "Bob Marley", "Frank Sinatra", "Louis Armstrong"]
        return [song for song in self.sample_songs 
                if any(artist in song["artist"] for artist in throwback_artists)]
    
    def get_new_releases(self) -> List[Dict]:
        """Get newer releases (simulated based on popularity and modern artists)"""
        modern_artists = ["The Weeknd", "Billie Eilish", "Harry Styles", "Dua Lipa", 
                         "The Chainsmokers", "Drake", "Kendrick Lamar"]
        return [song for song in self.sample_songs 
                if any(artist in song["artist"] for artist in modern_artists)]
    
    def search_songs(self, query: str, limit: int = 20) -> List[Dict]:
        """Search for songs (using sample data for demo)"""
        query_lower = query.lower()
        results = []
        
        for song in self.sample_songs:
            if (query_lower in song["name"].lower() or 
                query_lower in song["artist"].lower() or 
                any(query_lower in genre.lower() for genre in song["genres"])):
                results.append(song)
        
        return results[:limit]
    
    def search_playlists(self, query: str = "") -> List[Dict]:
        """Search playlists"""
        if not query:
            return list(self.playlists.values())
        
        query_lower = query.lower()
        results = []
        for playlist in self.playlists.values():
            if (query_lower in playlist["name"].lower() or 
                query_lower in playlist["description"].lower()):
                results.append(playlist)
        return results
    
    def get_playlist_by_id(self, playlist_id: str) -> Dict:
        """Get playlist by ID"""
        for playlist in self.playlists.values():
            if playlist["id"] == playlist_id:
                return playlist
        return None
    
    def get_all_songs(self) -> List[Dict]:
        """Get all available songs"""
        return self.sample_songs
    
    def get_all_playlists(self) -> List[Dict]:
        """Get all available playlists"""
        return list(self.playlists.values())
    
    # Real Spotify API Methods
    async def get_spotify_access_token(self):
        """Get access token for Spotify API"""
        if not self.client_id or not self.client_secret:
            return None
            
        import time
        current_time = time.time()
        
        # Check if token is still valid
        if self.access_token and current_time < self.token_expires_at:
            return self.access_token
        
        try:
            import base64
            
            # Prepare credentials
            credentials = f"{self.client_id}:{self.client_secret}"
            credentials_b64 = base64.b64encode(credentials.encode()).decode()
            
            # Token request
            headers = {
                'Authorization': f'Basic {credentials_b64}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            data = {'grant_type': 'client_credentials'}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    'https://accounts.spotify.com/api/token',
                    headers=headers,
                    data=data
                )
                
                if response.status_code == 200:
                    token_data = response.json()
                    self.access_token = token_data['access_token']
                    self.token_expires_at = current_time + token_data['expires_in'] - 60  # 60s buffer
                    return self.access_token
                else:
                    logger.error(f"Spotify token request failed: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting Spotify access token: {e}")
            return None
    
    async def search_spotify_tracks(self, query: str, limit: int = 20) -> List[Dict]:
        """Search tracks using real Spotify API"""
        if not self.use_real_spotify:
            return self.search_songs_fallback(query, limit)
        
        try:
            token = await self.get_spotify_access_token()
            if not token:
                return self.search_songs_fallback(query, limit)
            
            headers = {'Authorization': f'Bearer {token}'}
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    'https://api.spotify.com/v1/search',
                    headers=headers,
                    params={
                        'q': query,
                        'type': 'track',
                        'limit': limit,
                        'market': 'US'
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    tracks = data.get('tracks', {}).get('items', [])
                    
                    formatted_tracks = []
                    for track in tracks:
                        formatted_track = self.format_spotify_track(track)
                        formatted_tracks.append(formatted_track)
                    
                    return formatted_tracks
                else:
                    logger.error(f"Spotify search failed: {response.status_code}")
                    return self.search_songs_fallback(query, limit)
                    
        except Exception as e:
            logger.error(f"Error searching Spotify tracks: {e}")
            return self.search_songs_fallback(query, limit)
    
    def format_spotify_track(self, track: Dict) -> Dict:
        """Format Spotify track data to our format"""
        try:
            # Get album images
            images = track.get('album', {}).get('images', [])
            image_url = images[0]['url'] if images else None
            
            # Get genres from artists (may need additional API call for detailed genres)
            genres = []
            artists = track.get('artists', [])
            
            # Basic genre mapping based on track features (simplified)
            track_name = track['name'].lower()
            artist_name = artists[0]['name'].lower() if artists else ''
            
            # Simple genre detection (in production, you'd use Spotify's audio features API)
            if any(word in track_name + ' ' + artist_name for word in ['rock', 'metal', 'guitar']):
                genres.append('rock')
            elif any(word in track_name + ' ' + artist_name for word in ['hip hop', 'rap', 'trap']):
                genres.append('hip-hop')
            elif any(word in track_name + ' ' + artist_name for word in ['electronic', 'edm', 'house', 'techno']):
                genres.append('electronic')
            elif any(word in track_name + ' ' + artist_name for word in ['jazz', 'blues']):
                genres.append('jazz')
            elif any(word in track_name + ' ' + artist_name for word in ['classical', 'symphony', 'orchestra']):
                genres.append('classical')
            else:
                genres.append('pop')  # Default
            
            return {
                'spotify_id': track['id'],
                'name': track['name'],
                'artist': ', '.join([artist['name'] for artist in artists]),
                'album': track.get('album', {}).get('name', ''),
                'genres': genres,
                'duration_ms': track.get('duration_ms', 180000),
                'popularity': track.get('popularity', 50),
                'external_url': track.get('external_urls', {}).get('spotify', ''),
                'preview_url': track.get('preview_url'),
                'image_url': image_url
            }
        except Exception as e:
            logger.error(f"Error formatting Spotify track: {e}")
            return {}
    
    async def get_featured_playlists(self) -> List[Dict]:
        """Get featured playlists from Spotify"""
        if not self.use_real_spotify:
            return list(self.playlists.values())
        
        try:
            token = await self.get_spotify_access_token()
            if not token:
                return list(self.playlists.values())
            
            headers = {'Authorization': f'Bearer {token}'}
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    'https://api.spotify.com/v1/browse/featured-playlists',
                    headers=headers,
                    params={'limit': 20, 'country': 'US'}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    playlists = data.get('playlists', {}).get('items', [])
                    
                    formatted_playlists = []
                    for playlist in playlists:
                        images = playlist.get('images', [])
                        image_url = images[0]['url'] if images else None
                        
                        formatted_playlist = {
                            'id': f"spotify_playlist_{playlist['id']}",
                            'name': playlist['name'],
                            'description': playlist.get('description', ''),
                            'image_url': image_url,
                            'spotify_id': playlist['id'],
                            'songs': []  # Will be populated when playlist is opened
                        }
                        formatted_playlists.append(formatted_playlist)
                    
                    return formatted_playlists
                else:
                    return list(self.playlists.values())
                    
        except Exception as e:
            logger.error(f"Error getting featured playlists: {e}")
            return list(self.playlists.values())
    
    async def get_playlist_tracks(self, playlist_id: str) -> List[Dict]:
        """Get tracks from a Spotify playlist"""
        if not self.use_real_spotify or not playlist_id.startswith('spotify_playlist_'):
            return []
        
        try:
            # Extract actual Spotify playlist ID
            actual_playlist_id = playlist_id.replace('spotify_playlist_', '')
            
            token = await self.get_spotify_access_token()
            if not token:
                return []
            
            headers = {'Authorization': f'Bearer {token}'}
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'https://api.spotify.com/v1/playlists/{actual_playlist_id}/tracks',
                    headers=headers,
                    params={'limit': 50, 'market': 'US'}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    items = data.get('items', [])
                    
                    formatted_tracks = []
                    for item in items:
                        track = item.get('track')
                        if track and track.get('type') == 'track':
                            formatted_track = self.format_spotify_track(track)
                            if formatted_track:
                                formatted_tracks.append(formatted_track)
                    
                    return formatted_tracks
                else:
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting playlist tracks: {e}")
            return []
    
    async def get_top_tracks_by_genre(self, genre: str, limit: int = 20) -> List[Dict]:
        """Get top tracks by genre from Spotify"""
        if not self.use_real_spotify:
            return self.get_songs_by_genre(genre)[:limit]
        
        try:
            # Map our genres to Spotify search terms
            genre_mapping = {
                'bollywood': 'bollywood indian hindi',
                'punjabi': 'punjabi desi',
                'tamil': 'tamil indian',
                'telugu': 'telugu indian',
                'hindi': 'hindi bollywood',
                'rock': 'rock',
                'pop': 'pop',
                'hip-hop': 'hip hop rap',
                'electronic': 'electronic edm',
                'jazz': 'jazz',
                'classical': 'classical',
                'country': 'country',
                'r&b': 'r&b soul',
                'reggae': 'reggae',
                'latin': 'latin reggaeton'
            }
            
            search_term = genre_mapping.get(genre.lower(), genre)
            return await self.search_spotify_tracks(f'genre:{search_term}', limit)
            
        except Exception as e:
            logger.error(f"Error getting top tracks by genre: {e}")
            return self.get_songs_by_genre(genre)[:limit]
    
    def search_songs_fallback(self, query: str, limit: int = 20) -> List[Dict]:
        """Fallback search using sample data"""
        query_lower = query.lower()
        results = []
        
        for song in self.sample_songs:
            if (query_lower in song["name"].lower() or 
                query_lower in song["artist"].lower() or 
                any(query_lower in genre.lower() for genre in song["genres"])):
                results.append(song)
        
        return results[:limit]

spotify_service = SpotifyService()

# API Endpoints
@api_router.get("/")
async def root():
    return {"message": "Music Recommender AI API", "version": "1.0.0"}

# User endpoints
@api_router.post("/users", response_model=User)
async def create_user(user_data: UserCreate):
    user = User(**user_data.dict())
    await db.users.insert_one(user.dict())
    return user

@api_router.get("/users/{user_id}", response_model=User)
async def get_user(user_id: str):
    user = await db.users.find_one({"id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return User(**user)

@api_router.get("/users", response_model=List[User])
async def get_users():
    users = await db.users.find().to_list(100)
    return [User(**user) for user in users]

# Song endpoints
@api_router.post("/songs/search")
async def search_songs(search_data: SongSearch):
    """Search for songs using real Spotify API or fallback"""
    try:
        # Use real Spotify API if available
        if spotify_service.use_real_spotify:
            results = await spotify_service.search_spotify_tracks(search_data.query, search_data.limit)
        else:
            results = spotify_service.search_songs_fallback(search_data.query, search_data.limit)
        
        # Store new songs in database
        for song_data in results:
            if song_data:  # Check if song_data is not empty
                existing_song = await db.songs.find_one({"spotify_id": song_data["spotify_id"]})
                if not existing_song:
                    try:
                        song = Song(**song_data)
                        await db.songs.insert_one(song.dict())
                    except Exception as e:
                        logger.error(f"Error storing song {song_data.get('name', 'Unknown')}: {e}")
        
        return {"songs": results, "total": len(results), "source": "spotify" if spotify_service.use_real_spotify else "mock"}
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@api_router.get("/songs", response_model=List[Song])
async def get_songs():
    """Get all songs from database"""
    songs = await db.songs.find().to_list(100)
    return [Song(**song) for song in songs]

@api_router.get("/songs/{song_id}", response_model=Song)
async def get_song(song_id: str):
    song = await db.songs.find_one({"id": song_id})
    if not song:
        song = await db.songs.find_one({"spotify_id": song_id})
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")
    return Song(**song)

# Rating endpoints
@api_router.post("/ratings", response_model=Rating)
async def create_rating(rating_data: RatingCreate):
    # Check if rating already exists
    existing_rating = await db.ratings.find_one({
        "user_id": rating_data.user_id,
        "song_id": rating_data.song_id
    })
    
    if existing_rating:
        # Update existing rating
        await db.ratings.update_one(
            {"id": existing_rating["id"]},
            {"$set": {"rating": rating_data.rating}}
        )
        existing_rating["rating"] = rating_data.rating
        return Rating(**existing_rating)
    else:
        # Create new rating
        rating = Rating(**rating_data.dict())
        await db.ratings.insert_one(rating.dict())
        return rating

@api_router.get("/ratings/user/{user_id}", response_model=List[Rating])
async def get_user_ratings(user_id: str):
    ratings = await db.ratings.find({"user_id": user_id}).to_list(100)
    return [Rating(**rating) for rating in ratings]

# Favorite endpoints
@api_router.post("/favorites", response_model=Favorite)
async def create_favorite(favorite_data: FavoriteCreate):
    # Check if favorite already exists
    existing_favorite = await db.favorites.find_one({
        "user_id": favorite_data.user_id,
        "song_id": favorite_data.song_id
    })
    
    if existing_favorite:
        return Favorite(**existing_favorite)
    
    favorite = Favorite(**favorite_data.dict())
    await db.favorites.insert_one(favorite.dict())
    return favorite

@api_router.delete("/favorites/{user_id}/{song_id}")
async def remove_favorite(user_id: str, song_id: str):
    result = await db.favorites.delete_one({
        "user_id": user_id,
        "song_id": song_id
    })
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Favorite not found")
    
    return {"message": "Favorite removed successfully"}

@api_router.get("/favorites/user/{user_id}", response_model=List[Favorite])
async def get_user_favorites(user_id: str):
    favorites = await db.favorites.find({"user_id": user_id}).to_list(100)
    return [Favorite(**favorite) for favorite in favorites]

# Recommendation endpoints
@api_router.get("/recommendations/{user_id}")
async def get_recommendations(user_id: str, limit: int = Query(10, ge=1, le=50)):
    """Get AI-powered personalized recommendations"""
    try:
        # Get user preferences
        user = await db.users.find_one({"id": user_id})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get user's ratings and favorites
        ratings = await db.ratings.find({"user_id": user_id}).to_list(100)
        favorites = await db.favorites.find({"user_id": user_id}).to_list(100)
        
        # Get all available songs
        all_songs = spotify_service.get_all_songs()
        
        # Prepare user preferences
        high_rated_songs = [r for r in ratings if r["rating"] >= 4]
        user_preferences = {
            "favorite_genres": user["favorite_genres"],
            "recent_ratings": [f"{r['song_id']} (rating: {r['rating']})" for r in high_rated_songs[:10]],
            "favorite_songs": [f["song_id"] for f in favorites[:10]]
        }
        
        # Generate AI recommendations
        ai_recommendations = await ai_service.generate_recommendations(user_preferences, all_songs)
        
        # Match recommendations with song details
        recommendations = []
        for rec in ai_recommendations[:limit]:
            song_data = next((song for song in all_songs if song["spotify_id"] == rec["song_id"]), None)
            if song_data:
                recommendations.append({
                    "song_id": rec["song_id"],
                    "score": rec["score"],
                    "reason": rec["reason"],
                    "song_details": song_data
                })
        
        # If AI recommendations are insufficient, add popular songs
        if len(recommendations) < limit:
            popular_songs = sorted(all_songs, key=lambda x: x["popularity"], reverse=True)
            for song in popular_songs:
                if len(recommendations) >= limit:
                    break
                if not any(r["song_id"] == song["spotify_id"] for r in recommendations):
                    recommendations.append({
                        "song_id": song["spotify_id"],
                        "score": 0.8,
                        "reason": "Popular song that matches general music trends",
                        "song_details": song
                    })
        
        return {
            "user_id": user_id,
            "recommendations": recommendations[:limit],
            "total": len(recommendations[:limit])
        }
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

# Chat explanation endpoints
@api_router.post("/chat/explain", response_model=ChatResponse)
async def explain_recommendation(message: ChatMessage):
    """Get AI explanation for music recommendations"""
    try:
        response = await ai_service.explain_recommendation(
            message.message,
            message.context or {}
        )
        return response
    except Exception as e:
        logger.error(f"Error explaining recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")

# Playlist endpoints
@api_router.get("/playlists")
async def get_playlists(query: str = Query("", description="Search query for playlists")):
    """Get all playlists or search playlists"""
    try:
        # Get featured playlists from Spotify if available
        if spotify_service.use_real_spotify and not query:
            spotify_playlists = await spotify_service.get_featured_playlists()
            # Combine with our curated playlists
            curated_playlists = list(spotify_service.playlists.values())
            all_playlists = spotify_playlists + curated_playlists
        else:
            if query:
                all_playlists = spotify_service.search_playlists(query)
            else:
                all_playlists = spotify_service.get_all_playlists()
        
        return {
            "playlists": all_playlists, 
            "total": len(all_playlists), 
            "source": "spotify" if spotify_service.use_real_spotify else "mock"
        }
    except Exception as e:
        logger.error(f"Playlist fetch error: {e}")
        raise HTTPException(status_code=500, detail=f"Playlist fetch error: {str(e)}")

@api_router.get("/playlists/{playlist_id}")
async def get_playlist(playlist_id: str):
    """Get specific playlist by ID with tracks"""
    try:
        # Check if it's a Spotify playlist
        if playlist_id.startswith('spotify_playlist_'):
            # Get tracks from Spotify
            tracks = await spotify_service.get_playlist_tracks(playlist_id)
            
            # Get playlist info (you'd typically get this from the playlists endpoint first)
            playlist = {
                "id": playlist_id,
                "name": "Spotify Playlist",
                "description": "Featured playlist from Spotify",
                "image_url": None,
                "songs": tracks
            }
            
            return playlist
        else:
            # Get from our curated playlists
            playlist = spotify_service.get_playlist_by_id(playlist_id)
            if not playlist:
                raise HTTPException(status_code=404, detail="Playlist not found")
            
            return playlist
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Playlist fetch error: {e}")
        raise HTTPException(status_code=500, detail=f"Playlist fetch error: {str(e)}")

@api_router.get("/playlists/genre/{genre}")
async def get_playlists_by_genre(genre: str):
    """Get playlists that match a specific genre"""
    try:
        # If using Spotify, search for genre-specific tracks and create dynamic playlist
        if spotify_service.use_real_spotify:
            tracks = await spotify_service.get_top_tracks_by_genre(genre, 30)
            
            genre_playlist = {
                "id": f"dynamic_genre_{genre}",
                "name": f"{genre.title()} Hits",
                "description": f"Top {genre} tracks from Spotify",
                "image_url": None,
                "songs": tracks
            }
            
            return {"playlists": [genre_playlist], "total": 1, "genre": genre, "source": "spotify"}
        else:
            # Fallback to curated playlists
            all_playlists = spotify_service.get_all_playlists()
            genre_playlists = []
            
            for playlist in all_playlists:
                if (genre.lower() in playlist["name"].lower() or 
                    genre.lower() in playlist["description"].lower()):
                    genre_playlists.append(playlist)
            
            return {"playlists": genre_playlists, "total": len(genre_playlists), "genre": genre, "source": "mock"}
    except Exception as e:
        logger.error(f"Genre playlist fetch error: {e}")
        raise HTTPException(status_code=500, detail=f"Genre playlist fetch error: {str(e)}")

# Spotify Configuration Endpoints
@api_router.post("/spotify/configure")
async def configure_spotify(credentials: dict):
    """Configure Spotify API credentials"""
    try:
        client_id = credentials.get('client_id', '').strip()
        client_secret = credentials.get('client_secret', '').strip()
        
        if not client_id or not client_secret:
            raise HTTPException(status_code=400, detail="Both client_id and client_secret are required")
        
        # Update environment variables
        os.environ['SPOTIFY_CLIENT_ID'] = client_id
        os.environ['SPOTIFY_CLIENT_SECRET'] = client_secret
        
        # Update .env file
        env_path = Path(__file__).parent / '.env'
        env_content = env_path.read_text()
        
        # Update or add Spotify credentials
        lines = env_content.split('\n')
        updated_lines = []
        found_client_id = False
        found_client_secret = False
        
        for line in lines:
            if line.startswith('SPOTIFY_CLIENT_ID='):
                updated_lines.append(f'SPOTIFY_CLIENT_ID="{client_id}"')
                found_client_id = True
            elif line.startswith('SPOTIFY_CLIENT_SECRET='):
                updated_lines.append(f'SPOTIFY_CLIENT_SECRET="{client_secret}"')
                found_client_secret = True
            else:
                updated_lines.append(line)
        
        # Add new credentials if not found
        if not found_client_id:
            updated_lines.append(f'SPOTIFY_CLIENT_ID="{client_id}"')
        if not found_client_secret:
            updated_lines.append(f'SPOTIFY_CLIENT_SECRET="{client_secret}"')
        
        # Write back to .env file
        env_path.write_text('\n'.join(updated_lines))
        
        # Reinitialize Spotify service
        global spotify_service
        spotify_service = SpotifyService()
        
        # Test the connection
        test_token = await spotify_service.get_spotify_access_token()
        
        if test_token:
            return {
                "message": "Spotify credentials configured successfully",
                "spotify_enabled": True,
                "status": "connected"
            }
        else:
            return {
                "message": "Spotify credentials saved but connection failed",
                "spotify_enabled": False,
                "status": "error",
                "error": "Failed to authenticate with Spotify API"
            }
            
    except Exception as e:
        logger.error(f"Spotify configuration error: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")

@api_router.get("/spotify/status")
async def get_spotify_status():
    """Check Spotify API connection status"""
    try:
        client_id = os.environ.get('SPOTIFY_CLIENT_ID', '').strip()
        client_secret = os.environ.get('SPOTIFY_CLIENT_SECRET', '').strip()
        
        if not client_id or not client_secret:
            return {
                "spotify_enabled": False,
                "status": "no_credentials",
                "message": "Spotify credentials not configured"
            }
        
        # Test connection
        token = await spotify_service.get_spotify_access_token()
        
        if token:
            return {
                "spotify_enabled": True,
                "status": "connected",
                "message": "Spotify API connected successfully",
                "client_id_preview": f"{client_id[:8]}..." if len(client_id) > 8 else client_id
            }
        else:
            return {
                "spotify_enabled": False,
                "status": "connection_failed",
                "message": "Failed to connect to Spotify API"
            }
            
    except Exception as e:
        logger.error(f"Spotify status check error: {e}")
        return {
            "spotify_enabled": False,
            "status": "error",
            "message": f"Error checking Spotify status: {str(e)}"
        }

# Audio proxy endpoint to handle CORS issues with iTunes
@api_router.get("/audio/proxy")
async def proxy_audio(url: str):
    """
    Proxy audio URLs to avoid CORS issues
    """
    try:
        import httpx
        
        # Fetch the audio file
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            
            if response.status_code == 200:
                # Return the audio with proper headers
                from fastapi.responses import StreamingResponse
                import io
                
                # Get the original content-type from iTunes response
                content_type = response.headers.get("content-type", "audio/mp4")
                
                return StreamingResponse(
                    io.BytesIO(response.content),
                    media_type=content_type,
                    headers={
                        "Accept-Ranges": "bytes",
                        "Cache-Control": "public, max-age=3600",
                        "Access-Control-Allow-Origin": "*"
                    }
                )
            else:
                raise HTTPException(status_code=response.status_code, detail="Failed to fetch audio")
                
    except Exception as e:
        logger.error(f"Audio proxy error: {e}")
        raise HTTPException(status_code=500, detail=f"Audio proxy failed: {str(e)}")

# Enhanced music discovery endpoints
@api_router.get("/discover/trending")
async def get_trending_music():
    """Get trending music from Spotify"""
    try:
        if spotify_service.use_real_spotify:
            # Get featured playlists and extract some trending tracks
            playlists = await spotify_service.get_featured_playlists()
            trending_tracks = []
            
            # Get tracks from first few playlists
            for playlist in playlists[:3]:
                tracks = await spotify_service.get_playlist_tracks(playlist['id'])
                trending_tracks.extend(tracks[:5])  # Get 5 tracks from each playlist
            
            return {
                "tracks": trending_tracks[:20],  # Limit to 20 tracks
                "total": len(trending_tracks[:20]),
                "source": "spotify"
            }
        else:
            # Fallback to popular tracks from sample data
            popular_tracks = spotify_service.get_top_songs_by_popularity(20)
            return {
                "tracks": popular_tracks,
                "total": len(popular_tracks),
                "source": "mock"
            }
            
    except Exception as e:
        logger.error(f"Trending music error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting trending music: {str(e)}")

@api_router.get("/discover/new-releases")
async def get_new_releases():
    """Get new releases from Spotify"""
    try:
        if spotify_service.use_real_spotify:
            token = await spotify_service.get_spotify_access_token()
            if not token:
                raise HTTPException(status_code=500, detail="Spotify authentication failed")
            
            headers = {'Authorization': f'Bearer {token}'}
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    'https://api.spotify.com/v1/browse/new-releases',
                    headers=headers,
                    params={'limit': 20, 'country': 'US'}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    albums = data.get('albums', {}).get('items', [])
                    
                    new_releases = []
                    for album in albums:
                        # Get tracks from each album
                        album_response = await client.get(
                            f"https://api.spotify.com/v1/albums/{album['id']}/tracks",
                            headers=headers,
                            params={'limit': 3}  # Get first 3 tracks from each album
                        )
                        
                        if album_response.status_code == 200:
                            tracks_data = album_response.json()
                            tracks = tracks_data.get('items', [])
                            
                            for track in tracks:
                                # Create full track object
                                full_track = {
                                    **track,
                                    'album': album,
                                    'popularity': 50  # Default for new releases
                                }
                                formatted_track = spotify_service.format_spotify_track(full_track)
                                if formatted_track:
                                    new_releases.append(formatted_track)
                    
                    return {
                        "tracks": new_releases[:20],
                        "total": len(new_releases[:20]),
                        "source": "spotify"
                    }
                else:
                    raise HTTPException(status_code=500, detail="Failed to fetch new releases from Spotify")
        else:
            # Fallback to newer tracks from sample data
            new_tracks = spotify_service.get_new_releases()
            return {
                "tracks": new_tracks,
                "total": len(new_tracks),
                "source": "mock"
            }
            
    except Exception as e:
        logger.error(f"New releases error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting new releases: {str(e)}")

# Initialize sample data
@api_router.post("/init-data")
async def initialize_sample_data():
    """Initialize database with sample songs"""
    try:
        # Clear existing data
        await db.songs.delete_many({})
        
        # Add sample songs
        for song_data in spotify_service.get_all_songs():
            song = Song(**song_data)
            await db.songs.insert_one(song.dict())
        
        return {"message": "Sample data initialized successfully", "songs_added": len(spotify_service.get_all_songs())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization error: {str(e)}")

# Health check
@api_router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "database": "connected",
            "ai_service": "active",
            "spotify_service": "active"
        }
    }

# Include router
app.include_router(api_router)

# Shutdown event
@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
