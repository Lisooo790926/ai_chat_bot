"""
This script provides utilities for extracting transcripts from YouTube videos.
It uses the YouTube Transcript API to extract subtitles from videos and
provides helper functions for parsing YouTube URLs and retrieving video lists from channels.
"""

from urllib.parse import parse_qs
from urllib.parse import urlparse
from yt_dlp import YoutubeDL
import threading
import functools
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import Playlist
import logging
from utils.logger import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from components.embedding import VectorStoreManager, get_embedding_model
from langchain.schema import Document

DEFAULT_LANGUAGES = ["zh-TW", "zh-Hant", "zh", "zh-Hans", "ja", "en", "ko"]

ALLOWED_NETLOCS = {
    "youtu.be",
    "m.youtube.com",
    "youtube.com",
    "www.youtube.com",
    "www.youtube-nocookie.com",
    "vid.plus",
}
class YoutubeLoader:
    def __init__(self):
        self.processed_content = []  # Store all processed content
        
    def process_video(self, video_url, language="en"):
        try:
            logger.info(f"Fetching transcript for: {video_url}")
            video_id = parse_video_id(video_url)

            transcript_pieces: list[dict[str, str | float]] = (
                YouTubeTranscriptApi().get_transcript(video_id, [language])
            )

            lines = []
            for transcript_piece in transcript_pieces:
                text = str(transcript_piece.get("text", "")).strip()
                if text:
                    lines.append(text)
            transcript_text = "\n".join(lines)
            
            if not transcript_text:
                logger.warning(f"No transcript available for: {video_url}")
                return False, None, None
                
            logger.info(f"Successfully fetched transcript for: {video_url}")
            
            # Instead of directly saving, append to processed_content
            self.processed_content.append({
                'content': transcript_text,
                'file_path': None,
                'video_id': video_id,
                'dataset_name': None
            })
            
            return True, transcript_text, None
            
        except Exception as e:
            logger.error(f"Error processing video {video_url}: {str(e)}")
            return False, str(e), None
    
    def batch_embed_content(self, provider="gemini"):
        """Batch embed all processed content"""
        if not self.processed_content:
            logger.warning("No content to embed")
            return False, "No content to process"
            
        try:
            # Combine all content into one large document
            combined_docs = []
            for item in self.processed_content:
                doc = Document(
                    page_content=item['content'],
                    metadata={
                        "dataset": item['dataset_name'],
                        "video_id": item['video_id'],
                        "file": item['file_path']
                    }
                )
                combined_docs.append(doc)
            
            # Process all content in one batch
            success = self._embed_batch(combined_docs, provider)
            
            # Clear processed content after successful embedding
            if success:
                self.processed_content = []
                
            return success, "Batch processing completed"
            
        except Exception as e:
            logger.error(f"Error in batch embedding: {str(e)}")
            return False, str(e)
    
    def _embed_batch(self, documents, provider):
        """Internal method to handle batch embedding"""
        try:
            vector_manager = VectorStoreManager()
            
            # Get embedding model
            embedding_llm, _ = get_embedding_model(provider)
            collection_name = vector_manager.get_collection_name("bootcamp", provider)
            
            # Split documents into chunks
            markdown_splitter = RecursiveCharacterTextSplitter(
                separators=["#", "##", "###", "\n\n", "\n", " "],
                chunk_size=1000,
                chunk_overlap=100,
            )
            
            all_splits = []
            for doc in documents:
                splits = markdown_splitter.split_documents([doc])
                all_splits.extend(splits)
            
            # Batch embed all splits
            qdrant = QdrantClient()
            import uuid
            point_id = str(uuid.uuid4())
            qdrant.upsert(
                collection_name=collection_name,
                points=[
                    {
                        "id": point_id,
                        "vector": embedding_llm.embed_query(doc.page_content),
                        "payload": doc.metadata
                    }
                    for doc in all_splits
                ]
            )
            
            return True
            
        except Exception as e:
            if "429" in str(e) and provider == "gemini":
                logger.warning("Rate limit reached for Gemini, retrying with Azure")
                return self._embed_batch(documents, "azure")
            raise


def parse_video_id(url: str) -> str:
    """Parse a YouTube URL and return the video ID if valid, otherwise None."""
    parsed_url = urlparse(url)

    if parsed_url.scheme not in {"http", "https"}:
        raise f"unsupported URL scheme: {parsed_url.scheme}"

    if parsed_url.netloc not in ALLOWED_NETLOCS:
        raise f"unsupported URL netloc: {parsed_url.netloc}"

    path = parsed_url.path

    if path.endswith("/watch"):
        query = parsed_url.query
        parsed_query = parse_qs(query)
        if "v" in parsed_query:
            ids = parsed_query["v"]
            video_id = ids if isinstance(ids, str) else ids[0]
        else:
            raise f"no video found in URL: {url}"
    else:
        path = parsed_url.path.lstrip("/")
        video_id = path.split("/")[-1]

    if len(video_id) != 11:  # Video IDs are 11 characters long
        raise f"invalid video ID: {video_id}"

    return video_id


def get_youtube_videos(channel_url):
    """
    get youtube videos from channel

    Args:
        channel_url (str): channel url

    Returns:
        list: list of videos
    """
    ydl_opts = {
        "extract_flat": True,
        "quiet": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)

    video_list = [
        {"title": entry["title"], "url": entry["url"]}
        for entry in info.get("entries", [])
    ]
    return video_list


def get_playlist_videos(playlist_url: str) -> list[dict]:
    """
    Get videos from a YouTube playlist
    
    Args:
        playlist_url: URL of the YouTube playlist
        
    Returns:
        list: List of video dictionaries with url and title
    """
    try:
        playlist = Playlist(playlist_url)
        return playlist.video_urls
    except Exception as e:
        logging.error(f"Error getting playlist: {str(e)}")
        return []
