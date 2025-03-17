"""
This script provides utilities for extracting transcripts from YouTube videos.
It uses the YouTube Transcript API to extract subtitles from videos and
provides helper functions for parsing YouTube URLs and retrieving video lists from channels.
"""

from urllib.parse import parse_qs
from urllib.parse import urlparse
from yt_dlp import YoutubeDL
import threading
import time
import functools
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import find_dotenv, load_dotenv


DEFAULT_LANGUAGES = ["zh-TW", "zh-Hant", "zh", "zh-Hans", "ja", "en", "ko"]


ALLOWED_NETLOCS = {
    "youtu.be",
    "m.youtube.com",
    "youtube.com",
    "www.youtube.com",
    "www.youtube-nocookie.com",
    "vid.plus",
}


# Replace timeout_decorator with a threading-based implementation
def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutError("Function call timed out")]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            if isinstance(result[0], Exception):
                raise result[0]
            
            return result[0]
        return wrapper
    return decorator


class YoutubeLoader:
    def __init__(self, languages: list[str] | None = None) -> None:
        self.languages = languages or DEFAULT_LANGUAGES

    @timeout(20)  # Using our custom timeout instead of timeout_decorator
    def load(self, url: str) -> str:
        """
        Load youtube video subtitle
        Args:
            url (str): youtube video url
        Returns:
            str: video subtitle
        """
        video_id = parse_video_id(url)

        transcript_pieces: list[dict[str, str | float]] = (
            YouTubeTranscriptApi().get_transcript(video_id, self.languages)
        )

        lines = []
        for transcript_piece in transcript_pieces:
            text = str(transcript_piece.get("text", "")).strip()
            if text:
                lines.append(text)
        return "\n".join(lines)


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
