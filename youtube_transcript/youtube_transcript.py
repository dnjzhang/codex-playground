#!/usr/bin/env python3
import tiktoken
from dotenv import load_dotenv
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser
from langchain_community.document_loaders import YoutubeAudioLoader
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
import os
import argparse
import sys
from urllib.parse import parse_qs, urlparse

_ = load_dotenv()

def count_tokens(text):
    """Count tokens in the text using tiktoken."""
    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
    return len(encoding.encode(text))


def extract_video_id(url):
    """Extract a YouTube video ID from common URL formats."""
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()

    if hostname in ("youtu.be", "www.youtu.be"):
        return parsed.path.lstrip("/") or None

    if "youtube.com" in hostname or "youtube-nocookie.com" in hostname:
        if parsed.path == "/watch":
            query = parse_qs(parsed.query)
            return query.get("v", [None])[0]
        if parsed.path.startswith(("/embed/", "/shorts/")):
            parts = parsed.path.split("/")
            return parts[2] if len(parts) > 2 else None

    return None


def fetch_captions_transcript(url):
    """Fetch transcript from YouTube captions when audio transcription fails."""
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Unable to extract video ID for captions fallback.")

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    except (TranscriptsDisabled, VideoUnavailable) as exc:
        raise RuntimeError("Captions are not available for this video.") from exc

    try:
        transcript = transcript_list.find_manually_created_transcript(["en"])
    except NoTranscriptFound:
        try:
            transcript = transcript_list.find_generated_transcript(["en"])
        except NoTranscriptFound:
            transcript = next(iter(transcript_list))

    entries = transcript.fetch()
    return "\n".join(entry["text"] for entry in entries)


def process_youtube_video(url, base_dir, filename=None):
    """
    Process a YouTube video to create a transcript and count tokens.

    Args:
        url (str): YouTube video URL
        save_dir (str): Directory to save the transcript and audio files
        filename (str, optional): Custom filename for the transcript
    """
    save_dir = base_dir + filename
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    # Delete all existing files in the directory
    for file in os.listdir(save_dir):
        file_path = os.path.join(save_dir, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error: {e}")

    loader = GenericLoader(
        YoutubeAudioLoader([url], save_dir),
        OpenAIWhisperParser()
    )

    transcript = ""
    used_captions = False
    try:
        docs = loader.load()
        print(f"Found {len(docs)} transcript segments")
        transcript = '\n'.join(doc.page_content for doc in docs)
    except Exception as exc:
        print(f"Audio transcription failed ({exc}). Falling back to captions.", file=sys.stderr)
        transcript = fetch_captions_transcript(url)
        used_captions = True

    if not transcript:
        transcript = fetch_captions_transcript(url)
        used_captions = True

    if used_captions:
        print("Transcript pulled from YouTube captions.")

    # Determine output filename
    if filename:
        # Add .txt extension if not present
        if not filename.endswith('.txt'):
            filename += '.txt'
        output_file = os.path.join(save_dir, filename)
    else:
        # Use default naming convention
        video_id = url.split('watch?v=')[-1]
        output_file = os.path.join(save_dir, f'transcript_{video_id}.txt')

    # Write transcript to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(transcript)

    # Count tokens
    token_count = count_tokens(transcript)

    print(f"Transcript has been saved to: {output_file}")
    print(f"Total tokens used: {token_count}")
    return output_file, token_count


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Create transcript from YouTube video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Basic usage with required arguments
    %(prog)s "https://www.youtube.com/watch?v=VIDEO_ID" "transcript_name"

    # Specify a custom base directory
    %(prog)s "https://www.youtube.com/watch?v=VIDEO_ID" "transcript_name" --base_dir "./my_transcripts"
        '''
    )
    parser.add_argument(
        'url',
        type=str,
        help='YouTube video URL'
    )
    parser.add_argument(
        'filename',
        type=str,
        help='Custom filename for the transcript'
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default="transcripts/",
        help='Directory to save the transcript (default: %(default)s)'
    )

    args = parser.parse_args()
    try:
        output_file, token_count = process_youtube_video(
            args.url,
            args.base_dir,
            args.filename
        )
        print("\nProcess completed successfully!")
        return 0
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
