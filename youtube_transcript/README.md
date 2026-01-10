# YouTube Transcript Utility

Create a text transcript from a YouTube video and count tokens.

## Requirements
- System: `ffmpeg` installed and on PATH.
- Python: `pip install langchain-community yt-dlp openai-whisper tiktoken python-dotenv youtube-transcript-api`

## Run
- `python youtube_transcript/youtube_transcript.py "https://www.youtube.com/watch?v=VIDEO_ID" "my_talk" --base_dir transcripts/`
- `./transcript-youtube.sh "https://www.youtube.com/watch?v=VIDEO_ID" "my_talk" --base_dir transcripts/`

## Expected Output
- Prints segment count, saves a `.txt` transcript under `transcripts/my_talk/`, and prints total token count.
- Example lines:
  - `Found 42 transcript segments`
  - `Transcript has been saved to: transcripts/my_talk/my_talk.txt`
  - `Total tokens used: 12345`
  - If audio transcription fails and captions exist, it will fall back to YouTube captions.
