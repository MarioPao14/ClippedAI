"""Main script to process videos and create YouTube Shorts using AI-powered transcription and clip detection."""

import os
import pickle
import shutil
import subprocess
import sys
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
try:
    from omegaconf.nodes import AnyNode
    torch.serialization.add_safe_globals([AnyNode])
except Exception:
    pass

import nltk
from clipsai import Transcriber, ClipFinder, resize, MediaEditor, AudioVideoFile
from clipsai.clip.clip import Clip
from dotenv import load_dotenv

# Suppress unnecessary warnings
import warnings
warnings.filterwarnings("ignore", message="Model was trained with pyannote.audio")
warnings.filterwarnings("ignore", message="Model was trained with torch")
warnings.filterwarnings("ignore", message="Lightning automatically upgraded")
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", message="torchaudio._backend.list_audio_backends has been deprecated")

# Suppress unnecessary warnings via environment variables
os.environ['FFREPORT'] = 'file=ffmpeg.log:level=32'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Load environment variables
load_dotenv()

# Download required NLTK data
nltk.download('punkt')

# --- Directories ---
INPUT_DIR = "input"
OUTPUT_DIR = "output"
FONT_DIR = "fonts" 

# --- Hugging Face ---
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "your_huggingface_token_here")

# --- Clip Duration ---
MIN_CLIP_DURATION = int(os.getenv("MIN_CLIP_DURATION", "45"))
MAX_CLIP_DURATION = int(os.getenv("MAX_CLIP_DURATION", "120"))

# --- Transcription ---
TRANSCRIPTION_MODEL = os.getenv("TRANSCRIPTION_MODEL", "large-v1")

# --- Resizing ---
ASPECT_RATIO_WIDTH = int(os.getenv("ASPECT_RATIO_WIDTH", "9"))
ASPECT_RATIO_HEIGHT = int(os.getenv("ASPECT_RATIO_HEIGHT", "16"))

# --- Groq API ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")

# --- Subtitles ---
SUBTITLE_FONT = "Montserrat Extra Bold"
SUBTITLE_FONT_SIZE = 80
SUBTITLE_ALIGNMENT = 8  # Top center
SUBTITLE_MARGIN_V = 120
SUBTITLE_STYLES = {
    "Default": {
        "Fontname": SUBTITLE_FONT,
        "Fontsize": SUBTITLE_FONT_SIZE,
        "PrimaryColour": "&H00FFFFFF",  # White
        "SecondaryColour": "&H000000FF",
        "OutlineColour": "&H40000000",
        "BackColour": "&HFF000000",
        "Bold": -1,
        "Italic": 0,
        "Underline": 0,
        "StrikeOut": 0,
        "ScaleX": 100,
        "ScaleY": 100,
        "Spacing": 2,
        "Angle": 0,
        "BorderStyle": 1,
        "Outline": 15,
        "Shadow": 0,
        "Alignment": SUBTITLE_ALIGNMENT,
        "MarginL": 30,
        "MarginR": 30,
        "MarginV": SUBTITLE_MARGIN_V,
        "Encoding": 1,
    },
    "Yellow": {
        "Fontname": SUBTITLE_FONT,
        "Fontsize": SUBTITLE_FONT_SIZE,
        "PrimaryColour": "&H0000FFFF",  # Yellow
        "SecondaryColour": "&H000000FF",
        "OutlineColour": "&H40000000",
        "BackColour": "&HFF000000",
        "Bold": -1,
        "Italic": 0,
        "Underline": 0,
        "StrikeOut": 0,
        "ScaleX": 100,
        "ScaleY": 100,
        "Spacing": 2,
        "Angle": 0,
        "BorderStyle": 1,
        "Outline": 15,
        "Shadow": 0,
        "Alignment": SUBTITLE_ALIGNMENT,
        "MarginL": 30,
        "MarginR": 30,
        "MarginV": SUBTITLE_MARGIN_V,
        "Encoding": 1,
    },
}

def get_transcription_file_path(input_path: str) -> str:
    """Generate the transcription file path based on input video path."""
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(INPUT_DIR, f"{base_name}_transcription.pkl")

def load_existing_transcription(transcription_path: str):
    """Load existing transcription if it exists."""
    if os.path.exists(transcription_path):
        print(f"Found existing transcription: {transcription_path}")
        try:
            with open(transcription_path, "rb") as f:
                transcription = pickle.load(f)
            print("Successfully loaded existing transcription!")
            return transcription
        except Exception as e:
            print(f"Error loading existing transcription: {e}")
            return None
    return None

def save_transcription(transcription, transcription_path: str):
    """Save transcription to file."""
    try:
        with open(transcription_path, "wb") as f:
            pickle.dump(transcription, f)
        print(f"Transcription saved to: {transcription_path}")
    except Exception as e:
        print(f"Error saving transcription: {e}")

def ass_time(seconds: float) -> str:
    """Convert seconds to ASS time format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

def safe_filename(s: str) -> str:
    """Remove characters not allowed in filenames, but keep spaces, punctuation, and emojis."""
    # Define a set of characters that are generally safe for filenames
    # This includes alphanumeric, spaces, and some common punctuation
    safe_chars = string.ascii_letters + string.digits + " -_."
    # Add common punctuation that might be part of a title but needs careful handling
    safe_chars += "!?,:;@#$%^&+=[]{}"
    # Add a range of common emojis. This list can be expanded if needed.
    # Using a broad range to cover most common emojis without making the string excessively long.
    emoji_chars = "".join(chr(i) for i in range(0x1F600, 0x1F64F)) + \
                  "".join(chr(i) for i in range(0x1F300, 0x1F5FF)) + \
                  "".join(chr(i) for i in range(0x1F900, 0x1F9FF)) + \
                  "".join(chr(i) for i in range(0x1FA70, 0x1FAFF))
    
    valid_chars = safe_chars + emoji_chars
    return ''.join(c for c in s if c in valid_chars)


def get_font_path(font_name: str) -> str:
    """Get the path to a font file in the font directory."""
    # Try different extensions for the font file
    for ext in ['.ttf', '.otf', '.TTF', '.OTF']:
        font_path = os.path.join("fonts", f"{font_name}{ext}")
        if os.path.exists(font_path):
            return font_path
    # If not found with extension, return as is (for system fonts)
    return os.path.join("fonts", font_name)

def transcribe_with_progress(audio_file_path, transcriber):
    """Transcribe with progress tracking"""
    print('Transcribing video...')
    
    # Get video duration for progress calculation
    try:
        probe_cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', audio_file_path]
        duration = float(subprocess.check_output(probe_cmd).decode().strip())
        print(f"Video duration: {duration:.2f} seconds")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        duration = 0
        print(f"Could not determine video duration for progress tracking: {e}")
    
    # Custom progress callback
    def progress_callback(current_time):
        if duration > 0:
            progress = (current_time / duration) * 100
            print(f"Transcription progress: {progress:.1f}% ({current_time:.1f}s / {duration:.1f}s)") 
        else:
            print(f"Transcription progress: {current_time:.1f}s processed")
    
    # For now, we'll use a simple approach since clipsai doesn't expose progress directly
    # You can enhance this by modifying the clipsai library or using a different approach
    print("Starting transcription (progress updates may be limited)...")
    transcription = transcriber.transcribe(audio_file_path=audio_file_path, iso6391_lang_code='en')
    print("Transcription completed!")
    return transcription

def create_animated_subtitles(video_path, transcription, clip, output_path):
    """
    Create clean, bold subtitles matching the provided style: white bold for text, yellow bold for numbers/currency, no effects, TOP CENTER.
    """
    print('Creating styled subtitles...')
    
    # Get word info for the clip
    word_info = [w for w in transcription.get_word_info() if w["start_time"] >= clip.start_time and w["end_time"] <= clip.end_time]
    if not word_info:
        print('No word-level transcript found for the clip. Skipping subtitles.')
        return video_path
    
    # Build cues: group words into phrases of max 25 chars
    cues = []
    current_cue = {
        'words': [],
        'start_time': None,
        'end_time': None
    }
    
    for w in word_info:
        word = w["word"]
        start_time = w["start_time"] - clip.start_time
        end_time = w["end_time"] - clip.start_time  # Fixed: should be clip.start_time, not clip.end_time
        
        should_start_new = False
        if current_cue['start_time'] is None:
            should_start_new = True
        elif len(' '.join(current_cue['words']) + ' ' + word) > 25:
            should_start_new = True
        elif start_time - current_cue['end_time'] > 0.5:
            should_start_new = True
        
        if should_start_new:
            if current_cue['words']:
                cues.append({
                    'start': current_cue['start_time'],
                    'end': current_cue['end_time'],
                    'text': ' '.join(current_cue['words'])
                })
            current_cue = {
                'words': [word],
                'start_time': start_time,
                'end_time': end_time
            }
        else:
            current_cue['words'].append(word)
            current_cue['end_time'] = end_time
    if current_cue['words']:
        cues.append({
            'start': current_cue['start_time'],
            'end': current_cue['end_time'],
            'text': ' '.join(current_cue['words'])
        })
    
    # Determine font used and print to console
    font_used = "Montserrat Extra Bold"
    print(f"Subtitles will use font: {font_used}")
    print("NOTE: Ensure 'Montserrat Extra Bold' font is installed or is available in the fonts directory.")

    # Write ASS subtitle file with clean, bold styling at the TOP CENTER
    ass_file = os.path.join(OUTPUT_DIR, 'temp_subtitles.ass')
    with open(ass_file, 'w', encoding='utf-8') as f:
        f.write("""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 1
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Montserrat Extra Bold,80,&H00FFFFFF,&H000000FF,&H40000000,&HFF000000,-1,0,0,0,100,100,2,0,1,15,0,8,30,30,120,1
Style: Yellow,Montserrat Extra Bold,80,&H0000FFFF,&H000000FF,&H40000000,&HFF000000,-1,0,0,0,100,100,2,0,1,15,0,8,30,30,120,1
Style: Fallback,Arial Rounded MT Bold,80,&H00FFFFFF,&H000000FF,&H40000000,&HFF000000,-1,0,0,0,100,100,2,0,1,15,0,8,30,30,120,1
Style: FallbackYellow,Arial Rounded MT Bold,80,&H0000FFFF,&H000000FF,&H40000000,&HFF000000,-1,0,0,0,100,100,2,0,1,15,0,8,30,30,120,1
Style: Fallback2,Arial Black,80,&H00FFFFFF,&H000000FF,&H40000000,&HFF000000,-1,0,0,0,100,100,2,0,1,15,0,8,30,30,120,1
Style: Fallback2Yellow,Arial Black,80,&H0000FFFF,&H000000FF,&H40000000,&HFF000000,-1,0,0,0,100,100,2,0,1,15,0,8,30,30,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""")
        for cue in cues:
            start = ass_time(cue['start'])
            end = ass_time(cue['end'])
            words = cue['text'].split()
            line = ''
            for w in words:
                if any(char.isdigit() for char in w) or '$' in w or (',' in w and w.replace(',', '').isdigit()):
                    line += f'{{\\style Yellow}}{w} '
                else:
                    line += f'{w} '
            line = line.strip()
            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{line}\n")
    
    final_output = output_path.replace('.mp4', '_with_subtitles.mp4')
    ffmpeg_cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f"ass={ass_file}",
        '-c:a', 'copy',
        '-y',
        final_output
    ]
    try:
        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        os.remove(ass_file)
        print(f'Styled subtitles added successfully!')
        return final_output
    except subprocess.CalledProcessError as e:
        print(f'Error adding subtitles: {e}')
        print(f'FFmpeg stderr: {e.stderr.decode()}')
        print(f'FFmpeg stdout: {e.stdout.decode()}')
        return video_path

def get_viral_title(transcript_text, groq_api_key):
    import requests
    # Limit examples to avoid too long prompt
    examples = [
        "She was almost dead 😵", "He made $1,000,000 in 1 hour 💸", "This changed everything... 😲", 
        "They couldn't believe what happened! 😱", "He risked it all for this 😬"
    ]
    prompt = (
        "Given the following transcript, generate a catchy, viral YouTube Shorts title (max 7 words). "
        "ALWAYS include an emoji in the title. ONLY output the title, nothing else. Do NOT use hashtags. "
        "Do NOT explain, do NOT repeat the prompt, do NOT add quotes. The title should be in the style of these examples: "
        + ", ".join(examples) + ".\n\nTranscript:\n" + transcript_text
    )
    headers = {
        'Authorization': f'Bearer {groq_api_key}',
        'Content-Type': 'application/json',
    }
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 30,
        "temperature": 0.7,
        "top_p": 0.9
    }
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        result = response.json()
        # Just return the first line of the response as the title, and filter out any lines that look like explanations or quotes
        content = result['choices'][0]['message']['content']
        lines = [l.strip('"') for l in content.strip().split('\n') if l.strip() and not l.lower().startswith('here') and not l.lower().startswith('title:')]
        title = lines[0] if lines else "Untitled Clip"
        return title
    except requests.exceptions.HTTPError as e:
        print(f"Error with Groq API: {e}")
        print(f"Response status code: {response.status_code}")
        print(f"Response text: {response.text}")
        return "Untitled Clip"
    except Exception as e:
        print(f"Unexpected error with Groq API: {e}")
        return "Untitled Clip"

def calculate_engagement_score(clip, transcription):
    """
    Calculate a custom engagement score for a clip based on available data.
    Higher scores indicate more engaging content.
    """
    # Get words in the clip
    clip_words = [w for w in transcription.get_word_info() 
                  if w["start_time"] >= clip.start_time and w["end_time"] <= clip.end_time]
    
    if not clip_words:
        return 0.0
    
    # Calculate various engagement factors
    duration = clip.end_time - clip.start_time
    word_count = len(clip_words)
    word_density = word_count / duration if duration > 0 else 0
    
    # Count numbers, currency, and exclamation marks (engagement indicators)
    engagement_words = 0
    for word_info in clip_words:
        word = word_info["word"]
        if any(char.isdigit() for char in word) or '$' in word or '!' in word:
            engagement_words += 1
    
    # Calculate engagement score (0-1 scale)
    # Factors: word density (45%), engagement words ratio (30%), duration balance (25%)
    word_density_score = min(word_density / 3.0, 1.0)  # Normalize to 0-1
    engagement_ratio = engagement_words / word_count if word_count > 0 else 0
    duration_score = min(duration / 75.0, 1.0)  # Prefer clips around 75 seconds
    
    engagement_score = (word_density_score * 0.45 + 
                       engagement_ratio * 0.30 + 
                       duration_score * 0.25)
    
    return engagement_score

@dataclass(frozen=True)
class VideoJob:
    input_path: str
    transcription_path: str
    max_clips: int
    output_dir: str


@dataclass(frozen=True)
class ClipResult:
    file: str
    title: str
    start: float
    end: float
    duration: float


def load_or_transcribe(input_path: str, transcription_path: str):
    transcription = load_existing_transcription(transcription_path)
    if transcription is None:
        transcriber = Transcriber(model_size=os.getenv('TRANSCRIPTION_MODEL', 'large-v1'))
        transcription = transcribe_with_progress(input_path, transcriber)
        save_transcription(transcription, transcription_path)
    return transcription


def select_clips(clips, transcription, max_clips: int) -> list:
    valid_clips = [c for c in clips if MIN_CLIP_DURATION <= (c.end_time - c.start_time) <= MAX_CLIP_DURATION]
    selected_clips = []

    if valid_clips:
        clip_scores = [(clip, calculate_engagement_score(clip, transcription)) for clip in valid_clips]
        clip_scores.sort(key=lambda x: x[1], reverse=True)
        # Select up to max_clips, but only include clips with engagement >= 0.6 (for 3rd and beyond)
        for i, (clip, score) in enumerate(clip_scores):
            if i < 2 or score >= 0.6:
                if len(selected_clips) < max_clips:
                    selected_clips.append(clip)
            else:
                break
        print(f'Selected top {len(selected_clips)} clips:')
        for i, clip in enumerate(selected_clips):
            score = calculate_engagement_score(clip, transcription)
            print(f'  Clip {i+1}: {clip.start_time:.1f}s - {clip.end_time:.1f}s (duration: {clip.end_time - clip.start_time:.1f}s, engagement: {score:.3f})')
        print(f'Clip selection criteria: Top engaging clips within {MIN_CLIP_DURATION}-{MAX_CLIP_DURATION} second range')
        return selected_clips

    print(f'No clips found between {MIN_CLIP_DURATION} and {MAX_CLIP_DURATION} seconds.')
    short_clips = [c for c in clips if c.end_time - c.start_time < MIN_CLIP_DURATION]
    if short_clips:
        print('Attempting to extend most engaging short clips to minimum duration...')
        short_clip_scores = [(clip, calculate_engagement_score(clip, transcription)) for clip in short_clips]
        short_clip_scores.sort(key=lambda x: x[1], reverse=True)
        for i, (clip, score) in enumerate(short_clip_scores[:2]):
            extension_needed = MIN_CLIP_DURATION - (clip.end_time - clip.start_time)
            max_extension = min(extension_needed, MAX_CLIP_DURATION - (clip.end_time - clip.start_time))
            extended_clip = Clip(
                start_time=clip.start_time,
                end_time=clip.end_time + max_extension,
                start_char=clip.start_char,
                end_char=clip.end_char
            )
            selected_clips.append(extended_clip)
            print(f'Extended clip {i+1}: {extended_clip.start_time:.1f}s - {extended_clip.end_time:.1f}s (duration: {extended_clip.end_time - extended_clip.start_time:.1f}s)')
    else:
        print('All clips are too long. Trimming most engaging clips to maximum duration...')
        long_clip_scores = [(clip, calculate_engagement_score(clip, transcription)) for clip in clips]
        long_clip_scores.sort(key=lambda x: x[1], reverse=True)
        for i, (clip, score) in enumerate(long_clip_scores[:2]):
            if clip.end_time - clip.start_time <= MAX_CLIP_DURATION:
                continue
            trimmed_clip = Clip(
                start_time=clip.start_time,
                end_time=clip.start_time + MAX_CLIP_DURATION,
                start_char=clip.start_char,
                end_char=clip.end_char
            )
            selected_clips.append(trimmed_clip)
            print(f'Trimmed clip {i+1}: {trimmed_clip.start_time:.1f}s - {trimmed_clip.end_time:.1f}s (duration: {trimmed_clip.end_time - trimmed_clip.start_time:.1f}s)')
    return selected_clips[:max_clips]


def resize_to_short(media_editor, trimmed_path: str, output_path: str) -> None:
    crops = resize(
        video_file_path=trimmed_path,
        pyannote_auth_token=HUGGINGFACE_TOKEN,
        aspect_ratio=(ASPECT_RATIO_WIDTH, ASPECT_RATIO_HEIGHT)
    )
    media_editor.resize_video(
        original_video_file=AudioVideoFile(trimmed_path),
        resized_video_file_path=output_path,
        width=crops.crop_width,
        height=crops.crop_height,
        segments=crops.to_dict()["segments"],
    )


def render_clip(job: VideoJob, transcription, clip, clip_number: int) -> ClipResult:
    print(f'\n--- Processing Clip {clip_number} ---')
    media_editor = MediaEditor()
    trimmed_path = os.path.join(job.output_dir, f'trimmed_clip_{clip_number}.mp4')
    short_path = os.path.join(job.output_dir, f'yt_short_{clip_number}.mp4')
    print('Trimming video to selected clip...')
    media_editor.trim(
        media_file=AudioVideoFile(job.input_path),
        start_time=clip.start_time,
        end_time=clip.end_time,
        trimmed_media_file_path=trimmed_path
    )
    print('Resizing video to 9:16 aspect ratio...')
    resize_to_short(media_editor, trimmed_path, short_path)
    subtitled_path = create_animated_subtitles(short_path, transcription, clip, short_path)

    clip_text = " ".join([w["word"] for w in transcription.get_word_info() if w["start_time"] >= clip.start_time and w["end_time"] <= clip.end_time])
    title = get_viral_title(clip_text, GROQ_API_KEY)
    print(f"\nViral Title for Clip {clip_number}: {title}")

    final_path = os.path.join(job.output_dir, f"{clip_number:02d} {safe_filename(title).strip()}.mp4")
    shutil.move(subtitled_path, final_path)
    for intermediate in (trimmed_path, short_path):
        if os.path.exists(intermediate):
            os.remove(intermediate)
    print(f"Final video saved as: {final_path}\n")
    return ClipResult(
        file=os.path.basename(final_path),
        title=title,
        start=round(clip.start_time, 3),
        end=round(clip.end_time, 3),
        duration=round(clip.end_time - clip.start_time, 3),
    )


def process_video(job: VideoJob) -> List[ClipResult]:
    os.makedirs(job.output_dir, exist_ok=True)
    transcription = load_or_transcribe(job.input_path, job.transcription_path)
    clips = ClipFinder().find_clips(transcription=transcription)
    if not clips:
        print('No clips found in the video.')
        return []
    selected_clips = select_clips(clips, transcription, job.max_clips)
    return [render_clip(job, transcription, clip, number) for number, clip in enumerate(selected_clips, 1)]


def run_interactive() -> None:
    input_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.mp4')]
    if not input_files:
        raise FileNotFoundError('No mp4 file found in input directory.')

    transcription_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('_transcription.pkl')]

    # If more than one mp4, ask user to match transcription files (if any)
    video_transcription_map = {}
    if len(input_files) > 1:
        print("Multiple video files detected:")
        for idx, f in enumerate(input_files, 1):
            print(f"  {idx}) {f}")
        print("\nAvailable transcription files:")
        for idx, f in enumerate(transcription_files, 1):
            print(f"  {idx}) {f}")
        print("\nFor each video, enter the number of the matching transcription file, or 0 to transcribe from scratch.")
        for video_file in input_files:
            while True:
                try:
                    match = input(f"Match transcription for '{video_file}' (0 for none): ").strip().replace('\n', '')
                    match_idx = int(match)
                    if match_idx == 0:
                        video_transcription_map[video_file] = None
                        break
                    elif 1 <= match_idx <= len(transcription_files):
                        video_transcription_map[video_file] = transcription_files[match_idx-1]
                        break
                    else:
                        print("Invalid choice. Try again.")
                except Exception:
                    print("Invalid input. Try again.")
    else:
        video_file = input_files[0]
        expected_trans = f"{os.path.splitext(video_file)[0]}_transcription.pkl"
        video_transcription_map[video_file] = expected_trans if expected_trans in transcription_files else None

    # Prompt user for number of clips for each video BEFORE any processing
    video_max_clips = {}
    clip_ranges = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12)]
    for video_file in video_transcription_map:
        print(f"\nHow many clips do you want for '{video_file}'?")
        for i, (low, high) in enumerate(clip_ranges, 1):
            print(f"  {i}) {low}-{high}")
        try:
            user_choice = int(input("Your choice: ").strip().replace('\n', ''))
            if not (1 <= user_choice <= len(clip_ranges)):
                raise ValueError
        except Exception:
            print("Invalid input. Defaulting to 2 clips.")
            user_choice = 1
        max_clips = clip_ranges[user_choice-1][1]
        print(f"Will select up to {max_clips} clips (if available and engaging).\n")
        video_max_clips[video_file] = max_clips

    for video_idx, (video_file, transcription_file) in enumerate(video_transcription_map.items(), 1):
        print(f"\n=== Processing Video {video_idx}/{len(video_transcription_map)}: {video_file} ===")
        input_path = os.path.abspath(os.path.join(INPUT_DIR, video_file))
        # Choosing 0 forces a fresh transcription, as in the original flow
        transcription_path = (os.path.join(INPUT_DIR, transcription_file) if transcription_file
                              else get_transcription_file_path(input_path))
        if not transcription_file and os.path.exists(transcription_path):
            os.remove(transcription_path)
        process_video(VideoJob(input_path, transcription_path, video_max_clips[video_file], OUTPUT_DIR))


if __name__ == "__main__":
    run_interactive()
