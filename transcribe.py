import whisper
import anthropic
import yt_dlp
import os
import torch
from dotenv import load_dotenv

# Initialize models with CUDA if available (faster as it uses GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
whisper_model = whisper.load_model("base").to(DEVICE)



# Load environment variables from .env file
load_dotenv()
claude_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Medical analysis prompt
prompt = """You are a medical transcript analyzer. First, analyze the transcript and create a color-coded summary grouped by category. Then provide the full marked-up transcript.

Part 1 - Summary (Group findings by category):

1. Protected Health Information (RED):
<span style="color: red;">
- List all patient identifiers
- Demographics
- Organizations mentioned
</span>

2. Medical Conditions & History (GREEN):
<span style="background-color: lightgreen;">
- Current conditions/symptoms
- Past medical history
- Family history
</span>

3. Anatomical References (ITALICS):
<em>
- List all body parts/systems discussed
- Anatomical locations
</em>

4. Medications (YELLOW):
<span style="background-color: yellow;">
- Current medications
- Discussed medications
- Supplements
</span>

5. Tests, Treatments & Procedures (BLUE):
<span style="background-color: lightblue;">
- Performed procedures
- Recommended tests
- Treatment plans
</span>

Part 2 - Full Transcript:
Below is the color-coded transcript using the same formatting:
- PHI: <span style="color: red;">text</span>
- Conditions: <span style="background-color: lightgreen;">text</span>
- Anatomy: <em>text</em>
- Medications: <span style="background-color: yellow;">text</span>
- Procedures: <span style="background-color: lightblue;">text</span>

Please analyze the following transcript:"""

def download_youtube_audio(url, output_path="temp_audio"):
    """Download audio from YouTube video"""
    print(f"Downloading audio from YouTube: {url}")
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f'{output_path}.%(ext)s',
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    return f"{output_path}.mp3"

def process_input(input_path, output_path):
    """Process either a local file or YouTube URL"""
    temp_file = None
    
    try:
        # Handle YouTube URL
        if input_path.startswith(('http://', 'https://')) and ('youtube.com' in input_path or 'youtu.be' in input_path):
            print("Detected YouTube URL")
            temp_file = download_youtube_audio(input_path)
            audio_path = temp_file
        else:
            audio_path = input_path
        
        # Step 1: Transcribe audio using Whisper
        print("Transcribing audio...")
        result = whisper_model.transcribe(
            audio_path,
            fp16=torch.cuda.is_available()  # Enable half-precision on CUDA
        )
        transcript = result["text"]
        
        # Save raw transcript
        with open(f"{output_path}_transcript.txt", "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"Raw transcript saved to {output_path}_transcript.txt")

        # Step 2: Analyze with Claude
        print("Analyzing with Claude...")
        message = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            messages=[
                {"role": "user", "content": prompt + "\n\n" + transcript}
            ]
        )
        
        # Handle Claude's response properly
        response_content = message.content[0].text if isinstance(message.content, list) else message.content
        
        # Save formatted analysis
        with open(f"{output_path}_analyzed.html", "w", encoding="utf-8") as f:
            f.write(response_content)
        print(f"Analyzed transcript saved to {output_path}_analyzed.html")
    
    finally:
        # Clean up temporary file if it exists
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
            print("Cleaned up temporary files")

if __name__ == "__main__":
    # Example usage
    input_source = "https://www.youtube.com/watch?v=vlfdybUCzFw"  # Can be local file path or YouTube URL
    output_file = "output"  # Base name for output files
    
    process_input(input_source, output_file)