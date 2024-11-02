# AI Medical Transcription Analyzer

An automated tool that transcribes medical audio (from files or YouTube) and provides detailed medical analysis using AI. The system uses OpenAI's Whisper for speech-to-text transcription and Anthropic's Claude for medical content analysis.

## Features

- 🎯 Transcribe audio from local files or YouTube URLs
- 🏥 Detailed medical content analysis
- 🎨 Color-coded categorization of medical information
- 🔍 Identification of Protected Health Information (PHI)
- 💊 Medication and treatment tracking
- 📊 Structured output in both raw and analyzed formats

### Output Files

The script generates two output files:
- `{output_name}_transcript.txt`: Raw transcript of the audio
- `{output_name}_analyzed.html`: Color-coded HTML analysis with medical categorization

## Medical Analysis Categories

The analysis provides color-coded categorization of:

1. 🔴 Protected Health Information (RED)
   - Patient identifiers
   - Demographics
   - Organizations mentioned

2. 💚 Medical Conditions & History (GREEN)
   - Current conditions/symptoms
   - Past medical history
   - Family history

3. ✍️ Anatomical References (ITALICS)
   - Body parts/systems discussed
   - Anatomical locations

4. 💛 Medications (YELLOW)
   - Current medications
   - Discussed medications
   - Supplements

5. 💙 Tests, Treatments & Procedures (BLUE)
   - Performed procedures
   - Recommended tests
   - Treatment plans

## Technical Details

- Uses Whisper's "base" model for transcription
- Automatically utilizes GPU (CUDA) if available
- Supports various audio formats through FFmpeg
- YouTube downloads are handled via yt-dlp
- Analysis performed using Claude 3.5 Sonnet

## Limitations

- Whisper base model has a moderate level of accuracy
- Processing time depends on audio length and hardware
- YouTube videos must be publicly accessible
- API rate limits may apply for Claude analysis

## Security Note

⚠️ This tool processes potentially sensitive medical information. Ensure compliance with relevant privacy regulations (HIPAA, etc.) when using in a professional context.
