"""
Audio Processing Module for PlaudBlender

Enables rich audio analysis beyond transcription:
1. Audio file download and storage from Plaud
2. Whisper transcription with speaker diarization
3. CLAP audio embeddings for audio-to-audio similarity search
4. Gemini audio analysis for tone/sentiment/context extraction

This transforms PlaudBlender from a transcript-only system into a
full multimodal audio intelligence platform.

Reference: gemini-deep-research2.txt - Multimodal RAG, ColPali concept applied to audio
"""
import os
import io
import hashlib
import logging
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, BinaryIO
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Audio storage directory
AUDIO_STORAGE_DIR = os.getenv("AUDIO_STORAGE_DIR", "data/audio")
Path(AUDIO_STORAGE_DIR).mkdir(parents=True, exist_ok=True)


@dataclass
class AudioSegment:
    """A timestamped segment of audio with speaker info."""
    text: str
    start_ms: int
    end_ms: int
    speaker: Optional[str] = None
    confidence: float = 1.0
    language: Optional[str] = None
    
    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "speaker": self.speaker,
            "confidence": self.confidence,
            "language": self.language,
        }


@dataclass
class AudioAnalysis:
    """Complete analysis of an audio file."""
    recording_id: str
    duration_ms: int
    
    # Transcription
    transcript: str = ""
    segments: List[AudioSegment] = field(default_factory=list)
    speakers: List[str] = field(default_factory=list)
    language: str = "en"
    
    # Embeddings
    audio_embedding: Optional[List[float]] = None
    embedding_model: str = ""
    
    # Sentiment/Tone (from Gemini)
    tone: Optional[str] = None
    sentiment: Optional[str] = None
    energy_level: Optional[str] = None
    key_moments: List[Dict] = field(default_factory=list)
    gemini_insights: Optional[str] = None
    
    # Metadata
    file_path: Optional[str] = None
    file_size_bytes: int = 0
    audio_format: str = ""
    sample_rate: int = 0
    analyzed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "recording_id": self.recording_id,
            "duration_ms": self.duration_ms,
            "transcript": self.transcript[:500] + "..." if len(self.transcript) > 500 else self.transcript,
            "segment_count": len(self.segments),
            "speakers": self.speakers,
            "language": self.language,
            "has_embedding": self.audio_embedding is not None,
            "embedding_model": self.embedding_model,
            "tone": self.tone,
            "sentiment": self.sentiment,
            "energy_level": self.energy_level,
            "key_moments": len(self.key_moments),
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "analyzed_at": self.analyzed_at,
        }


# ============================================================================
# AUDIO FILE MANAGEMENT
# ============================================================================

class AudioFileManager:
    """
    Manages audio file download, storage, and retrieval.
    
    Audio files are stored locally with consistent naming:
    - data/audio/{recording_id}.{format}
    - Supports mp3, wav, m4a, webm
    """
    
    def __init__(self, storage_dir: str = AUDIO_STORAGE_DIR):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def get_audio_path(self, recording_id: str, format: str = "mp3") -> Path:
        """Get local path for audio file."""
        return self.storage_dir / f"{recording_id}.{format}"
    
    def audio_exists(self, recording_id: str) -> bool:
        """Check if audio file exists locally."""
        for ext in ["mp3", "wav", "m4a", "webm"]:
            if self.get_audio_path(recording_id, ext).exists():
                return True
        return False
    
    def get_existing_audio_path(self, recording_id: str) -> Optional[Path]:
        """Get path to existing audio file if it exists."""
        for ext in ["mp3", "wav", "m4a", "webm"]:
            path = self.get_audio_path(recording_id, ext)
            if path.exists():
                return path
        return None
    
    def download_from_url(
        self, 
        url: str, 
        recording_id: str,
        format: str = "mp3",
    ) -> Path:
        """
        Download audio file from URL.
        
        Args:
            url: Audio file URL
            recording_id: Recording ID for filename
            format: Audio format extension
            
        Returns:
            Path to downloaded file
        """
        import requests
        
        path = self.get_audio_path(recording_id, format)
        
        if path.exists():
            logger.info(f"Audio already exists: {path}")
            return path
        
        logger.info(f"⬇️ Downloading audio: {recording_id}")
        
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"✅ Downloaded: {path} ({path.stat().st_size / 1024:.1f} KB)")
        return path
    
    def save_audio_bytes(
        self,
        audio_bytes: bytes,
        recording_id: str,
        format: str = "mp3",
    ) -> Path:
        """Save audio bytes to file."""
        path = self.get_audio_path(recording_id, format)
        path.write_bytes(audio_bytes)
        logger.info(f"💾 Saved audio: {path}")
        return path
    
    def list_audio_files(self) -> List[Dict]:
        """List all stored audio files."""
        files = []
        for path in self.storage_dir.glob("*.*"):
            if path.suffix.lower() in [".mp3", ".wav", ".m4a", ".webm"]:
                files.append({
                    "recording_id": path.stem,
                    "path": str(path),
                    "format": path.suffix[1:],
                    "size_bytes": path.stat().st_size,
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                })
        return files
    
    def get_audio_stats(self) -> Dict:
        """Get statistics about stored audio."""
        files = self.list_audio_files()
        total_size = sum(f["size_bytes"] for f in files)
        return {
            "total_files": len(files),
            "total_size_mb": total_size / (1024 * 1024),
            "formats": list(set(f["format"] for f in files)),
        }


# ============================================================================
# WHISPER TRANSCRIPTION WITH DIARIZATION
# ============================================================================

class WhisperTranscriber:
    """
    High-quality transcription using OpenAI Whisper.
    
    Features:
    - Accurate transcription with timestamps
    - Speaker diarization (who said what)
    - Language detection
    - Word-level timestamps
    """
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper transcriber.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        self._model = None
        self._diarization_pipeline = None
    
    def _load_whisper(self):
        """Lazy load Whisper model."""
        if self._model is None:
            try:
                import whisper
                logger.info(f"Loading Whisper model: {self.model_size}")
                self._model = whisper.load_model(self.model_size)
            except ImportError:
                logger.error("Whisper not installed. Run: pip install openai-whisper")
                raise
        return self._model
    
    def _load_diarization(self):
        """Lazy load speaker diarization pipeline."""
        if self._diarization_pipeline is None:
            try:
                from pyannote.audio import Pipeline
                import torch
                
                hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
                if not hf_token:
                    logger.warning("HF_TOKEN not set, diarization unavailable")
                    return None
                
                logger.info("Loading speaker diarization model...")
                self._diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token,
                )
                
                if torch.cuda.is_available():
                    self._diarization_pipeline.to(torch.device("cuda"))
                    
            except ImportError:
                logger.warning("pyannote.audio not installed for diarization")
                return None
            except Exception as e:
                logger.warning(f"Could not load diarization: {e}")
                return None
        
        return self._diarization_pipeline
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        enable_diarization: bool = True,
    ) -> Tuple[str, List[AudioSegment]]:
        """
        Transcribe audio file with timestamps.
        
        Args:
            audio_path: Path to audio file
            language: Language code (None for auto-detect)
            enable_diarization: Enable speaker identification
            
        Returns:
            Tuple of (full transcript, list of segments)
        """
        model = self._load_whisper()
        
        logger.info(f"🎤 Transcribing: {audio_path}")
        
        # Transcribe with Whisper
        result = model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            verbose=False,
        )
        
        full_text = result["text"]
        detected_lang = result.get("language", "en")
        
        # Build segments from Whisper output
        segments = []
        for seg in result.get("segments", []):
            segments.append(AudioSegment(
                text=seg["text"].strip(),
                start_ms=int(seg["start"] * 1000),
                end_ms=int(seg["end"] * 1000),
                confidence=seg.get("avg_logprob", 0) + 1,  # Normalize
                language=detected_lang,
            ))
        
        # Add speaker diarization if enabled
        if enable_diarization:
            segments = self._add_speaker_labels(audio_path, segments)
        
        logger.info(f"✅ Transcribed {len(segments)} segments, {len(full_text)} chars")
        
        return full_text, segments
    
    def _add_speaker_labels(
        self,
        audio_path: str,
        segments: List[AudioSegment],
    ) -> List[AudioSegment]:
        """Add speaker labels to segments using diarization."""
        pipeline = self._load_diarization()
        if pipeline is None:
            return segments
        
        try:
            logger.info("🎭 Running speaker diarization...")
            diarization = pipeline(audio_path)
            
            # Build speaker timeline
            speaker_timeline = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_timeline.append({
                    "start_ms": int(turn.start * 1000),
                    "end_ms": int(turn.end * 1000),
                    "speaker": speaker,
                })
            
            # Assign speakers to segments
            for segment in segments:
                seg_mid = (segment.start_ms + segment.end_ms) / 2
                for turn in speaker_timeline:
                    if turn["start_ms"] <= seg_mid <= turn["end_ms"]:
                        segment.speaker = turn["speaker"]
                        break
            
            speakers = list(set(s.speaker for s in segments if s.speaker))
            logger.info(f"✅ Identified {len(speakers)} speakers")
            
        except Exception as e:
            logger.warning(f"Diarization failed: {e}")
        
        return segments


# ============================================================================
# CLAP AUDIO EMBEDDINGS
# ============================================================================

class CLAPEmbedder:
    """
    Audio embeddings using CLAP (Contrastive Language-Audio Pretraining).
    
    Enables:
    - Audio-to-audio similarity search
    - Text-to-audio search (find audio matching a description)
    - Audio clustering and classification
    """
    
    def __init__(self, model_name: str = "laion/larger_clap_general"):
        """
        Initialize CLAP embedder.
        
        Args:
            model_name: HuggingFace model name for CLAP
        """
        self.model_name = model_name
        self._model = None
        self._processor = None
    
    def _load_model(self):
        """Lazy load CLAP model."""
        if self._model is None:
            try:
                from transformers import ClapModel, ClapProcessor
                import torch
                
                logger.info(f"Loading CLAP model: {self.model_name}")
                self._model = ClapModel.from_pretrained(self.model_name)
                self._processor = ClapProcessor.from_pretrained(self.model_name)
                
                if torch.cuda.is_available():
                    self._model = self._model.to("cuda")
                
            except ImportError:
                logger.error("transformers not installed. Run: pip install transformers")
                raise
        
        return self._model, self._processor
    
    def embed_audio(self, audio_path: str) -> List[float]:
        """
        Generate embedding for audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio embedding vector
        """
        import librosa
        import torch
        
        model, processor = self._load_model()
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=48000)
        
        # Process and embed
        inputs = processor(audios=audio, sampling_rate=sr, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            audio_embed = model.get_audio_features(**inputs)
        
        embedding = audio_embed.squeeze().cpu().numpy().tolist()
        logger.info(f"🔊 Generated {len(embedding)}d audio embedding")
        
        return embedding
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text query (for audio search).
        
        Args:
            text: Text description to search for
            
        Returns:
            Text embedding vector (same space as audio)
        """
        import torch
        
        model, processor = self._load_model()
        
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            text_embed = model.get_text_features(**inputs)
        
        return text_embed.squeeze().cpu().numpy().tolist()
    
    def compute_similarity(
        self,
        audio_embedding: List[float],
        query_embedding: List[float],
    ) -> float:
        """Compute cosine similarity between embeddings."""
        import numpy as np
        
        a = np.array(audio_embedding)
        b = np.array(query_embedding)
        
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ============================================================================
# GEMINI AUDIO ANALYSIS
# ============================================================================

class GeminiAudioAnalyzer:
    """
    Rich audio analysis using Gemini's native audio understanding.
    
    Extracts:
    - Tone and sentiment from voice
    - Key emotional moments
    - Meeting energy/dynamics
    - Non-verbal cues (pauses, emphasis)
    - Contextual insights beyond transcription
    """
    
    ANALYSIS_PROMPT = '''Analyze this audio recording and provide insights that go beyond the transcript.

Focus on:
1. TONE: What is the overall emotional tone? (professional, casual, tense, enthusiastic, etc.)
2. SENTIMENT: What is the predominant sentiment? (positive, negative, neutral, mixed)
3. ENERGY: How would you describe the energy level? (high-energy, calm, urgent, relaxed)
4. KEY MOMENTS: Identify 3-5 significant moments with timestamps (if audible)
5. SPEAKER DYNAMICS: How do speakers interact? Any tension or agreement?
6. NON-VERBAL: Any notable pauses, emphasis, or vocal cues?
7. INSIGHTS: What can you infer that isn't explicitly stated?

Respond in JSON format:
{
  "tone": "...",
  "sentiment": "positive|negative|neutral|mixed",
  "energy_level": "high|medium|low",
  "key_moments": [
    {"timestamp": "approx MM:SS", "description": "...", "significance": "..."}
  ],
  "speaker_dynamics": "...",
  "non_verbal_cues": ["..."],
  "insights": "...",
  "overall_summary": "..."
}'''

    def __init__(self):
        self._model = None
    
    def _get_model(self):
        """Lazy load Gemini model."""
        if self._model is None:
            import google.generativeai as genai
            
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY required for Gemini audio analysis")
            
            genai.configure(api_key=api_key)
            # Use Gemini 2.0 Flash which has native audio support
            self._model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        return self._model
    
    def analyze_audio(self, audio_path: str) -> Dict:
        """
        Analyze audio file with Gemini.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Analysis results dict
        """
        import google.generativeai as genai
        import json
        
        model = self._get_model()
        
        logger.info(f"🧠 Analyzing audio with Gemini: {audio_path}")
        
        # Upload audio file to Gemini
        audio_file = genai.upload_file(audio_path)
        
        # Generate analysis
        response = model.generate_content([
            self.ANALYSIS_PROMPT,
            audio_file,
        ])
        
        # Parse response
        text = response.text.strip()
        
        try:
            # Extract JSON from response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            analysis = json.loads(text)
            logger.info(f"✅ Gemini analysis complete: tone={analysis.get('tone')}")
            return analysis
            
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse Gemini response as JSON: {e}")
            return {
                "raw_analysis": text,
                "parse_error": str(e),
            }
    
    def quick_sentiment(self, audio_path: str) -> Tuple[str, str]:
        """
        Quick sentiment and tone check.
        
        Returns:
            Tuple of (sentiment, tone)
        """
        analysis = self.analyze_audio(audio_path)
        return (
            analysis.get("sentiment", "unknown"),
            analysis.get("tone", "unknown"),
        )


# ============================================================================
# UNIFIED AUDIO PROCESSOR
# ============================================================================

class AudioProcessor:
    """
    Unified audio processing pipeline.
    
    Combines all audio analysis capabilities:
    1. Download/store audio files
    2. Transcribe with Whisper + diarization
    3. Generate CLAP embeddings
    4. Analyze with Gemini
    
    Usage:
        processor = AudioProcessor()
        analysis = processor.process_recording(recording_id, audio_url)
    """
    
    def __init__(
        self,
        storage_dir: str = AUDIO_STORAGE_DIR,
        whisper_model: str = "base",
        enable_clap: bool = True,
        enable_gemini: bool = True,
    ):
        """
        Initialize unified processor.
        
        Args:
            storage_dir: Directory for audio file storage
            whisper_model: Whisper model size
            enable_clap: Enable CLAP audio embeddings
            enable_gemini: Enable Gemini analysis
        """
        self.file_manager = AudioFileManager(storage_dir)
        self.transcriber = WhisperTranscriber(whisper_model)
        
        self.enable_clap = enable_clap
        self.enable_gemini = enable_gemini
        
        self._clap = None
        self._gemini = None
    
    def _get_clap(self) -> Optional[CLAPEmbedder]:
        """Get CLAP embedder if enabled."""
        if not self.enable_clap:
            return None
        if self._clap is None:
            try:
                self._clap = CLAPEmbedder()
            except Exception as e:
                logger.warning(f"CLAP unavailable: {e}")
        return self._clap
    
    def _get_gemini(self) -> Optional[GeminiAudioAnalyzer]:
        """Get Gemini analyzer if enabled."""
        if not self.enable_gemini:
            return None
        if self._gemini is None:
            try:
                self._gemini = GeminiAudioAnalyzer()
            except Exception as e:
                logger.warning(f"Gemini unavailable: {e}")
        return self._gemini
    
    def process_recording(
        self,
        recording_id: str,
        audio_url: Optional[str] = None,
        audio_path: Optional[str] = None,
        enable_transcription: bool = True,
        enable_diarization: bool = True,
        enable_embeddings: bool = True,
        enable_analysis: bool = True,
    ) -> AudioAnalysis:
        """
        Full audio processing pipeline.
        
        Args:
            recording_id: Unique recording identifier
            audio_url: URL to download audio (if not already stored)
            audio_path: Direct path to audio file (overrides URL)
            enable_transcription: Run Whisper transcription
            enable_diarization: Run speaker diarization
            enable_embeddings: Generate CLAP embeddings
            enable_analysis: Run Gemini analysis
            
        Returns:
            Complete AudioAnalysis object
        """
        import time
        start_time = time.time()
        
        logger.info(f"🎵 Processing audio: {recording_id}")
        
        # Get audio file path
        if audio_path:
            path = Path(audio_path)
        elif audio_url:
            path = self.file_manager.download_from_url(audio_url, recording_id)
        else:
            path = self.file_manager.get_existing_audio_path(recording_id)
        
        if not path or not path.exists():
            raise FileNotFoundError(f"No audio file for recording: {recording_id}")
        
        # Get audio duration
        try:
            import librosa
            duration = librosa.get_duration(path=str(path))
            duration_ms = int(duration * 1000)
        except:
            duration_ms = 0
        
        # Initialize analysis object
        analysis = AudioAnalysis(
            recording_id=recording_id,
            duration_ms=duration_ms,
            file_path=str(path),
            file_size_bytes=path.stat().st_size,
            audio_format=path.suffix[1:],
        )
        
        # 1. Transcription
        if enable_transcription:
            try:
                transcript, segments = self.transcriber.transcribe(
                    str(path),
                    enable_diarization=enable_diarization,
                )
                analysis.transcript = transcript
                analysis.segments = segments
                analysis.speakers = list(set(s.speaker for s in segments if s.speaker))
                if segments:
                    analysis.language = segments[0].language or "en"
            except Exception as e:
                logger.error(f"Transcription failed: {e}")
        
        # 2. CLAP Embeddings
        if enable_embeddings:
            clap = self._get_clap()
            if clap:
                try:
                    analysis.audio_embedding = clap.embed_audio(str(path))
                    analysis.embedding_model = "laion/larger_clap_general"
                except Exception as e:
                    logger.warning(f"CLAP embedding failed: {e}")
        
        # 3. Gemini Analysis
        if enable_analysis:
            gemini = self._get_gemini()
            if gemini:
                try:
                    gemini_result = gemini.analyze_audio(str(path))
                    analysis.tone = gemini_result.get("tone")
                    analysis.sentiment = gemini_result.get("sentiment")
                    analysis.energy_level = gemini_result.get("energy_level")
                    analysis.key_moments = gemini_result.get("key_moments", [])
                    analysis.gemini_insights = gemini_result.get("insights")
                except Exception as e:
                    logger.warning(f"Gemini analysis failed: {e}")
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Audio processing complete in {elapsed:.1f}s")
        
        return analysis
    
    def process_batch(
        self,
        recordings: List[Dict],
        **kwargs,
    ) -> List[AudioAnalysis]:
        """Process multiple recordings."""
        results = []
        for rec in recordings:
            try:
                analysis = self.process_recording(
                    recording_id=rec.get("id"),
                    audio_url=rec.get("audio_url"),
                    **kwargs,
                )
                results.append(analysis)
            except Exception as e:
                logger.error(f"Failed to process {rec.get('id')}: {e}")
        return results


# ============================================================================
# AUDIO SEARCH
# ============================================================================

class AudioSearchService:
    """
    Search for recordings by audio similarity.
    
    Enables queries like:
    - "Find recordings that sound similar to this one"
    - "Find recordings matching 'urgent meeting discussion'"
    - "Find recordings with high energy"
    """
    
    def __init__(self, embedder: Optional[CLAPEmbedder] = None):
        self.embedder = embedder or CLAPEmbedder()
        self._embeddings_cache: Dict[str, List[float]] = {}
    
    def index_audio(self, recording_id: str, embedding: List[float]):
        """Add audio embedding to search index."""
        self._embeddings_cache[recording_id] = embedding
    
    def search_by_text(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Search for audio matching text description.
        
        Args:
            query: Text description (e.g., "excited conversation")
            limit: Max results
            
        Returns:
            List of (recording_id, similarity_score) tuples
        """
        if not self._embeddings_cache:
            return []
        
        query_embedding = self.embedder.embed_text(query)
        
        results = []
        for rec_id, audio_emb in self._embeddings_cache.items():
            score = self.embedder.compute_similarity(audio_emb, query_embedding)
            results.append((rec_id, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def search_by_audio(
        self,
        audio_path: str,
        limit: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find recordings similar to given audio file.
        
        Args:
            audio_path: Path to query audio
            limit: Max results
            
        Returns:
            List of (recording_id, similarity_score) tuples
        """
        if not self._embeddings_cache:
            return []
        
        query_embedding = self.embedder.embed_audio(audio_path)
        
        results = []
        for rec_id, audio_emb in self._embeddings_cache.items():
            score = self.embedder.compute_similarity(audio_emb, query_embedding)
            results.append((rec_id, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def get_audio_processor(**kwargs) -> AudioProcessor:
    """Factory for audio processor."""
    return AudioProcessor(**kwargs)


def get_audio_file_manager() -> AudioFileManager:
    """Factory for file manager."""
    return AudioFileManager()


if __name__ == "__main__":
    # Quick test
    print("\n" + "="*70)
    print("Audio Processing Module Test")
    print("="*70)
    
    manager = get_audio_file_manager()
    stats = manager.get_audio_stats()
    print(f"\n📊 Audio Storage Stats: {stats}")
    
    print("\n✅ Module loaded successfully!")
    print("Components available:")
    print("  - AudioFileManager: Download and store audio files")
    print("  - WhisperTranscriber: Transcribe with speaker diarization")
    print("  - CLAPEmbedder: Audio embeddings for similarity search")
    print("  - GeminiAudioAnalyzer: Tone/sentiment/insight extraction")
    print("  - AudioProcessor: Unified processing pipeline")
    print("  - AudioSearchService: Audio-to-audio and text-to-audio search")
