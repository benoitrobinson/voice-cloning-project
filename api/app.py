"""
FastAPI implementation for Voice Cloning System
Provides RESTful API endpoints for voice cloning operations
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import asyncio
from pathlib import Path
import uuid
import logging
from datetime import datetime

# Import our modules
from src.voice_cloner import VoiceCloner
from src.audio_processor import AudioProcessor
from src.consent_manager import ConsentManager, ConsentType
from src.quality_enhancer import QualityEnhancer
from config.settings import Settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Voice Cloning API",
    description="Professional voice cloning system with ethical safeguards",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global instances (in production, use dependency injection)
settings = Settings()
audio_processor = AudioProcessor(settings)
consent_manager = ConsentManager(settings)
voice_cloner = VoiceCloner(settings)
quality_enhancer = QualityEnhancer(settings)


# Pydantic models for API
class ConsentRequest(BaseModel):
    speaker_name: str = Field(..., description="Name of the speaker")
    speaker_email: str = Field(..., description="Email for consent verification")
    consent_type: str = Field(default="basic", description="Type of consent")
    use_cases: List[str] = Field(..., description="List of approved use cases")
    duration_days: Optional[int] = Field(None, description="Consent duration in days")
    restrictions: Optional[List[str]] = Field(default=[], description="Usage restrictions")


class ConsentResponse(BaseModel):
    consent_id: str
    status: str
    expires_at: str
    form_url: Optional[str] = None


class VoiceCloneRequest(BaseModel):
    speaker_name: str
    consent_id: str
    language: str = Field(default="en", description="Target language code")
    enhance_quality: bool = Field(default=True, description="Apply quality enhancement")


class VoiceCloneResponse(BaseModel):
    voice_id: str
    speaker_name: str
    language: str
    quality_score: float
    status: str


class SynthesisRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize", max_length=5000)
    voice_id: str = Field(..., description="ID of the cloned voice")
    emotion: Optional[str] = Field(None, description="Emotion (happy, sad, angry, etc.)")
    style_params: Optional[Dict[str, Any]] = Field(default={}, description="Style parameters")
    output_format: str = Field(default="wav", description="Output format (wav, mp3)")


class QualityCheckResponse(BaseModel):
    passed: bool
    metrics: Dict[str, float]
    recommendations: List[str]


# Helper functions
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key for authentication."""
    if credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return credentials.credentials


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Voice Cloning API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs"
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "gpu_available": voice_cloner.device == "cuda"
    }


@app.post("/consent/create", response_model=ConsentResponse, tags=["Consent"])
async def create_consent(
    request: ConsentRequest,
    api_key: str = Depends(verify_api_key)
):
    """Create a new consent record."""
    try:
        consent_type = ConsentType(request.consent_type)
        
        consent_id = consent_manager.create_consent(
            speaker_name=request.speaker_name,
            speaker_email=request.speaker_email,
            consent_type=consent_type,
            use_cases=request.use_cases,
            duration_days=request.duration_days,
            restrictions=request.restrictions
        )
        
        # Generate consent form
        form_path = consent_manager.generate_consent_form(consent_type)
        
        consent = consent_manager.get_consent(consent_id)
        
        return ConsentResponse(
            consent_id=consent_id,
            status=consent['status'],
            expires_at=consent['expires_at'],
            form_url=f"/consent/form/{Path(form_path).name}"
        )
        
    except Exception as e:
        logger.error(f"Error creating consent: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/consent/{consent_id}", tags=["Consent"])
async def get_consent(
    consent_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get consent details."""
    try:
        consent = consent_manager.get_consent(consent_id)
        return consent
    except ValueError:
        raise HTTPException(status_code=404, detail="Consent not found")
    except Exception as e:
        logger.error(f"Error getting consent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/consent/{consent_id}/revoke", tags=["Consent"])
async def revoke_consent(
    consent_id: str,
    reason: str,
    api_key: str = Depends(verify_api_key)
):
    """Revoke a consent."""
    try:
        success = consent_manager.revoke_consent(
            consent_id=consent_id,
            reason=reason,
            revoked_by="api_user"
        )
        
        if success:
            return {"status": "revoked", "consent_id": consent_id}
        else:
            raise HTTPException(status_code=400, detail="Failed to revoke consent")
            
    except Exception as e:
        logger.error(f"Error revoking consent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice/check-quality", response_model=QualityCheckResponse, tags=["Voice"])
async def check_audio_quality(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """Check if audio file meets quality requirements."""
    try:
        # Save uploaded file temporarily
        temp_path = Path(f"/tmp/{uuid.uuid4()}_{file.filename}")
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Validate quality
        result = audio_processor.validate_recording_quality(temp_path)
        
        # Clean up
        temp_path.unlink()
        
        return QualityCheckResponse(
            passed=result['passed'],
            metrics=result['metrics'],
            recommendations=result['recommendations']
        )
        
    except Exception as e:
        logger.error(f"Error checking quality: {e}")
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice/clone", response_model=VoiceCloneResponse, tags=["Voice"])
async def clone_voice(
    request: VoiceCloneRequest,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """Clone a voice from uploaded audio."""
    temp_path = None
    try:
        # Verify consent
        if not consent_manager.verify_consent(
            request.consent_id,
            request.speaker_name,
            "voice_cloning"
        ):
            raise HTTPException(
                status_code=403,
                detail="Valid consent not found or expired"
            )
        
        # Save uploaded file
        temp_path = Path(f"/tmp/{uuid.uuid4()}_{file.filename}")
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process audio
        processed_audio = audio_processor.process_audio(temp_path)
        
        # Clone voice
        voice_profile = voice_cloner.create_voice_clone(
            processed_audio,
            request.speaker_name,
            request.language
        )
        
        # Enhance if requested
        if request.enhance_quality:
            voice_profile = quality_enhancer.enhance_voice(voice_profile)
        
        # Save voice profile
        voice_id = voice_cloner.save_voice_profile(
            voice_profile,
            request.speaker_name,
            request.consent_id
        )
        
        # Calculate quality score
        quality_score = quality_enhancer.evaluate_quality(voice_profile)
        
        return VoiceCloneResponse(
            voice_id=voice_id,
            speaker_name=request.speaker_name,
            language=request.language,
            quality_score=quality_score,
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cloning voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


@app.post("/voice/synthesize", tags=["Voice"])
async def synthesize_speech(
    request: SynthesisRequest,
    api_key: str = Depends(verify_api_key)
):
    """Synthesize speech using cloned voice."""
    try:
        # Load voice profile
        voice_profile = voice_cloner.load_voice_profile(request.voice_id)
        
        # Apply emotion if specified
        if request.emotion:
            voice_profile = voice_cloner.apply_emotion(
                voice_profile,
                request.emotion,
                strength=request.style_params.get("emotion_strength", 1.0)
            )
        
        # Run synthesis in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        audio_path = await loop.run_in_executor(
            None,
            voice_cloner.synthesize,
            request.text,
            voice_profile,
            request.style_params
        )
        
        # Apply post-processing
        enhanced_path = quality_enhancer.post_process(audio_path)
        
        # Convert format if needed
        if request.output_format != "wav":
            # Implement format conversion
            pass
        
        # Return audio file
        return FileResponse(
            enhanced_path,
            media_type=f"audio/{request.output_format}",
            filename=f"synthesis_{request.voice_id}_{uuid.uuid4()}.{request.output_format}"
        )
        
    except Exception as e:
        logger.error(f"Error synthesizing speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/voice/{voice_id}", tags=["Voice"])
async def get_voice_info(
    voice_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get voice profile information."""
    try:
        voice_profile = voice_cloner.load_voice_profile(voice_id)
        
        # Remove sensitive embedding data
        safe_profile = {
            "voice_id": voice_profile["voice_id"],
            "speaker_name": voice_profile["speaker_name"],
            "language": voice_profile["language"],
            "created_at": voice_profile["created_at"],
            "model_version": voice_profile["model_version"],
            "audio_duration": voice_profile["audio_duration"]
        }
        
        return safe_profile
        
    except ValueError:
        raise HTTPException(status_code=404, detail="Voice not found")
    except Exception as e:
        logger.error(f"Error getting voice info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/voices", tags=["Voice"])
async def list_voices(
    api_key: str = Depends(verify_api_key)
):
    """List all available voices."""
    try:
        voices_dir = Path(settings.voice_storage_path)
        voices = []
        
        for voice_file in voices_dir.glob("*.json"):
            if voice_file.name.startswith("."):
                continue
                
            try:
                voice_id = voice_file.stem
                voice_profile = voice_cloner.load_voice_profile(voice_id)
                
                voices.append({
                    "voice_id": voice_id,
                    "speaker_name": voice_profile["speaker_name"],
                    "language": voice_profile["language"],
                    "created_at": voice_profile["created_at"]
                })
            except Exception as e:
                logger.error(f"Error reading voice {voice_file}: {e}")
                continue
        
        return {"voices": voices, "count": len(voices)}
        
    except Exception as e:
        logger.error(f"Error listing voices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time synthesis (optional)
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws/synthesize")
async def websocket_synthesize(websocket: WebSocket):
    """WebSocket endpoint for real-time synthesis."""
    await websocket.accept()
    
    try:
        while True:
            # Receive synthesis request
            data = await websocket.receive_json()
            
            # Validate API key
            if data.get("api_key") != settings.api_key:
                await websocket.send_json({"error": "Invalid API key"})
                continue
            
            # Synthesize
            try:
                voice_profile = voice_cloner.load_voice_profile(data["voice_id"])
                
                # Apply emotion if specified
                if data.get("emotion"):
                    voice_profile = voice_cloner.apply_emotion(
                        voice_profile,
                        data["emotion"]
                    )
                
                # Synthesize in chunks for streaming
                # (Simplified - implement actual streaming)
                audio_path = voice_cloner.synthesize(
                    data["text"],
                    voice_profile,
                    data.get("style_params", {})
                )
                
                # Send result
                await websocket.send_json({
                    "status": "complete",
                    "audio_url": f"/audio/{Path(audio_path).name}"
                })
                
            except Exception as e:
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info"
    )
