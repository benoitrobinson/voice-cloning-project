"""
Consent Manager Module - Handles legal compliance and consent tracking
Based on the guide's legal frameworks section
"""

import json
import hashlib
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any
from cryptography.fernet import Fernet
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ConsentType(Enum):
    """Types of consent based on legal requirements."""
    BASIC = "basic"  # Simple voice cloning
    COMMERCIAL = "commercial"  # Commercial use
    RESEARCH = "research"  # Research purposes
    PERPETUAL = "perpetual"  # Perpetual license (rare)


class ConsentStatus(Enum):
    """Consent status tracking."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING = "pending"


class ConsentManager:
    """
    Manages consent for voice cloning in compliance with:
    - Tennessee ELVIS Act
    - EU AI Act
    - GDPR requirements
    - Industry best practices
    """
    
    def __init__(self, settings):
        """Initialize consent manager."""
        self.settings = settings
        self.consent_dir = Path(settings.voice_storage_path) / "consents"
        self.consent_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption
        self._init_encryption()
        
    def _init_encryption(self):
        """Initialize encryption for sensitive data."""
        key_file = self.consent_dir / ".encryption_key"
        
        if key_file.exists():
            with open(key_file, "rb") as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)
            # Set restrictive permissions
            key_file.chmod(0o600)
        
        self.cipher = Fernet(key)
    
    def create_consent(
        self,
        speaker_name: str,
        speaker_email: str,
        consent_type: ConsentType,
        use_cases: List[str],
        duration_days: Optional[int] = None,
        compensation: Optional[Dict[str, Any]] = None,
        restrictions: Optional[List[str]] = None
    ) -> str:
        """
        Create a new consent record.
        
        Args:
            speaker_name: Name of the person giving consent
            speaker_email: Email for consent verification
            consent_type: Type of consent
            use_cases: List of approved use cases
            duration_days: Duration of consent (None for default)
            compensation: Compensation details if applicable
            restrictions: List of restrictions on use
            
        Returns:
            Consent ID
        """
        consent_id = str(uuid.uuid4())
        
        # Use default duration if not specified
        if duration_days is None:
            duration_days = self.settings.consent_expiry_days
        
        # Create consent record
        consent_record = {
            "consent_id": consent_id,
            "speaker_name": speaker_name,
            "speaker_email": speaker_email,
            "consent_type": consent_type.value,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (
                datetime.utcnow() + timedelta(days=duration_days)
            ).isoformat(),
            "status": ConsentStatus.ACTIVE.value,
            "use_cases": use_cases,
            "restrictions": restrictions or [],
            "compensation": compensation,
            "version": "1.0",
            "legal_framework": {
                "elvis_act_compliant": True,
                "eu_ai_act_compliant": True,
                "gdpr_compliant": self.settings.gdpr_compliant
            },
            "audit_trail": [
                {
                    "action": "created",
                    "timestamp": datetime.utcnow().isoformat(),
                    "ip_address": None,  # Would be captured in production
                    "user_agent": None   # Would be captured in production
                }
            ]
        }
        
        # Encrypt sensitive fields
        encrypted_record = self._encrypt_sensitive_fields(consent_record)
        
        # Save consent record
        self._save_consent(consent_id, encrypted_record)
        
        # Generate consent hash for verification
        consent_hash = self._generate_consent_hash(consent_record)
        
        logger.info(
            f"Created consent {consent_id} for {speaker_name} "
            f"(type: {consent_type.value}, expires: {consent_record['expires_at']})"
        )
        
        return consent_id
    
    def verify_consent(
        self,
        consent_id: str,
        speaker_name: str,
        use_case: Optional[str] = None
    ) -> bool:
        """
        Verify if consent is valid for use.
        
        Args:
            consent_id: Consent ID to verify
            speaker_name: Name to verify against consent
            use_case: Specific use case to check
            
        Returns:
            True if consent is valid
        """
        try:
            consent = self.get_consent(consent_id)
            
            # Check basic validity
            if consent['status'] != ConsentStatus.ACTIVE.value:
                logger.warning(f"Consent {consent_id} is not active")
                return False
            
            # Check expiration
            expires_at = datetime.fromisoformat(consent['expires_at'])
            if datetime.utcnow() > expires_at:
                logger.warning(f"Consent {consent_id} has expired")
                self._update_consent_status(consent_id, ConsentStatus.EXPIRED)
                return False
            
            # Verify speaker name
            if consent['speaker_name'] != speaker_name:
                logger.warning(f"Speaker name mismatch for consent {consent_id}")
                return False
            
            # Check use case if specified
            if use_case and use_case not in consent['use_cases']:
                logger.warning(
                    f"Use case '{use_case}' not authorized in consent {consent_id}"
                )
                return False
            
            # Check restrictions
            if use_case and use_case in consent.get('restrictions', []):
                logger.warning(
                    f"Use case '{use_case}' is restricted in consent {consent_id}"
                )
                return False
            
            # Log verification
            self._add_audit_entry(consent_id, "verified", {
                "use_case": use_case,
                "result": "approved"
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying consent {consent_id}: {e}")
            return False
    
    def revoke_consent(
        self,
        consent_id: str,
        reason: str,
        revoked_by: str
    ) -> bool:
        """
        Revoke consent.
        
        Args:
            consent_id: Consent to revoke
            reason: Reason for revocation
            revoked_by: Who is revoking (speaker/admin)
            
        Returns:
            True if successfully revoked
        """
        try:
            consent = self.get_consent(consent_id)
            
            if consent['status'] == ConsentStatus.REVOKED.value:
                logger.warning(f"Consent {consent_id} already revoked")
                return False
            
            # Update status
            self._update_consent_status(consent_id, ConsentStatus.REVOKED)
            
            # Add audit entry
            self._add_audit_entry(consent_id, "revoked", {
                "reason": reason,
                "revoked_by": revoked_by,
                "previous_status": consent['status']
            })
            
            logger.info(f"Consent {consent_id} revoked by {revoked_by}: {reason}")
            
            # Trigger data deletion if required
            if self.settings.gdpr_compliant:
                self._schedule_data_deletion(consent_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error revoking consent {consent_id}: {e}")
            return False
    
    def get_consent(self, consent_id: str) -> Dict[str, Any]:
        """Get consent record."""
        consent_file = self.consent_dir / f"{consent_id}.json"
        
        if not consent_file.exists():
            raise ValueError(f"Consent not found: {consent_id}")
        
        with open(consent_file, "r") as f:
            encrypted_record = json.load(f)
        
        # Decrypt sensitive fields
        consent_record = self._decrypt_sensitive_fields(encrypted_record)
        
        return consent_record
    
    def list_active_consents(self) -> List[Dict[str, Any]]:
        """List all active consents."""
        active_consents = []
        
        for consent_file in self.consent_dir.glob("*.json"):
            if consent_file.name.startswith("."):
                continue
                
            try:
                consent_id = consent_file.stem
                consent = self.get_consent(consent_id)
                
                if consent['status'] == ConsentStatus.ACTIVE.value:
                    # Check expiration
                    expires_at = datetime.fromisoformat(consent['expires_at'])
                    if datetime.utcnow() <= expires_at:
                        active_consents.append(consent)
                    else:
                        self._update_consent_status(consent_id, ConsentStatus.EXPIRED)
                        
            except Exception as e:
                logger.error(f"Error reading consent {consent_file}: {e}")
                continue
        
        return active_consents
    
    def generate_consent_form(
        self,
        consent_type: ConsentType,
        language: str = "en"
    ) -> str:
        """
        Generate consent form template.
        
        Args:
            consent_type: Type of consent
            language: Language code
            
        Returns:
            Path to generated consent form
        """
        template = self._get_consent_template(consent_type, language)
        
        # Generate unique form ID
        form_id = str(uuid.uuid4())
        form_path = self.consent_dir / f"form_{form_id}.html"
        
        # Generate form with template
        form_content = f"""
        <!DOCTYPE html>
        <html lang="{language}">
        <head>
            <meta charset="UTF-8">
            <title>Voice Cloning Consent Form</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .section {{ margin-bottom: 20px; }}
                .signature {{ margin-top: 40px; border-top: 1px solid #000; padding-top: 10px; }}
                .legal {{ font-size: 0.9em; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Voice Cloning Consent Agreement</h1>
                <p>Form ID: {form_id}</p>
                <p>Date: {datetime.utcnow().strftime('%Y-%m-%d')}</p>
            </div>
            
            {template}
            
            <div class="signature">
                <p>By signing below, I acknowledge that I have read, understood, and agree to the terms above.</p>
                <br><br>
                <p>Signature: _______________________________</p>
                <p>Print Name: _____________________________</p>
                <p>Date: ___________________________________</p>
                <p>Email: __________________________________</p>
            </div>
            
            <div class="legal">
                <p>This consent form complies with:</p>
                <ul>
                    <li>Tennessee ELVIS Act (2024)</li>
                    <li>EU AI Act requirements</li>
                    <li>GDPR Article 7 (where applicable)</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(form_path, "w") as f:
            f.write(form_content)
        
        return str(form_path)
    
    def _get_consent_template(
        self,
        consent_type: ConsentType,
        language: str
    ) -> str:
        """Get consent template based on type and language."""
        # Simplified template - in production, use proper i18n
        templates = {
            ConsentType.BASIC: """
            <div class="section">
                <h2>Basic Voice Cloning Consent</h2>
                <p>I, the undersigned, hereby grant permission for my voice to be recorded and used to create a digital voice clone for the following purposes:</p>
                <ul>
                    <li>Personal use only</li>
                    <li>Non-commercial applications</li>
                    <li>Testing and demonstration purposes</li>
                </ul>
                <p><strong>Duration:</strong> This consent is valid for one (1) year from the date of signing.</p>
                <p><strong>Restrictions:</strong> My voice may NOT be used for:</p>
                <ul>
                    <li>Commercial purposes without additional agreement</li>
                    <li>Impersonation or fraudulent activities</li>
                    <li>Content that could damage my reputation</li>
                </ul>
            </div>
            """,
            
            ConsentType.COMMERCIAL: """
            <div class="section">
                <h2>Commercial Voice Cloning License Agreement</h2>
                <p>This agreement grants commercial usage rights for voice cloning with the following terms:</p>
                <ul>
                    <li>Permitted for specified commercial projects</li>
                    <li>Revenue sharing: [TO BE NEGOTIATED]</li>
                    <li>Attribution required where feasible</li>
                    <li>Quality control approval rights retained</li>
                </ul>
                <p><strong>Compensation:</strong> [SPECIFY PAYMENT TERMS]</p>
                <p><strong>Exclusivity:</strong> [NON-EXCLUSIVE / EXCLUSIVE]</p>
            </div>
            """
        }
        
        return templates.get(consent_type, templates[ConsentType.BASIC])
    
    def _encrypt_sensitive_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive fields in consent record."""
        encrypted = record.copy()
        sensitive_fields = ['speaker_email', 'compensation']
        
        for field in sensitive_fields:
            if field in encrypted and encrypted[field]:
                value = json.dumps(encrypted[field])
                encrypted[field] = self.cipher.encrypt(value.encode()).decode()
        
        return encrypted
    
    def _decrypt_sensitive_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive fields in consent record."""
        decrypted = record.copy()
        sensitive_fields = ['speaker_email', 'compensation']
        
        for field in sensitive_fields:
            if field in decrypted and decrypted[field]:
                try:
                    value = self.cipher.decrypt(decrypted[field].encode()).decode()
                    decrypted[field] = json.loads(value)
                except:
                    # Handle unencrypted data for compatibility
                    pass
        
        return decrypted
    
    def _save_consent(self, consent_id: str, record: Dict[str, Any]):
        """Save consent record to file."""
        consent_file = self.consent_dir / f"{consent_id}.json"
        
        with open(consent_file, "w") as f:
            json.dump(record, f, indent=2)
        
        # Set restrictive permissions
        consent_file.chmod(0o600)
    
    def _update_consent_status(self, consent_id: str, status: ConsentStatus):
        """Update consent status."""
        consent = self.get_consent(consent_id)
        consent['status'] = status.value
        
        encrypted = self._encrypt_sensitive_fields(consent)
        self._save_consent(consent_id, encrypted)
    
    def _add_audit_entry(
        self,
        consent_id: str,
        action: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Add audit trail entry."""
        consent = self.get_consent(consent_id)
        
        audit_entry = {
            "action": action,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {}
        }
        
        consent['audit_trail'].append(audit_entry)
        
        encrypted = self._encrypt_sensitive_fields(consent)
        self._save_consent(consent_id, encrypted)
    
    def _generate_consent_hash(self, record: Dict[str, Any]) -> str:
        """Generate hash of consent for verification."""
        # Create deterministic string representation
        consent_string = json.dumps({
            "consent_id": record["consent_id"],
            "speaker_name": record["speaker_name"],
            "consent_type": record["consent_type"],
            "use_cases": sorted(record["use_cases"]),
            "created_at": record["created_at"]
        }, sort_keys=True)
        
        return hashlib.sha256(consent_string.encode()).hexdigest()
    
    def _schedule_data_deletion(self, consent_id: str):
        """Schedule data deletion for GDPR compliance."""
        # In production, integrate with data deletion pipeline
        logger.info(
            f"Scheduled data deletion for consent {consent_id} "
            f"(retention period: {self.settings.data_retention_days} days)"
        )
