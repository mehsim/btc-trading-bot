import os
import base64
from typing import Optional

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False


def _get_fernet_key(salt: bytes = b"btc_bot_static_salt_v1") -> bytes:
    master_key = os.environ.get("MASTER_ENCRYPTION_KEY", "default_antigravity_key_2026").encode()
    if CRYPTOGRAPHY_AVAILABLE:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(master_key))
    else:
        return base64.urlsafe_b64encode(master_key.ljust(32)[:32])


def decrypt_secret(secret_val: Optional[str]) -> str:
    """Decrypts AES-256 encrypted environment secret strings prefixed with 'enc:'."""
    if not secret_val:
        return ""
    if not secret_val.startswith("enc:"):
        return secret_val
    
    cipher_text = secret_val[4:]
    if not CRYPTOGRAPHY_AVAILABLE:
        print("[SecretManager Warning] cryptography package not installed. Returning raw ciphertext.")
        return cipher_text
        
    try:
        key = _get_fernet_key()
        f = Fernet(key)
        decrypted = f.decrypt(cipher_text.encode()).decode()
        return decrypted
    except Exception as e:
        print(f"[SecretManager Error] Failed to decrypt secret: {e}")
        return cipher_text


def get_secure_env(key: str, default: str = "") -> str:
    """Fetches and decrypts an environment variable if encrypted."""
    val = os.environ.get(key, default)
    return decrypt_secret(val)
