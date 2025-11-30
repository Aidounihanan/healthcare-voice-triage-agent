# stt_elevenlabs_client.py
import os
from typing import Optional

from elevenlabs import ElevenLabs  # pip install elevenlabs

# On supporte les deux noms d'env possibles
_API_KEY = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_API_KEY")
if not _API_KEY:
    raise RuntimeError("Missing ELEVENLABS_API_KEY or ELEVEN_API_KEY in environment.")

# Client ElevenLabs officiel
client = ElevenLabs(api_key=_API_KEY)

# Modèle STT par défaut : Scribe v1 (recommandé pour transcription offline)
DEFAULT_STT_MODEL = os.getenv("ELEVEN_STT_MODEL", "scribe_v1")

# Map simple pour language_code (ElevenLabs utilise souvent des codes à 3 lettres, ex: 'eng')
LANGUAGE_MAP = {
    "en": "eng",
    "fr": "fra",
    "es": "spa",
    "ar": "ara",
}


def transcribe_audio_file(file_path: str, language: str = "en") -> Optional[str]:
    """
    Transcrit un fichier audio en texte avec ElevenLabs Speech-to-Text (Scribe v1).
    - file_path : chemin local vers le fichier audio (.wav, .mp3, etc.)
    - language  : code 2 lettres (en, fr, ...), converti en code 3 lettres pour ElevenLabs.

    Retourne la transcription texte, ou None si échec.
    """
    lang_code = LANGUAGE_MAP.get(language, None)  # tu peux laisser None = auto-detect

    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    # Appel SDK ElevenLabs
    transcription = client.speech_to_text.convert(
        file=audio_bytes,
        model_id=DEFAULT_STT_MODEL,
        language_code=lang_code,
        diarize=False,
        tag_audio_events=False,
    )

    # Selon la doc, transcription.text contient le texte principal
    try:
        text = transcription.text
    except AttributeError:
        # Debug si la réponse est différente
        print("[Eleven STT] Unexpected transcription object:", transcription)
        return None

    return text
