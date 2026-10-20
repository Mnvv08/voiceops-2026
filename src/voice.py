"""Voice layer: Gnani STT -> agent -> Gnani TTS. Gnani is the ONLY speech service used."""
import re

from .config import GNANI_LANG, GNANI_TTS_MODEL, GNANI_VOICE

FALLBACK = "Sorry, I didn't catch that. Could you say it again?"


def for_the_ear(text: str) -> str:
    """Make text easier to speak: strip markdown, space out long digit strings (order IDs, phones)."""
    text = re.sub(r"[*_#`>]", "", text)
    return re.sub(r"\d{5,}", lambda m: " ".join(m.group(0)), text)


class Voice:
    def __init__(self):
        from gnani.stt import GnaniSTTClient   # imported lazily so text-only runs need no SDK
        from gnani.tts import GnaniTTSClient
        self.stt = GnaniSTTClient()            # reads GNANI_API_KEY from the environment
        self.tts = GnaniTTSClient()

    def turn(self, audio_path: str, agent, out_wav: str, language: str = GNANI_LANG):
        heard = self.stt.transcribe(audio_path, language_code=language)["transcript"]
        reply = FALLBACK if not heard.strip() else agent.respond(heard)
        self.tts.synthesize(for_the_ear(reply), voice=GNANI_VOICE, model=GNANI_TTS_MODEL,
                            language=language, output_file=out_wav)
        return heard, reply
