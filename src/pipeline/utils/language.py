from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # ensures consistent results

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"  # fallback

def is_english(text: str) -> bool:
    return detect_language(text) == "en"

def is_spanish(text: str) -> bool:
    return detect_language(text) == "es"
