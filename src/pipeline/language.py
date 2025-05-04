from langdetect import detect

def detect_language(text):
    """
    Detect the language of a given text using langdetect.
    """
    try:
        return detect(text)
    except:
        return "unknown"
