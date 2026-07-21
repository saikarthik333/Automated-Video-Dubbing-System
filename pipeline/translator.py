"""Step 2: translate transcribed segments into natural English.

Uses Meta's NLLB-200 (distilled-600M), a free, open-source model covering
200 languages, so the same code path handles German, French, Hindi, or
anything else Whisper detects -- no per-language branching needed.

It's a full sentence-to-sentence translator (not a dictionary lookup), so it
naturally produces idiomatic phrasing rather than a literal word-for-word
conversion.

Swap-in note: for Indian-language sources specifically, AI4Bharat's
IndicTrans2 is purpose-built and tends to outperform NLLB. To use it instead,
implement a `translate_indictrans2()` function with the same signature
(list[Segment] -> list[str]) and select it in main.py when the detected
language is one of IndicTrans2's supported Indic languages.
"""

from typing import List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .transcriber import Segment
from .utils import log

# Whisper's ISO 639-1 code -> NLLB's FLORES-200 code (most common languages;
# extend as needed -- NLLB supports ~200).
WHISPER_TO_NLLB = {
    "en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn", "es": "spa_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "ru": "rus_Cyrl", "zh": "zho_Hans",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "ar": "arb_Arab", "nl": "nld_Latn",
    "hi": "hin_Deva", "bn": "ben_Beng", "ta": "tam_Taml", "te": "tel_Telu",
    "mr": "mar_Deva", "gu": "guj_Gujr", "kn": "kan_Knda", "ml": "mal_Mlym",
    "pa": "pan_Guru", "ur": "urd_Arab", "tr": "tur_Latn", "vi": "vie_Latn",
    "th": "tha_Thai", "id": "ind_Latn", "pl": "pol_Latn", "uk": "ukr_Cyrl",
    "fa": "pes_Arab",
}

MODEL_NAME = "facebook/nllb-200-distilled-600M"


def translate_segments(segments: List[Segment], source_lang: str, device: str = "cuda") -> List[str]:
    """Translate each segment's text to English. Returns list of English strings
    in the same order as `segments`.
    """
    if source_lang == "en":
        log("translate", "Source is already English, skipping translation")
        return [seg.text for seg in segments]

    src_code = WHISPER_TO_NLLB.get(source_lang)
    if src_code is None:
        raise ValueError(
            f"No NLLB language code mapped for Whisper language '{source_lang}'. "
            f"Add it to WHISPER_TO_NLLB in translator.py."
        )

    log("translate", f"Loading NLLB-200 ({MODEL_NAME}) for {source_lang} -> en")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=src_code)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    try:
        model = model.to(device)
    except Exception:
        device = "cpu"
        model = model.to(device)

    eng_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    translations: List[str] = []

    for i, seg in enumerate(segments):
        inputs = tokenizer(seg.text, return_tensors="pt").to(device)
        generated = model.generate(
            **inputs,
            forced_bos_token_id=eng_token_id,
            max_length=256,
            num_beams=4,
        )
        english = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        translations.append(english)
        log("translate", f"({i+1}/{len(segments)}) {seg.text!r} -> {english!r}")

    return translations
