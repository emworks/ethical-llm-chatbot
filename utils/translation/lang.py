import locale
from . import messages

DEFAULT_LANG = "ru"
CURRENT_LANG = DEFAULT_LANG
SUPPORTED_LANGS = {"ru", "en", "de", "fr"}


def set_lang(lang: str):
    global CURRENT_LANG
    if lang in SUPPORTED_LANGS:
        CURRENT_LANG = lang
    else:
        CURRENT_LANG = DEFAULT_LANG


def get_current_lang():
    return CURRENT_LANG


def detect_system_lang():
    loc = locale.getdefaultlocale()[0]
    if not loc:
        return "en"
    return loc.split("_")[0]


def t(key: str, **kwargs):
    lang = get_current_lang()
    template = messages.MESSAGES[key][lang]
    
    if not kwargs:
        return template
    
    return template.format(**kwargs)
