"""Crude Russian tokenization shared by retrieval and clustering: 6-letter stems."""
import re

STOP = set('для при как что это или его она они так уже еще также после было были будет может могут более менее году года '
           'рублей млрд млн тыс компании компания россии российских россия число время издели продук товаро'.split())


def stems(text):
    return [w[:6] for w in re.findall(r'[а-яёa-z0-9]{3,}', text.lower()) if w not in STOP and w[:6] not in STOP]
