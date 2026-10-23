"""Bind generated page assets to their PDF; retain superseded paid outputs for inspection."""
import hashlib
import json
from pathlib import Path

from common.news_corpus import write_json


def pdf_hash(pdf: Path) -> str:
    return hashlib.sha256(pdf.read_bytes()).hexdigest()


def current(pdf: Path, folder: Path) -> bool:
    manifest = folder / f'{pdf.stem}.source.json'
    # Existing team assets predate manifests; keep their compatibility without inventing provenance.
    return not manifest.exists() or json.loads(manifest.read_text(encoding='utf-8')).get('pdf_sha256') == pdf_hash(pdf)


def prepare(pdf: Path, folder: Path, recipe: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / f'{pdf.stem}.source.json'
    source = {'pdf_sha256': pdf_hash(pdf), 'recipe': recipe}
    old = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else None
    comparable = {k: v for k, v in (old or {}).items() if k != 'legacy_unverified'}
    changed = old is not None and comparable != source
    if changed:
        version = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()[:16]
        archive = folder / 'previous' / f'{pdf.stem}-{version}'
        archive.mkdir(parents=True, exist_ok=True)
        for path in folder.glob(f'{pdf.stem}_p*'):
            if path.is_file():
                path.replace(archive / path.name)
    source['legacy_unverified'] = bool(not changed and (old.get('legacy_unverified', False) if old else
        any(folder.glob(f'{pdf.stem}_p*'))))
    write_json(manifest, source)
