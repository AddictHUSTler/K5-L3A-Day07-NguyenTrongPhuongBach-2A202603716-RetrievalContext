"""Extract the complete tutorial into a separate, page-addressable corpus."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unicodedata
import uuid
import xml.etree.ElementTree as ET

from src.retrieval import ROOT, QueryEmbeddings, write_json

OUT = ROOT / 'data/bonus'
STORE = ROOT / '.local/bonus_qdrant'


def extract(pdf):
    layout = subprocess.check_output(['pdftotext', '-layout', str(pdf), '-'], text=True)
    xml = subprocess.check_output(['pdftotext', '-bbox-layout', str(pdf), '-'])
    tree = ET.fromstring(xml)
    ns = {'p': 'http://www.w3.org/1999/xhtml'}
    geometry = tree.findall('.//p:page', ns)
    raw_pages = layout.split('\f')
    if not raw_pages[-1].strip():
        raw_pages.pop()
    if len(raw_pages) != len(geometry):
        raise ValueError('Text/geometry page counts disagree')
    pages, chunks = [], []
    for number, (raw, geo) in enumerate(zip(raw_pages, geometry), 1):
        # Retain front matter, appendices, tables and code; exclude only running furniture.
        lines = [line.rstrip() for line in unicodedata.normalize('NFC', raw).splitlines()]
        nonempty = [i for i, line in enumerate(lines) if line.strip()]
        for i in nonempty[:1] + nonempty[-1:]:
            if (re.fullmatch(r'\s*(?:\d+|[ivxlcdm]+)\s*', lines[i])
                    or 'Python Tutorial, Release 3.12.11' in lines[i]
                    or re.match(r'^\s*(?:\d+\s+)?(?:Chapter \d+\.|Appendix [A-Z]\.|CONTENTS\s*$|\d+\.\d+\. )', lines[i])):
                lines[i] = ''
        text = '\n'.join(lines).strip()
        blocks = []
        for block in geo.findall('.//p:block', ns):
            blocks.append({'bbox': [float(block.attrib[k]) for k in ('xMin', 'yMin', 'xMax', 'yMax')],
                           'text': '\n'.join(' '.join(w.text or '' for w in line.findall('p:word', ns))
                                             for line in block.findall('p:line', ns))})
        pages.append({'page': number, 'text': text, 'width': float(geo.attrib['width']),
                      'height': float(geo.attrib['height']), 'blocks': blocks})
        # Whole pages preserve table headers/rows and code indentation together.
        if text:
            chunks.append({'id': hashlib.sha256(f'tutorial-full-v1:{number}:{text}'.encode()).hexdigest(),
                'source_id': 'python-tutorial-3.12.11-full', 'source': 'data/tutorial.pdf',
                'page': number, 'text': text,
                'metadata': {'scope': 'full-tutorial', 'region': 'page body',
                             'page_url': f'data/tutorial.pdf#page={number}',
                             'geometry_file': 'data/bonus/pages.json', 'geometry_page': number}})
    return pages, chunks


def prepare():
    pdf = ROOT / 'data/tutorial.pdf'
    pages, chunks = extract(pdf)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'pages.json', pages)
    write_json(OUT / 'chunks.json', chunks)
    images = subprocess.check_output(['pdfimages', '-list', str(pdf)], text=True)
    manifest = {'source': 'data/tutorial.pdf', 'source_id': 'python-tutorial-3.12.11-full',
        'sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(), 'version': '3.12.11',
        'snapshot_date': '2025-09-29', 'license': 'PSF-2.0; example code also 0BSD',
        'license_file': 'data/LICENSE-PYTHON.txt', 'physical_pages': len(pages),
        'indexed_pages': len(chunks), 'chunks': len(chunks),
        'excluded_pages': [p['page'] for p in pages if not p['text']],
        'chunking': 'One complete nonempty page per chunk; layout whitespace preserved.',
        'extraction': 'Poppler pdftotext -layout and -bbox-layout; no OCR or generated content.',
        'scope': 'All pages, including front matter, appendices, glossary, license and index.',
        'visual_support': 'Text tables and ASCII diagrams retain their layout; block coordinates saved.',
        'embedded_images': max(0, len(images.strip().splitlines()) - 2),
        'bonus_status': 'Supported subset only; required chart/image benchmark coverage is missing.',
        'basic_index_unchanged': True}
    write_json(OUT / 'manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('physical_pages', 'indexed_pages', 'chunks', 'excluded_pages')}))


def index():
    from qdrant_client import QdrantClient, models
    chunks = json.loads((OUT / 'chunks.json').read_text())
    embedder = QueryEmbeddings(cache_dir=ROOT / '.local/bonus_embeddings')
    client = QdrantClient(path=str(STORE))
    try:
        existing = {}
        if client.collection_exists('python_C'):
            rows, offset = client.scroll('python_C', limit=len(chunks) + 1, with_payload=True)
            if offset is not None:
                raise ValueError('Unexpected extra index points')
            existing = {p.payload['chunk_id']: p.payload for p in rows}
            expected = {c['id'] for c in chunks}
            if set(existing) - expected:
                raise ValueError('Existing index belongs to another corpus')
        else:
            client.create_collection('python_C', vectors_config=models.VectorParams(
                size=embedder.spec['dimensions'], distance=models.Distance.COSINE))
        for start in range(0, len(chunks), 24):
            batch = [c for c in chunks[start:start + 24] if c['id'] not in existing]
            if not batch:
                continue
            vectors = embedder.embed([c['text'] for c in batch])
            client.upsert('python_C', points=[models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c['id'])), vector=v,
                payload={'chunk_id': c['id'], 'document_id': c['source_id'],
                         'text': c['text'], 'source': c['source'], 'page': c['page'], **c['metadata']})
                for c, v in zip(batch, vectors)])
        rows, offset = client.scroll('python_C', limit=len(chunks) + 1,
                                     with_payload=True, with_vectors=True)
        if offset is not None or len(rows) != len(chunks):
            raise ValueError('Incomplete full-document index')
        expected = {c['id']: c for c in chunks}
        for point in rows:
            c = expected[point.payload['chunk_id']]
            if any(point.payload[k] != c[k] for k in ('text', 'page', 'source')):
                raise ValueError('Stored payload mismatch')
            embedder._validate(point.vector)
        old = ROOT / 'results/bonus/index.json'
        previous = json.loads(old.read_text()) if old.exists() else {'embedding_usage': []}
        write_json(old, {'store': '.local/bonus_qdrant', 'collection': 'python_C',
            'points_verified': len(rows), 'model': embedder.spec['name'],
            'dimensions': embedder.spec['dimensions'], 'distance': 'Cosine',
            'embedding_usage': previous['embedding_usage'] + embedder.records})
        print(f'Full tutorial: {len(rows)} real vectors verified.')
    finally:
        client.close()


def crop_visuals():
    regions = json.loads((OUT / 'regions.json').read_text())
    for region in regions:
        scale = region['dpi'] / 72
        x0, y0, x1, y1 = region['bbox_points']
        rectangle = [round(value * scale) for value in (x0, y0, x1 - x0, y1 - y0)]
        destination = ROOT / region['crop']
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['pdftoppm', '-f', str(region['page']), '-l', str(region['page']),
                        '-r', str(region['dpi']), '-x', str(rectangle[0]), '-y', str(rectangle[1]),
                        '-W', str(rectangle[2]), '-H', str(rectangle[3]), '-singlefile', '-png',
                        str(ROOT / region['source']), str(destination.with_suffix(''))], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', action='store_true', help='Also create real embeddings and a separate Qdrant store')
    parser.add_argument('--visuals', action='store_true', help='Recreate PDF crops from saved region coordinates')
    parser.add_argument('--run-extension', action='store_true', help='Run the frozen pipeline on saved extension queries')
    args = parser.parse_args()
    if args.visuals:
        crop_visuals()
    elif not args.run_extension:
        prepare()
    if args.index:
        index()
    if args.run_extension:
        from src import agent, bonus
        frozen = json.loads((ROOT / 'artifacts/bonus_selected.json').read_text())
        if frozen['config'] != bonus.CONFIG or frozen['hashes'] != bonus.hashes():
            raise ValueError('Frozen pipeline changed; select on dev before a fresh test')
        agent.Model()
        bonus.run('extension')
