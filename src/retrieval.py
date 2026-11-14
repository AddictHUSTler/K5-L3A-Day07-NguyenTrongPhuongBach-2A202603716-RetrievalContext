"""Day 7 retrieval: original Day 6 dense index, TF-IDF, and RRF hybrid."""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from starter.baseline import TfidfIndex

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


class QueryEmbeddings:
    """Cache by exact input/model/dimensions. Cache never substitutes document vectors."""

    def __init__(self, env_file=ROOT / '.env', cache_dir=ROOT / '.local/query_embeddings'):
        self.spec = json.loads((ROOT / 'artifacts/embedding_model.json').read_text())
        self.env_file = env_file
        self.cache_dir = cache_dir
        self.records = []

    def embed(self, texts):
        keys = [hashlib.sha256(json.dumps([self.spec['name'], self.spec['dimensions'], t],
                                         ensure_ascii=False).encode()).hexdigest() for t in texts]
        cached = {}
        missing = []
        for key, text in zip(keys, texts):
            path = self.cache_dir / f'{key}.json'
            if path.exists():
                row = json.loads(path.read_text())
                if (row['model'], row['dimensions'], row['text']) != (
                        self.spec['name'], self.spec['dimensions'], text):
                    raise ValueError('Query embedding cache metadata mismatch')
                cached[key] = row['vector']
            elif key not in [k for k, _ in missing]:
                missing.append((key, text))
        record = {'model': self.spec['name'], 'dimensions': self.spec['dimensions'],
                  'inputs': len(texts), 'cache_hits': len(texts) - len(missing),
                  'api_requests': 0, 'new_tokens': 0, 'estimated_new_cost_usd': 0,
                  'price_usd_per_million_tokens': self.spec['price_usd_per_million_tokens'],
                  'price_source': 'artifacts/embedding_model.json'}
        started = time.perf_counter()
        if missing:
            key = os.environ.get('OPENAI_API_KEY')
            if not key and self.env_file.exists():
                from dotenv import dotenv_values
                key = dotenv_values(self.env_file).get('OPENAI_API_KEY')
            if not key:
                raise ValueError('Set OPENAI_API_KEY in .env or the environment; dense search needs real query embeddings')
            request = Request(self.spec['endpoint'], data=json.dumps({
                'model': self.spec['name'], 'dimensions': self.spec['dimensions'],
                'encoding_format': 'float', 'input': [t for _, t in missing]}).encode(),
                headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
            try:
                with urlopen(request, timeout=90) as response:
                    result = json.load(response)
                    request_id = response.headers.get('x-request-id')
            except HTTPError as exc:
                raise RuntimeError(f'Embedding API HTTP {exc.code}; no fallback') from None
            except (URLError, TimeoutError):
                raise RuntimeError('Embedding API connection failed; no fallback') from None
            rows = sorted(result['data'], key=lambda r: r['index'])
            if result['model'] != self.spec['name'] or [r['index'] for r in rows] != list(range(len(missing))):
                raise ValueError('Embedding API returned unexpected model or indices')
            for (cache_key, text), row in zip(missing, rows):
                vector = row['embedding']
                self._validate(vector)
                cached[cache_key] = vector
                write_json(self.cache_dir / f'{cache_key}.json', {
                    'model': result['model'], 'dimensions': self.spec['dimensions'],
                    'text': text, 'vector': vector, 'request_id': request_id})
            record.update(api_requests=1, new_tokens=result['usage']['total_tokens'],
                          estimated_new_cost_usd=result['usage']['total_tokens'] *
                          self.spec['price_usd_per_million_tokens'] / 1_000_000,
                          request_id=request_id, usage=result['usage'])
        record['elapsed_ms'] = (time.perf_counter() - started) * 1000
        self.records.append(record)
        if record['api_requests']:
            log = ROOT / 'results/embedding_requests.jsonl'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record) + '\n')
        vectors = [cached[k] for k in keys]
        for vector in vectors:
            self._validate(vector)
        return vectors

    def _validate(self, vector):
        if len(vector) != self.spec['dimensions'] or not all(math.isfinite(x) for x in vector):
            raise ValueError('Invalid query embedding dimensions or values')


def validate_filter(filters):
    # Only explicit user-provided chapter constraints are supported in this corpus.
    if filters and (set(filters) != {'chapter'} or type(filters['chapter']) is not int
                    or filters['chapter'] not in (3, 4, 5)):
        raise ValueError('Only chapter filters 3, 4, or 5 are supported')


def fuse(dense, lexical, top_k, rrf_k=60):
    scores, chunks, components = Counter(), {}, {}
    for method, results in [('dense', dense), ('tfidf', lexical)]:
        for rank, chunk in enumerate(results, 1):
            cid = chunk['id']
            scores[cid] += 1 / (rrf_k + rank)
            chunks[cid] = chunk
            components.setdefault(cid, {})[method] = {'rank': rank, 'score': chunk['score']}
    ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))[:top_k]
    return [{**chunks[cid], 'score': scores[cid], 'strategy': 'hybrid',
             'rank': rank, 'components': components[cid]} for rank, cid in enumerate(ordered, 1)]


class Retriever:
    def __init__(self, chunks, store_path=ROOT / '.local/qdrant'):
        self.chunks = chunks
        self.lexical = TfidfIndex(chunks)
        self.store_path = store_path
        self.client = None
        self.last_stages = None

    def close(self):
        if self.client is not None:
            self.client.close()

    def search(self, question, method, top_k=3, filters=None, vector=None,
               candidates=20, rrf_k=60):
        validate_filter(filters)
        if top_k < 1 or candidates < top_k or rrf_k < 1:
            raise ValueError('Require 1 <= top_k <= candidates and rrf_k >= 1')
        if method == 'tfidf':
            # Rank all lexical matches, filter, then cut off; no post-top-k filtering.
            rows = self.lexical.search(question, top_k=len(self.chunks))
            rows = [r for r in rows if not filters or r['metadata']['chapter'] == filters['chapter']]
            return [{**r, 'rank': i} for i, r in enumerate(rows[:top_k], 1)]
        if method == 'dense':
            if vector is None:
                raise ValueError('Dense search requires a real query embedding')
            from qdrant_client import QdrantClient, models
            if self.client is None:
                self.client = QdrantClient(path=str(self.store_path))
            query_filter = None
            if filters:
                query_filter = models.Filter(must=[models.FieldCondition(
                    key='chapter', match=models.MatchValue(value=filters['chapter']))])
            points = self.client.query_points('python_C', query=vector, limit=top_k,
                                              query_filter=query_filter, with_payload=True).points
            rows = []
            for rank, point in enumerate(points, 1):
                p = point.payload
                rows.append({'id': p['chunk_id'], 'source_id': p['document_id'],
                             'source': p['source'], 'page': p['page'], 'text': p['text'],
                             'metadata': {k: v for k, v in p.items() if k not in
                                          ('chunk_id', 'document_id', 'source', 'page', 'text')},
                             'score': point.score, 'rank': rank, 'strategy': 'dense'})
            return rows
        if method == 'hybrid':
            dense = self.search(question, 'dense', candidates, filters, vector, candidates)
            lexical = self.search(question, 'tfidf', candidates, filters, candidates=candidates)
            self.last_stages = {'dense': dense, 'tfidf': lexical}
            return fuse(dense, lexical, top_k, rrf_k)
        raise ValueError(f'Unknown retrieval method: {method}')
