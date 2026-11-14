"""GUIDE part 2: compare retrieval on dev without exposing gold to retrievers."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

from starter.baseline import hit, pack_context
from src.retrieval import ROOT, QueryEmbeddings, Retriever, write_json


def metric(n, total):
    return {'n': n, 'N': total, 'rate': n / total if total else None}


def summarize(traces, labels):
    answerable = [q for q in labels if q['answerable']]
    by_id = {t['query_id']: t for t in traces}
    grades = []
    for q in answerable:
        trace = by_id[q['id']]
        rows = trace['results']
        grades.append({'query_id': q['id'],
                       'hit1': any(hit(r, q['gold']) for r in rows[:1]),
                       'hit3': any(hit(r, q['gold']) for r in rows[:3]),
                       'context_hit': any(hit(r, q['gold']) and r['id'] in trace['context_ids'] for r in rows)})
    return {'queries': len(traces), 'answerable': len(answerable),
            'errors': sum(t['status'] == 'error' for t in traces),
            **{name: metric(sum(g[name] for g in grades), len(answerable))
               for name in ('hit1', 'hit3', 'context_hit')},
            'median_search_ms': statistics.median(t['search_ms'] for t in traces),
            'total_search_ms': sum(t['search_ms'] for t in traces),
            'avg_context_chars': statistics.mean(len(t['context']) for t in traces),
            'new_retrieval_service_cost_usd': 0}, grades


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', choices=['dense', 'tfidf', 'hybrid'],
                        default=['dense', 'tfidf', 'hybrid'])
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/retrieval-dev')
    args = parser.parse_args()
    queries = json.loads((ROOT / 'data/queries_dev.json').read_text())
    # Runtime query files must contain no labels or evidence.
    if any(set(q) - {'id', 'question', 'filter'} for q in queries):
        raise ValueError('Runtime queries contain unexpected fields')
    chunks = json.loads((ROOT / 'data/chunks.json').read_text())
    config = {'split': 'dev', 'top_k': 3, 'candidates_per_retriever': 20, 'rrf_k': 60,
              'budget_chars': 6000, 'context_policy': 'ranked whole chunks, deduplicate by ID; skip blocks that do not fit',
              'query_processing': 'original question, no rewrite or inferred filter',
              'lexical_tokenizer': 'Unicode NFC, casefold, regex word tokens; no Vietnamese segmentation',
              'filter_policy': 'explicit chapter from runtime query, applied before cutoff',
              'collection': 'python_C', 'index_rebuilt': False,
              'corpus_sha256': hashlib.sha256((ROOT / 'data/chunks.json').read_bytes()).hexdigest(),
              'queries_sha256': hashlib.sha256((ROOT / 'data/queries_dev.json').read_bytes()).hexdigest(),
              'started_at': datetime.now(timezone.utc).isoformat()}
    write_json(args.output / 'config.json', config)
    embedder = QueryEmbeddings(env_file=args.env_file)
    vectors, embedding_error = [None] * len(queries), None
    if any(m in args.methods for m in ['dense', 'hybrid']):
        try:
            vectors = embedder.embed([q['question'] for q in queries])
        except (ValueError, RuntimeError) as exc:
            embedding_error = str(exc)
    write_json(args.output / 'embedding_usage.json', {
        'shared_by': [m for m in args.methods if m in ('dense', 'hybrid')],
        'records': embedder.records, 'error': embedding_error,
        'note': 'Query embedding preparation is shared once, excluded from search timing; zero new cost on cache reuse.'})
    initialized = time.perf_counter()
    retriever = Retriever(chunks)
    config['lexical_index_build_ms'] = (time.perf_counter() - initialized) * 1000
    config['timing_note'] = 'Search excludes embedding and lexical index build; first dense query includes local Qdrant opening.'
    write_json(args.output / 'config.json', config)
    summaries = {}
    try:
        for method in args.methods:
            traces = []
            for query, vector in zip(queries, vectors):
                retriever.last_stages = None
                started = time.perf_counter()
                rows, error = [], None
                try:
                    if method != 'tfidf' and embedding_error:
                        raise RuntimeError(embedding_error)
                    rows = retriever.search(query['question'], method, filters=query.get('filter'), vector=vector)
                except Exception as exc:
                    # Store controlled messages only; third-party errors may include credentials.
                    error = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__
                search_ms = (time.perf_counter() - started) * 1000
                context, ids = pack_context(rows, config['budget_chars'])
                traces.append({'query_id': query['id'], 'question': query['question'],
                               'method': method, 'filter': query.get('filter'),
                               'filter_origin': 'explicit user query' if query.get('filter') else None,
                               'top_k': 3, 'search_ms': search_ms, 'status': 'error' if error else 'ok',
                               'error': error, 'results': rows, 'context': context, 'context_ids': ids,
                               'candidate_stages': retriever.last_stages,
                               'budget_chars': config['budget_chars']})
            write_json(args.output / method / 'traces.json', traces)
            # Gold is loaded only in the evaluator after retrieval has finished.
            labels = json.loads((ROOT / 'benchmark/dev.json').read_text())
            if {q['id'] for q in labels} != {q['id'] for q in queries}:
                raise ValueError('Dev labels and runtime query IDs do not match')
            summary, grades = summarize(traces, labels)
            write_json(args.output / method / 'grades.json', grades)
            summaries[method] = summary
    finally:
        retriever.close()
    write_json(args.output / 'summary.json', summaries)
    with (args.output / 'attempts.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'started_at': config['started_at'],
                                 'methods': args.methods, 'embedding_error': embedding_error,
                                 'errors_by_method': {m: s['errors'] for m, s in summaries.items()}}) + '\n')
    print(json.dumps(summaries, indent=2))
    if any(s['errors'] for s in summaries.values()):
        raise SystemExit('Incomplete benchmark: see error traces; no fallback results were used.')


if __name__ == '__main__':
    main()
