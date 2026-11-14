"""Run and review the full-tutorial supported subset independently of the basic lab."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import time

from src import agent
from src.prepare_bonus import OUT, STORE
from src.retrieval import ROOT, QueryEmbeddings, Retriever, write_json
from starter.baseline import pack_context

RESULTS = ROOT / 'results/bonus'
CONFIG = dict(method='tfidf', top_k=6, max_searches=2, budget=24000, dedup_text=True)
PROMPT = agent.SYSTEM.replace('chapters 3–5.', 'complete document, all chapters and appendices.')
PROMPT += '''
Check each citation against the actual evidence page, never a table of contents or
an unrelated page. For an answer combining pages, cite every necessary page next
to the claims it supports. If the first search has no evidence for a requested
fact, use the remaining search for an English query with the relevant Python API
terms before abstaining. Do not mistake a glossary entry for the requested example.
'''


def read(path):
    return json.loads(path.read_text())


def hashes():
    paths = ['src/bonus.py', 'src/agent.py', 'src/retrieval.py', 'starter/baseline.py',
             'data/bonus/chunks.json', 'data/bonus/queries_dev.json', 'data/bonus/queries_test.json']
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def supported(evidence, rows):
    return any(c['source_id'] == evidence['source_id'] and c['page'] == evidence['page']
               and evidence['evidence'] in c['text'] for c in rows)


def retrieval_dev():
    output = RESULTS / 'retrieval-dev'
    if output.exists():
        raise ValueError('Retrieval dev results already exist')
    chunks = read(OUT / 'chunks.json')
    queries = read(OUT / 'queries_dev.json')
    gold = {c['id']: c for c in read(ROOT / 'benchmark/bonus/dev.json')}
    embedder = QueryEmbeddings(cache_dir=ROOT / '.local/bonus_query_embeddings')
    vectors = embedder.embed([q['question'] for q in queries])
    retriever = Retriever(chunks, store_path=STORE)
    try:
        summary = {}
        for method in ('dense', 'tfidf', 'hybrid'):
            traces = []
            for query, vector in zip(queries, vectors):
                start = time.perf_counter()
                rows = retriever.search(query['question'], method, top_k=6, vector=vector)
                context, ids = pack_context(rows, CONFIG['budget'])
                case = gold[query['id']]
                trace = dict(query_id=query['id'], results=rows, context=context, context_ids=ids,
                             elapsed_ms=(time.perf_counter() - start) * 1000)
                if case['answerable']:
                    for k in (1, 3, 6):
                        # Complete-evidence metric requires every multi-hop page, even at top-1.
                        trace[f'complete_hit{k}'] = all(supported(g, rows[:k]) for g in case['gold'])
                    trace['context_complete'] = all(supported(g, [r for r in rows if r['id'] in ids])
                                                     and g['evidence'] in context for g in case['gold'])
                traces.append(trace)
            answerable = [t for t in traces if 'context_complete' in t]
            summary[method] = {key: {'n': sum(t[key] for t in answerable), 'N': len(answerable)}
                for key in ('complete_hit1', 'complete_hit3', 'complete_hit6', 'context_complete')}
            summary[method]['search_elapsed_ms'] = sum(t['elapsed_ms'] for t in traces)
            write_json(output / method / 'traces.json', traces)
        write_json(output / 'embedding_usage.json', embedder.records)
        write_json(output / 'summary.json', summary)
        print(json.dumps(summary))
    finally:
        retriever.close()


def run(split):
    output = RESULTS / f'agent-{split}'
    if output.exists():
        raise ValueError(f'Results already exist: {output}')
    if split == 'test':
        frozen = read(ROOT / 'artifacts/bonus_selected.json')
        if frozen['config'] != CONFIG or frozen['hashes'] != hashes():
            raise ValueError('Configuration/code/corpus changed after freeze')
    queries = read(OUT / f'queries_{split}.json')
    if any(set(q) != {'id', 'question'} for q in queries):
        raise ValueError('Unexpected runtime input fields')
    agent.SYSTEM = PROMPT
    embedder = QueryEmbeddings(cache_dir=ROOT / '.local/bonus_query_embeddings')
    retriever = Retriever(read(OUT / 'chunks.json'), store_path=STORE)
    write_json(output / 'config.json', {'config': CONFIG, 'model': agent.MODEL,
        'system_prompt': PROMPT, 'hashes': hashes(), 'started_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'supported subset; no complete advanced bonus score'})
    traces = []
    try:
        model = agent.Model()
        for query in queries:
            trace = agent.run(query, CONFIG, model, retriever, embedder)
            traces.append(trace)
            write_json(output / 'traces.json', traces)
            write_json(output / 'embedding_usage.json', embedder.records)
            print(query['id'], trace['status'], trace['error'], flush=True)
    finally:
        retriever.close()


def freeze():
    traces = read(RESULTS / 'agent-dev/traces.json')
    queries = read(OUT / 'queries_dev.json')
    reviews = read(RESULTS / 'agent-dev/reviews.json')
    if ({t['query_id'] for t in traces} != {q['id'] for q in queries}
            or {r['query_id'] for r in reviews} != {q['id'] for q in queries}):
        raise ValueError('Complete dev run and explicit reviews required before freeze')
    destination = ROOT / 'artifacts/bonus_selected.json'
    if destination.exists():
        raise ValueError('Freeze already exists')
    write_json(destination, {'frozen_at': datetime.now(timezone.utc).isoformat(),
        'config': CONFIG, 'hashes': hashes(), 'model': agent.MODEL,
        'dev_errors': sum(bool(t['error']) for t in traces),
        'reason': 'TF-IDF top-6 retrieved complete evidence for 14/15 dev cases, versus hybrid 13/15 and dense 11/15. Full pages retain table headers/rows; two searches permit missing-page follow-up.',
        'scope': 'Supported subset, not a qualifying advanced benchmark.'})


def evaluate(split):
    output = RESULTS / f'agent-{split}'
    cases = read(ROOT / f'benchmark/bonus/{split}.json')
    traces = {t['query_id']: t for t in read(output / 'traces.json')}
    reviews = {r['query_id']: r for r in read(output / 'reviews.json')}
    expected = {c['id'] for c in cases}
    if set(traces) != expected or set(reviews) != expected:
        raise ValueError('Every case needs a trace and explicit review')
    groups = {}
    errors = 0
    for case in cases:
        t, r = traces[case['id']], reviews[case['id']]
        if type(r['passed']) is not bool or not r.get('reason'):
            raise ValueError('Review needs boolean passed and reason')
        if r['passed'] and (t['error'] or (case['answerable'] and t['status'] != 'answered')
                            or (not case['answerable'] and t['status'] != 'abstain')):
            raise ValueError('Review conflicts with trace status')
        g = groups.setdefault(case['group'], {'n': 0, 'N': 0})
        g['n'] += r['passed']
        g['N'] += 1
        errors += bool(t['error'])
    summary = {'scope': 'supported subset only; not an ADVANCED.md bonus pass rate', 'groups': groups,
        'passed': sum(g['n'] for g in groups.values()), 'total': len(cases), 'errors': errors,
        'elapsed_ms': sum(t['elapsed_ms'] for t in traces.values()),
        'model_cost_usd': sum(t['estimated_model_cost_usd'] for t in traces.values()),
        'embedding_cost_usd': sum(r['estimated_new_cost_usd'] for r in read(output / 'embedding_usage.json')),
        'search_count_distribution': dict(Counter(sum(e['event'] == 'search' for e in t['events']) for t in traces.values())),
        'missing_test_groups': {'image_diagram': 5, 'chart': 5}, 'qualifies_for_bonus': False}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['retrieval-dev', 'run', 'freeze', 'evaluate'])
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    args = parser.parse_args()
    if args.action == 'retrieval-dev':
        retrieval_dev()
    elif args.action == 'run':
        run(args.split)
    elif args.action == 'freeze':
        freeze()
    else:
        evaluate(args.split)
