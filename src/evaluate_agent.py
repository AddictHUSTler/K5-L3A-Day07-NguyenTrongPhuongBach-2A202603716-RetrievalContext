"""Evaluate saved traces against separate gold and explicit evidence reviews."""
import argparse
from collections import Counter
import json
import statistics

from starter.agent_metrics import aggregate
from starter.baseline import hit
from src.benchmark_retrieval import metric
from src.retrieval import ROOT, write_json


def evaluate(split, name):
    path = ROOT / f'results/agent-{split}' / name
    traces = json.loads((path / 'traces.json').read_text())
    cases = json.loads((ROOT / f'benchmark/{split}.json').read_text())
    reviews = json.loads((path / 'reviews.json').read_text())
    if {t['query_id'] for t in traces} != {q['id'] for q in cases}:
        raise ValueError('Incomplete split')
    summary = aggregate(cases, traces, reviews)
    by_id = {t['query_id']: t for t in traces}
    grades = []
    for q in cases:
        if not q['answerable']:
            continue
        t = by_id[q['id']]
        searches = [e for e in t['events'] if e['event'] == 'search']
        first = searches[0]['results'] if searches else []
        all_rows = {c['id']: c for e in searches for c in e['results']}
        ids = t['context_ids']
        grades.append(dict(query_id=q['id'], hit1=any(hit(c, q['gold']) for c in first[:1]),
            hit3=any(hit(c, q['gold']) for c in first[:3]),
            context_hit=False))
        # Multiple gold items are alternatives, not cumulative requirements.
        grades[-1]['context_hit'] = any(
            hit(all_rows[cid], q['gold']) and any(g['evidence'] in t['context'] for g in q['gold'])
            for cid in ids if cid in all_rows)
    summary.update({key: metric(sum(g[key] for g in grades), len(grades))
                    for key in ('hit1', 'hit3', 'context_hit')})
    summary['search_count_distribution'] = dict(Counter(
        sum(e['event'] == 'search' for e in t['events']) for t in traces))
    summary['context_review_distribution'] = dict(Counter(r['context_grade'] for r in reviews if 'context_grade' in r))
    summary['median_elapsed_ms'] = statistics.median(t['elapsed_ms'] for t in traces)
    summary['total_elapsed_ms'] = sum(t['elapsed_ms'] for t in traces)
    summary['estimated_model_cost_usd'] = sum(t['estimated_model_cost_usd'] for t in traces)
    embeddings = json.loads((path / 'embedding_usage.json').read_text())
    summary['estimated_new_embedding_cost_usd'] = sum(r['estimated_new_cost_usd'] for r in embeddings)
    summary['model_calls'] = sum(len(t['model_calls']) for t in traces)
    write_json(path / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['dev', 'test'], required=True)
    parser.add_argument('--configs', nargs='+', required=True)
    args = parser.parse_args()
    for name in args.configs:
        print(name, json.dumps(evaluate(args.split, name), ensure_ascii=False))
