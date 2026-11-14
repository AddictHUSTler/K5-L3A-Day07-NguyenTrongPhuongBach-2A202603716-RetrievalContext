"""Submission integrity checks for full-document coverage and multi-page evidence."""
import hashlib
import json
import unittest

from src.bonus import supported
from src.retrieval import ROOT


def read(name):
    return json.loads((ROOT / name).read_text())


class BonusTests(unittest.TestCase):
    def test_every_physical_page_accounted_for_and_basic_corpus_preserved(self):
        manifest = read('data/bonus/manifest.json')
        pages = read('data/bonus/pages.json')
        chunks = read('data/bonus/chunks.json')
        self.assertEqual([p['page'] for p in pages], list(range(1, 162)))
        self.assertEqual({c['page'] for c in chunks}, {p['page'] for p in pages if p['text']})
        self.assertEqual(set(manifest['excluded_pages']), {p['page'] for p in pages if not p['text']})
        self.assertEqual(manifest['sha256'], hashlib.sha256((ROOT / 'data/tutorial.pdf').read_bytes()).hexdigest())
        self.assertEqual(len(read('data/chunks.json')), 120)
        self.assertTrue(set(c['id'] for c in chunks).isdisjoint(c['id'] for c in read('data/chunks.json')))

    def test_gold_resolves_and_runtime_has_no_answers(self):
        chunks = read('data/bonus/chunks.json')
        seen = set()
        for split in ('dev', 'test'):
            cases = read(f'benchmark/bonus/{split}.json')
            queries = read(f'data/bonus/queries_{split}.json')
            self.assertEqual([q['id'] for q in queries], [c['id'] for c in cases])
            for q, case in zip(queries, cases):
                self.assertEqual(set(q), {'id', 'question'})
                self.assertNotIn(case['question'], seen)
                seen.add(case['question'])
                if case['answerable']:
                    self.assertEqual(case['evidence_policy'], 'all_required')
                    for evidence in case['gold']:
                        self.assertTrue(supported(evidence, chunks), case['id'])
                        self.assertTrue(evidence['region'])
                    if case['group'] == 'multi_hop':
                        self.assertGreaterEqual(len({g['page'] for g in case['gold']}), 2)
                else:
                    corpus = '\n'.join(c['text'] for c in chunks).casefold()
                    for term in case['absence_check']['terms_not_found']:
                        self.assertNotIn(term.casefold(), corpus)

    def test_wrong_page_cannot_satisfy_required_evidence(self):
        evidence = {'source_id': 'tutorial', 'page': 63, 'evidence': 'binary mode'}
        row = {'source_id': 'tutorial', 'page': 91, 'text': 'binary mode'}
        self.assertFalse(supported(evidence, [row]))
        row['page'] = 63
        self.assertTrue(supported(evidence, [row]))

    def test_extension_uses_original_diagrams_and_separate_runtime_inputs(self):
        cases = read('benchmark/bonus/extension.json')
        queries = read('data/bonus/queries_extension.json')
        chunks = read('data/bonus/chunks.json')
        regions = {r['id']: r for r in read('data/bonus/regions.json')}
        self.assertEqual(len(cases), 6)
        self.assertEqual(sum(c['group'] == 'image_diagram' for c in cases), 5)
        self.assertEqual(queries, [{k: c[k] for k in ('id', 'question')} for c in cases])
        old_questions = {c['question'] for split in ('dev', 'test')
                         for c in read(f'benchmark/bonus/{split}.json')}
        for case in cases:
            self.assertNotIn(case['question'], old_questions)
            for evidence in case['gold']:
                self.assertTrue(supported(evidence, chunks))
                if case['group'] == 'image_diagram':
                    region = regions[evidence['region']]
                    self.assertEqual(region['page'], evidence['page'])
                    self.assertTrue((ROOT / region['crop']).is_file())
                    x0, y0, x1, y1 = region['bbox_points']
                    self.assertTrue(0 <= x0 < x1 < 595.28)
                    self.assertTrue(0 <= y0 < y1 < 841.89)

if __name__ == '__main__':
    unittest.main()
