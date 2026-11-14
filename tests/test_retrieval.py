"""Check fusion, pre-cutoff filters, metric denominators, and real Qdrant mapping."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from src.benchmark_retrieval import summarize
from src.retrieval import Retriever, fuse


def chunk(cid, text, chapter):
    return {'id': cid, 'text': text, 'source_id': 'doc', 'source': 'tutorial.pdf',
            'page': chapter, 'metadata': {'chapter': chapter}}


class RetrievalTests(unittest.TestCase):
    def test_rrf_combines_ranks_instead_of_incompatible_scores(self):
        a, b = chunk('a', 'alpha', 3), chunk('b', 'beta', 4)
        dense = [{**a, 'score': 0.9}, {**b, 'score': 0.8}]
        lexical = [{**b, 'score': 9999}, {**a, 'score': 0.01}]
        rows = fuse(dense, lexical, 2)
        self.assertEqual([r['id'] for r in rows], ['a', 'b'])
        self.assertAlmostEqual(rows[0]['score'], 1 / 61 + 1 / 62)
        self.assertEqual(rows[0]['components']['tfidf']['rank'], 2)

    def test_lexical_filter_precedes_cutoff(self):
        retriever = Retriever([chunk('a', 'alpha', 3), chunk('b', 'alpha beta', 4)])
        rows = retriever.search('alpha', 'tfidf', top_k=1, filters={'chapter': 4})
        self.assertEqual([r['id'] for r in rows], ['b'])
        with self.assertRaises(ValueError):
            retriever.search('alpha', 'tfidf', filters={'page': 4})

    def test_absent_queries_excluded_and_errors_count_as_misses(self):
        c = chunk('a', 'evidence', 3)
        labels = [{'id': 'q', 'answerable': True,
                   'gold': [{'source_id': 'doc', 'page': 3, 'evidence': 'evidence'}]},
                  {'id': 'missing', 'answerable': False}]
        traces = [{'query_id': 'q', 'status': 'ok', 'results': [c],
                   'context_ids': [], 'context': '', 'search_ms': 1},
                  {'query_id': 'missing', 'status': 'ok', 'results': [c],
                   'context_ids': ['a'], 'context': 'evidence', 'search_ms': 1}]
        summary, grades = summarize(traces, labels)
        self.assertEqual(summary['hit3'], {'n': 1, 'N': 1, 'rate': 1})
        self.assertEqual(summary['context_hit']['n'], 0)
        traces[0].update(status='error', results=[])
        summary, _ = summarize(traces, labels)
        self.assertEqual(summary['hit3'], {'n': 0, 'N': 1, 'rate': 0})
        self.assertEqual(summary['errors'], 1)

    def test_dense_does_not_fall_back_to_lexical(self):
        with self.assertRaises(ValueError):
            Retriever([chunk('a', 'alpha', 3)]).search('alpha', 'dense')

    @unittest.skipUnless(importlib.util.find_spec('qdrant_client'), 'requires qdrant-client')
    def test_qdrant_filter_payload_and_hybrid(self):
        from qdrant_client import QdrantClient, models
        chunks = [chunk('a', 'alpha', 3), chunk('b', 'alpha beta', 4)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'qdrant'
            client = QdrantClient(path=str(path))
            client.create_collection('python_C', vectors_config=models.VectorParams(
                size=3, distance=models.Distance.COSINE))
            client.upsert('python_C', points=[models.PointStruct(
                id=i, vector=v, payload={'chunk_id': c['id'], 'document_id': c['source_id'],
                                        'source': c['source'], 'page': c['page'], 'text': c['text'],
                                        **c['metadata']})
                for i, (c, v) in enumerate(zip(chunks, [[1., 0., 0.], [0.9, 0.1, 0.]]))])
            client.close()
            retriever = Retriever(chunks, store_path=path)
            try:
                for method in ['dense', 'hybrid']:
                    rows = retriever.search('alpha', method, filters={'chapter': 4},
                                            vector=[1., 0., 0.])
                    self.assertEqual([r['id'] for r in rows], ['b'])
                    self.assertEqual(rows[0]['metadata'], {'chapter': 4})
                    self.assertEqual(rows[0]['source_id'], 'doc')
            finally:
                retriever.close()


if __name__ == '__main__':
    unittest.main()
