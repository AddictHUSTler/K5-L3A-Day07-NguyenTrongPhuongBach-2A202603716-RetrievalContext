import unittest
from src.agent import run, select_context, validate_answer


class ContextTests(unittest.TestCase):
    def test_budget_and_duplicate_content(self):
        rows = [dict(id='a', text='evidence', source='doc', page=1),
                dict(id='b', text='evidence', source='doc', page=1)]
        text, ids = select_context(rows, 100, True)
        self.assertEqual(ids, ['a'])
        self.assertLessEqual(len(text), 100)
        self.assertEqual(select_context(rows, 1, True), ('', []))

    def test_reject_citation_outside_context(self):
        with self.assertRaises(ValueError):
            validate_answer(dict(status='answered', answer='Claim [x]', citations=['x']), ['y'])

    def test_no_answer_without_retrieval(self):
        class SkipsSearch:
            def call(self, messages, choice):
                return {'finish_reason': 'stop', 'message': {'content': '{}'}}, {'estimated_cost_usd': 0}
        trace = run(dict(id='q', question='question'),
                    dict(max_searches=1), SkipsSearch(), None, None)
        self.assertEqual(trace['status'], 'error')
        self.assertEqual(trace['error'], 'Model skipped required search')

    def test_provider_failure_remains_error(self):
        class Broken:
            def call(self, messages, choice):
                raise RuntimeError('Model API HTTP 429')
        trace = run(dict(id='q', question='question'),
                    dict(max_searches=1), Broken(), None, None)
        self.assertEqual(trace['status'], 'error')
        self.assertEqual(trace['error'], 'Model API HTTP 429')


if __name__ == '__main__':
    unittest.main()
