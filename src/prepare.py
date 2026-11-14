"""Prepare GUIDE part 1 from the existing Day 6 submission; no API calls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAY6 = ROOT.parent / 'K5-L3A-DAY06-NguyenTrongPhuongBach-2A202603716-DataIngestion'


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def exact_evidence(chunks, page, quote):
    # Recover original whitespace without changing the chunk text or its vector.
    pattern = re.compile(r'\s+'.join(re.escape(word) for word in quote.split()))
    for chunk in chunks:
        match = pattern.search(chunk['text']) if chunk['page'] == page else None
        if match:
            return {'source_id': chunk['source_id'], 'page': page, 'evidence': match.group()}
    raise ValueError(f'Evidence not contained in a chunk on page {page}')


def make_benchmark(day6, chunks):
    questions = []
    for old in read_jsonl(day6 / 'benchmark/questions.jsonl'):
        if old['split'] != 'dev':
            continue
        gold = [exact_evidence(chunks, old['gold_page'], old['evidence_quote'])]
        for alt in old.get('alternative_gold', []):
            gold.append(exact_evidence(chunks, alt['gold_page'], alt['evidence_quote']))
        q = {key: old[key] for key in ('id', 'split', 'question', 'expected_answer')}
        q.update(answerable=True, gold=gold)
        if old.get('filter'):
            q['filter'] = old['filter']
        questions.append(q)

    # New test topics, distinct from Day 6 q16–q20; no test retrieval is run here.
    new_test = [
        (23, 'Trong câu lệnh if, có thể có bao nhiêu phần elif và có bắt buộc else không?',
         'Có thể có từ 0 phần elif trở lên; else không bắt buộc.',
         'There can be zero or more elif parts, and the else part is optional.'),
        (23, 'Vòng for của Python duyệt các phần tử của một sequence theo thứ tự nào?',
         'Theo thứ tự các phần tử xuất hiện trong sequence.',
         'Python’s for statement iterates over the items of any sequence (a list or a string), in the order that they appear in the sequence.'),
        (40, 'Dùng list làm stack LIFO: thêm vào đỉnh và lấy khỏi đỉnh bằng thao tác nào?',
         'Thêm bằng append(); lấy bằng pop() không truyền index. Phần tử thêm cuối được lấy trước.',
         'To add an item to the top of the stack, use append(). To retrieve an item from the top of the stack, use pop() without an explicit index.'),
        (46, 'Khi duyệt dictionary, phương thức nào lấy đồng thời key và value?',
         'Dùng phương thức items().',
         'When looping through dictionaries, the key and corresponding value can be retrieved at the same time using the items() method.'),
        (46, 'Hàm nào lấy đồng thời vị trí index và giá trị khi duyệt sequence?',
         'Dùng enumerate().',
         'When looping through a sequence, the position index and corresponding value can be retrieved at the same time using the enumerate() function.'),
    ]
    for number, (page, question, answer, quote) in enumerate(new_test, 1):
        questions.append({'id': f'test{number:02}', 'split': 'test', 'answerable': True,
                          'question': question, 'expected_answer': answer,
                          'gold': [exact_evidence(chunks, page, quote)]})

    missing = [
        ('dev', 'asyncio.TaskGroup bảo đảm hủy task còn lại thế nào khi một task lỗi?', ['TaskGroup'],
         'Không có nội dung TaskGroup hoặc quy tắc hủy task trong corpus chương 3–5.'),
        ('dev', 'Theo tài liệu, cấu hình isolation level của sqlite3.connect như thế nào?', ['sqlite3', 'isolation_level'],
         'Corpus không có API sqlite3.connect hoặc hướng dẫn isolation level.'),
        ('test', 'Theo tài liệu, dùng tracemalloc để đo peak memory của một list như thế nào?', ['tracemalloc'],
         'Corpus có list nhưng không có API tracemalloc để đo peak memory.'),
        ('test', 'Theo tài liệu, ProcessPoolExecutor.map đặt chunksize thế nào để tăng tốc?', ['ProcessPoolExecutor', 'chunksize'],
         'Corpus không có ProcessPoolExecutor hoặc hướng dẫn chunksize của executor.'),
    ]
    corpus = '\n'.join(c['text'] for c in chunks).casefold()
    for number, (split, question, terms, reason) in enumerate(missing, 1):
        if any(term.casefold() in corpus for term in terms):
            raise ValueError(f'Recheck absence evidence for {terms}')
        questions.append({'id': f'absent{number:02}', 'split': split, 'answerable': False,
                          'question': question, 'expected_behavior': 'search_and_abstain',
                          'absence_reason': reason,
                          'absence_check': {'scope': 'All 35 indexed pages, chapters 3–5',
                                            'terms_not_found': terms,
                                            'method': 'Full-corpus term scan and chapter/topic review; not top-k search'}})
    counts = Counter((q['split'], q['answerable']) for q in questions)
    assert counts == {('dev', True): 15, ('dev', False): 2,
                      ('test', True): 5, ('test', False): 2}
    assert len({q['id'] for q in questions}) == len(questions)
    return questions


def verify_store(path, chunks, dimensions):
    from qdrant_client import QdrantClient
    client = QdrantClient(path=str(path))
    try:
        info = client.get_collection('python_C')
        assert info.config.params.vectors.size == dimensions
        assert info.config.params.vectors.distance.value == 'Cosine'
        points, offset = client.scroll('python_C', limit=len(chunks) + 1,
                                       with_payload=True, with_vectors=True)
        assert offset is None and len(points) == len(chunks)
        expected = {c['id']: c for c in chunks}
        for point in points:
            p = point.payload
            c = expected[p['chunk_id']]
            assert (p['text'], p['document_id'], p['source'], p['page']) == (
                c['text'], c['source_id'], c['source'], c['page'])
            assert {k: p[k] for k in c['metadata']} == c['metadata']
            assert len(point.vector) == dimensions
        # Exercise real vector search with an existing vector, without a new embedding.
        first = points[0]
        hits = client.query_points('python_C', query=first.vector, limit=3).points
        assert hits and hits[0].score > 0.999
        return {'collection': 'python_C', 'chunks': len(points), 'dimensions': dimensions,
                'payloads_match_export': True, 'stored_vector_search': 'passed',
                'api_requests': 0, 'new_embedding_tokens': 0, 'new_embedding_cost_usd': 0,
                'note': 'Connection check only; not a query or agent benchmark.'}
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--day6', type=Path, default=DEFAULT_DAY6)
    args = parser.parse_args()
    day6 = args.day6.resolve()
    manifest = json.loads((day6 / 'data/manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((day6 / 'data/tutorial.pdf').read_bytes()).hexdigest() == manifest['sha256']
    raw = read_jsonl(day6 / 'results/chunks_C.jsonl')
    chunks = [{'id': c['chunk_id'], 'text': c['text'], 'source_id': c['document_id'],
               'source': c['source'], 'page': c['page'], 'metadata': c['metadata']} for c in raw]
    assert len(chunks) == 120 and len({c['page'] for c in chunks}) == 35
    questions = make_benchmark(day6, chunks)
    for name in ('tutorial.pdf', 'LICENSE-PYTHON.txt'):
        shutil.copyfile(day6 / 'data' / name, ROOT / 'data' / name)
    manifest.pop('pages_path', None)
    write_json(ROOT / 'data/manifest.json', manifest)
    write_json(ROOT / 'data/chunks.json', chunks)
    for split in ('dev', 'test'):
        selected = [q for q in questions if q['split'] == split]
        write_json(ROOT / f'benchmark/{split}.json', selected)
        # Runtime input has no answers, evidence, or absence labels.
        write_json(ROOT / f'data/queries_{split}.json', [
            {k: q[k] for k in ('id', 'question', 'filter') if k in q} for q in selected])
    model = json.loads((day6 / 'artifacts/embedding_model.json').read_text())
    write_json(ROOT / 'artifacts/embedding_model.json', model)
    store_path = ROOT / '.local/qdrant'
    if not store_path.exists():
        store_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(day6 / '.local/qdrant', store_path, ignore=shutil.ignore_patterns('.lock'))
    check = verify_store(store_path, chunks, model['dimensions'])
    check['benchmark_counts'] = {'dev_answerable': 15, 'dev_unanswerable': 2,
                                 'test_answerable': 5, 'test_unanswerable': 2}
    write_json(ROOT / 'results/preparation.json', check)
    write_json(ROOT / 'artifacts/day6_connection.json', {
        'day6_root': str(day6), 'backend': 'Qdrant local', 'path': '.local/qdrant',
        'collection': 'python_C', 'distance': 'cosine', 'qdrant_client_version': '1.20.0',
        'embedding_model': model['name'], 'dimensions': model['dimensions'],
        'query_processing': 'Unprefixed question text, same model and dimensions as documents',
        'source_sha256': manifest['sha256'], 'index_rebuilt': False})
    print('Part 1 OK: 120 chunks, 35 pages, Qdrant verified, dev 17 / test 7. No API calls.')


if __name__ == '__main__':
    main()
