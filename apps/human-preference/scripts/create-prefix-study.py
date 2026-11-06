"""Freeze verified 100-word prefixes without changing the original 35-word study."""
import argparse, copy, hashlib, json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--reference', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
old = json.loads((root / 'study/study.json').read_bytes())
output = root / 'study-100'
if output.exists():
    raise SystemExit('Refusing to overwrite an existing frozen study.')
sha = lambda b: hashlib.sha256(b).hexdigest()
encode = lambda value: (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
spec = copy.deepcopy(old)
spec.update(title='OWT Human Preference · 100 words', excerpt_words=100,
    cohort='hosted-prefix100-2026-11',
    excerpt_rule='First 100 whitespace-delimited words, whitespace joined by one space; shorter samples retained unchanged; original tokens and artifacts retained; ellipsis shown only for truncation.',
    protocol='Judge the fluency, coherence and overall quality of the displayed excerpts. Prefer the text that reads better; choose A or B, or Skip if you cannot decide. Judge only the visible excerpt. These are independent unconditional samples, not responses to a shared prompt.')
spec.pop('length_reference')
spec['predecessor_study_id'] = old['id']
spec['collection_policy'] = 'Independent collection and Elo ratings; no responses from the 35-word study are pooled into this study.'
spec['llm_source']['caveat'] = spec['llm_source']['caveat'].replace('35-word', 'up-to-100-word')
spec['source_repository'] = {'url': 'https://github.com/antoniofrancaib/dlmbench/tree/f3f3eb64322a2ab3b1fc7aa85e295cc70b9d0b6e/samples', 'commit': 'f3f3eb64322a2ab3b1fc7aa85e295cc70b9d0b6e', 'reference': 'Recovered separately from the original reference bank; verified against its frozen source checksum and all 1,024 full-text hashes.'}
files, short = {}, {}
for bank in spec['banks']:
    original = json.loads((root / 'study' / bank['file']).read_bytes())
    path = args.reference if bank['id'] == spec['reference_id'] else root.parents[1] / 'samples/owt' / bank['id'] / 'samples.jsonl'
    raw = path.read_bytes()
    assert sha(raw) == bank['source_sha256'] and len(raw) == bank['source_bytes'], bank['id']
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(records) == bank['count'] == len(original['samples'])
    frozen = {k: v for k, v in original.items() if k != 'samples'}
    frozen['samples'] = []
    short[bank['id']] = 0
    for previous, record in zip(original['samples'], records):
        text = record['text']
        assert sha(text.encode()) == previous['full_text_sha256']
        assert str(record['id']) == previous['sample_id']
        words = text.split()
        assert ' '.join(words[:35]) == previous['text']
        excerpt = ' '.join(words[:100])
        frozen['samples'].append({**previous, 'text': excerpt, 'word_count': min(len(words), 100), 'truncated': len(words) > 100, 'excerpt_sha256': sha(excerpt.encode())})
        short[bank['id']] += len(words) < 100
    files[bank['file']] = encode(frozen)
    bank['sha256'] = sha(files[bank['file']])
spec['short_sample_counts'] = short
spec['id'] = 'owt-prefix100-' + sha(encode({k:v for k,v in spec.items() if k != 'id'}))[:16]
files['study.json'] = encode(spec)
for name, raw in files.items():
    path = output / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
print(json.dumps({'study_id': spec['id'], 'sources': len(spec['banks']), 'samples': spec['sample_count'], 'short_samples': sum(short.values())}))
