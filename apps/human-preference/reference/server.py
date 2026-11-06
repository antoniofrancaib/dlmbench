"""Local blinded OWT preference collection. Python standard library only.

Immutable, fsynced, hash-chained JSON events are authoritative. A single writer
replays them at startup. Test with a separate --events directory.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
# Historical ties remain replayable; new judgments use binary preferences.
CHOICES = {'a': 1.0, 'b': 0.0, 'both_good': 0.5, 'both_bad': 0.5, 'skip': None}
ACTIVE_CHOICES = ('a', 'b', 'skip')
RESPONSE_POLICY = 'binary-v1'
RATING_PROMPT = ('Judge fluency, coherence and overall quality of the displayed excerpts. '
                 'Choose A or B; skip if you cannot decide. Judge only the visible excerpt. '
                 'These are independent unconditional samples, not responses to a shared prompt.')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_immutable(directory, filename, raw):
    path = directory / filename
    if path.exists():
        with path.open('rb') as file:
            if file.read() != raw:
                raise ValueError('Existing event bytes differ: ' + str(path))
            os.fsync(file.fileno())
        sync_directory(directory)
        return
    temporary = directory / (filename + '.' + secrets.token_hex(8) + '.pending')
    with temporary.open('xb') as file:
        file.write(raw)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
    sync_directory(directory)


def ranks(values):
    result = [0.0] * len(values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for i in order[start:end]:
            result[i] = (start + 1 + end) / 2
        start = end
    return result


def spearman(a, b):
    if len(a) < 3:
        return None
    x, y = ranks(a), ranks(b)
    mx, my = sum(x) / len(x), sum(y) / len(y)
    xx, yy = sum((v-mx)**2 for v in x), sum((v-my)**2 for v in y)
    if xx == 0 or yy == 0:
        return None
    return sum((v-mx)*(w-my) for v,w in zip(x,y)) / math.sqrt(xx*yy)


class Conflict(ValueError):
    pass


class DurabilityError(RuntimeError):
    """A write may be committed; restart and replay before accepting more work."""


DURABILITY_MESSAGE = ('Saving could not be confirmed. This response may already be on disk. '
                      'Keep the pending response and retry after the server restarts; '
                      'no further responses can be saved until the journal is recovered.')


class Study:
    def __init__(self, data: Path, events: Path, backup_events: Path | None = None):
        self.data, self.events_dir = data, events
        self.backup_events = Path(backup_events) if backup_events is not None else None
        if self.backup_events is not None and self.backup_events.resolve() == events.resolve():
            raise ValueError('The event backup must use a different directory.')
        self.unhealthy_reason = None
        self.spec = json.loads((data / 'study.json').read_text())
        self.banks = {}
        for bank in self.spec['banks']:
            raw = (data / bank['file']).read_bytes()
            if sha(raw) != bank['sha256']:
                raise ValueError('Bank snapshot checksum mismatch: ' + bank['id'])
            loaded = json.loads(raw)
            assert len(loaded['samples']) == bank['count'] == 1024
            assert loaded['source_sha256'] == bank['source_sha256']
            for sample in loaded['samples']:
                assert sha(sample['text'].encode()) == sample['excerpt_sha256']
                assert len(sample['text'].split()) == sample['word_count'] <= self.spec['excerpt_words']
            self.banks[bank['id']] = loaded
        assert len(self.banks) == self.spec['source_count'] == 13
        self.ids = sorted(self.banks)
        self.lock = threading.RLock()
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.lockfile = (events / 'writer.lock').open('a')
        try:
            fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lockfile.close()
            raise RuntimeError('Another server already owns this event journal.') from None
        self.events, self.participants, self.pairs, self.votes = [], {}, {}, {}
        self.codes = {}
        self.imports = {}
        self.requests = {}
        self.response_policy = None
        self.previous_sha = None
        try:
            event_paths = sorted(events.glob('[0-9]*.json'))
            if self.backup_events is not None:
                self.backup_events.mkdir(parents=True, exist_ok=True)
                unexpected = {p.name for p in self.backup_events.glob('[0-9]*.json')} - {p.name for p in event_paths}
                if unexpected:
                    raise ValueError('Backup contains events absent from the authoritative journal; recovery is required.')
            for path in event_paths:
                raw = path.read_bytes()
                event = json.loads(raw)
                checksum = event.get('event_sha256')
                contents = {k: v for k, v in event.items() if k != 'event_sha256'}
                if checksum != sha(canonical(contents)):
                    raise ValueError('Event checksum mismatch: ' + str(path))
                if event['sequence'] != len(self.events)+1 or event['previous_sha256'] != self.previous_sha:
                    raise ValueError('Broken event sequence or hash chain: ' + str(path))
                if event['study_id'] != self.spec['id']:
                    raise ValueError('Events belong to a different study; use their original data.')
                self.events.append(event)
                self.apply(event)
                self.previous_sha = sha(raw)
            if self.events and self.events[0]['payload']['spec_sha256'] != sha((data / 'study.json').read_bytes()):
                raise ValueError('The study protocol changed after collection began.')
            # Existing primary bytes remain untouched when backup is enabled later.
            for path in event_paths:
                self.mirror_event(path.name, path.read_bytes())
            if not self.events:
                order = list(self.ids)
                random.SystemRandom().shuffle(order)
                self.append('study_started', {'codes': {id:f'{i+1:02d}' for i,id in enumerate(order)}, 'spec_sha256': sha((data / 'study.json').read_bytes())})
            if self.events[0]['payload']['spec_sha256'] != sha((data / 'study.json').read_bytes()):
                raise ValueError('The study protocol changed after collection began.')
            if self.response_policy != RESPONSE_POLICY:
                self.append('response_policy_changed', dict(policy=RESPONSE_POLICY, choices=list(ACTIVE_CHOICES),
                    reason='User requested A-or-B preference choices; preserve earlier judgments.'))
        except Exception:
            self.close()
            raise

    def close(self):
        if not self.lockfile.closed:
            fcntl.flock(self.lockfile, fcntl.LOCK_UN)
            self.lockfile.close()

    def ensure_healthy(self):
        if self.unhealthy_reason is not None:
            raise DurabilityError(DURABILITY_MESSAGE)

    def mirror_event(self, filename, raw):
        if self.backup_events is not None:
            write_immutable(self.backup_events, filename, raw)

    def append(self, kind, payload):
        with self.lock:
            self.ensure_healthy()
            event = dict(sequence=len(self.events)+1, id=secrets.token_hex(16), study_id=self.spec['id'],
                         at=now(), type=kind, previous_sha256=self.previous_sha, payload=payload)
            event['event_sha256'] = sha(canonical(event))
            raw = canonical(event) + b'\n'
            filename = f"{event['sequence']:09d}-{event['id']}.json"
            try:
                write_immutable(self.events_dir, filename, raw)
                self.mirror_event(filename, raw)
                self.events.append(event)
                self.previous_sha = sha(raw)
                self.apply(event)
            except Exception as error:
                # A rename may already have committed an event. Never keep writing
                # from stale memory or acknowledge before both durable copies exist.
                self.unhealthy_reason = f'{type(error).__name__}: {error}'
                raise DurabilityError(DURABILITY_MESSAGE) from error
            return event

    def apply(self, event):
        kind, p = event['type'], event['payload']
        if kind == 'study_started':
            self.codes = p['codes']
        elif kind == 'external_import':
            self.imports[p['id']] = p
            participant = p['participant']
            pid = participant['id']
            self.participants[pid] = dict(participant, revealed=False, origin='external_csv')
            for ordinal, record in enumerate(p['records'], 1):
                pair, vote = record['pair'], record['vote']
                self.pairs[pair['id']] = dict(pair, status='answered')
                self.votes[vote['id']] = dict(vote, at=event['at'], sequence=event['sequence'],
                    sequence_in_event=ordinal, undone=False, timestamp_meaning='import_time')
        elif kind == 'response_policy_changed':
            self.response_policy = p['policy']
        elif kind == 'participant':
            self.participants.setdefault(p['id'], dict(id=p['id'], name='', revealed=False))['name'] = p['name']
        elif kind == 'pair_assigned':
            self.pairs[p['id']] = dict(p, status='pending', assigned_at=event['at'])
        elif kind == 'vote':
            self.votes[event['id']] = dict(p, id=event['id'], at=event['at'], sequence=event['sequence'], undone=False,
                                            response_policy=p.get('response_policy', 'four-outcome-v1'))
            self.pairs[p['pair_id']]['status'] = 'answered'
            if p.get('request_id') is not None:
                self.register_request(p)
        elif kind == 'request_acknowledged':
            self.register_request(p)
        elif kind == 'undo':
            self.votes[p['vote_id']]['undone'] = True
            for pair in self.pairs.values():
                if pair['participant_id'] == p['participant_id'] and pair['status'] == 'pending':
                    pair['status'] = 'superseded'
            self.pairs[self.votes[p['vote_id']]['pair_id']]['status'] = 'pending'
        elif kind in ('reveal', 'named_export'):
            self.participants[p['participant_id']]['revealed'] = True
        else:
            raise ValueError('Unknown event: ' + kind)

    def register_request(self, payload):
        request_id = payload['request_id']
        identity = (payload['participant_id'], payload['pair_id'], payload['choice'])
        if request_id in self.requests and self.requests[request_id] != identity:
            raise ValueError('Conflicting persisted request ID.')
        self.requests[request_id] = identity

    def participant(self, pid, name=None):
        if not isinstance(pid, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', pid):
            raise ValueError('A valid participant ID is required.')
        if name is not None and (not isinstance(name, str) or len(name) > 100):
            raise ValueError('Participant name must be at most 100 characters.')
        if pid not in self.participants or (name is not None and name != self.participants[pid]['name']):
            self.append('participant', dict(id=pid, name=name or 'Anonymous'))
        return self.participants[pid]

    def active_votes(self, pid=None):
        return [v for v in self.votes.values() if not v['undone'] and (pid is None or v['participant_id'] == pid)]

    def next_pair(self, pid):
        current = [p for p in self.pairs.values() if p['participant_id'] == pid and p['status'] == 'pending']
        if current:
            return current[-1]
        # Exposure balancing is independent of quality/outcomes. Skips still advance coverage.
        pair_count, source_count, samples_seen = {}, dict.fromkeys(self.ids, 0), {id:{} for id in self.ids}
        for pair in self.pairs.values():
            if pair['participant_id'] != pid:
                continue
            key = tuple(sorted((pair['a']['bank_id'], pair['b']['bank_id'])))
            pair_count[key] = pair_count.get(key, 0)+1
            for side in ('a', 'b'):
                bank, index = pair[side]['bank_id'], pair[side]['index']
                source_count[bank] += 1
                samples_seen[bank][index] = samples_seen[bank].get(index, 0)+1
        options = list(itertools.combinations(self.ids, 2))
        priority = lambda pair: (pair_count.get(pair, 0), source_count[pair[0]]+source_count[pair[1]])
        lowest = min(map(priority, options))
        seed = secrets.token_hex(16)
        rng = random.Random(seed)
        selected = list(rng.choice([pair for pair in options if priority(pair) == lowest]))
        rng.shuffle(selected)
        sides = []
        for bank in selected:
            least_seen = min(samples_seen[bank].get(s['index'], 0) for s in self.banks[bank]['samples'])
            sample = rng.choice([s for s in self.banks[bank]['samples'] if samples_seen[bank].get(s['index'], 0) == least_seen])
            sides.append(dict(sample, bank_id=bank, bank_sha256=self.banks[bank]['source_sha256']))
        pair = dict(id=secrets.token_hex(16), participant_id=pid, seed=seed,
                    a=sides[0], b=sides[1], revealed_at_assignment=self.participants[pid]['revealed'],
                    excerpt_words=self.spec['excerpt_words'], response_policy=RESPONSE_POLICY)
        self.append('pair_assigned', pair)
        return self.pairs[pair['id']]

    def import_preferences(self, batch):
        # This trusted-local CLI path commits one fsynced event for the whole batch.
        # Validate every record before appending anything; never create partial imports.
        with self.lock:
            self.ensure_healthy()
            if batch.get('study_id') != self.spec['id'] or batch.get('version') != 1:
                raise ValueError('Import belongs to a different study or format.')
            import_id = batch['id']
            if import_id in self.imports:
                old = self.imports[import_id]
                if old['inclusion_policy'] != batch['inclusion_policy']:
                    raise Conflict('This CSV is already imported with another inclusion policy.')
                return dict(already_imported=True, **old['summary'])
            if any(old['source']['sha256'] == batch['source']['sha256'] for old in self.imports.values()):
                raise Conflict('This source CSV has already been imported.')
            pid = batch['participant']['id']
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', pid) or pid in self.participants:
                raise ValueError('Imported participant identity is invalid or already in use.')
            if set(batch['mapping'].values()) != set(self.ids):
                raise ValueError('Import source mapping differs from the eligible cohort.')
            if len(batch['records']) != 78:
                raise ValueError('Expected 78 complete imported comparisons.')
            seen, pair_ids_seen, vote_ids_seen = set(), set(), set()
            for ordinal, record in enumerate(batch['records'], 1):
                pair, vote = record['pair'], record['vote']
                if pair['participant_id'] != pid or vote['participant_id'] != pid:
                    raise ValueError('Inconsistent imported participant.')
                if (pair['id'] in self.pairs or pair['id'] in pair_ids_seen or
                        vote['id'] in self.votes or vote['id'] in vote_ids_seen or vote['pair_id'] != pair['id']):
                    raise ValueError('Invalid imported comparison identity.')
                pair_ids_seen.add(pair['id'])
                vote_ids_seen.add(vote['id'])
                if vote['choice'] not in ('a', 'b') or vote['score_a'] != CHOICES[vote['choice']]:
                    raise ValueError('Invalid imported choice or score.')
                if vote['import_row'] != ordinal or not isinstance(vote['include_in_elo'], bool):
                    raise ValueError('Invalid import ordering or inclusion flag.')
                ids = tuple(sorted(pair[side]['bank_id'] for side in ('a', 'b')))
                if ids[0] == ids[1] or ids in seen:
                    raise ValueError('Duplicate or self comparison in import.')
                seen.add(ids)
                for side in ('a', 'b'):
                    sample = pair[side]
                    if sample['bank_id'] not in self.banks:
                        raise ValueError('Ineligible source in import.')
                    if sha(sample['text'].encode()) != sample['excerpt_sha256']:
                        raise ValueError('Imported displayed-text checksum mismatch.')
                    if sample['bank_sha256'] != self.banks[sample['bank_id']]['source_sha256']:
                        raise ValueError('Imported bank checksum mismatch.')
            if seen != set(itertools.combinations(self.ids, 2)):
                raise ValueError('Imported pair coverage is incomplete.')
            if batch['summary']['included'] != sum(r['vote']['include_in_elo'] for r in batch['records']):
                raise ValueError('Import summary disagrees with responses.')
            self.append('external_import', batch)
            return dict(already_imported=False, **batch['summary'])

    def leaderboard(self):
        self.ensure_healthy()
        rows = {id: dict(id=id, code=self.codes[id], elo=float(self.spec['elo']['initial']), matches=0, wins=0, ties=0, losses=0) for id in self.ids}
        counts = {}
        for vote in self.active_votes():
            outcome = CHOICES[vote['choice']]
            if outcome is None or not vote.get('include_in_elo', True):
                continue
            pair = self.pairs[vote['pair_id']]
            aid, bid = pair['a']['bank_id'], pair['b']['bank_id']
            a, b = rows[aid], rows[bid]
            expected = 1/(1+10**max(-300, min(300, (b['elo']-a['elo'])/400)))
            delta = self.spec['elo']['k'] * (outcome-expected)
            a['elo'] += delta
            b['elo'] -= delta
            for row, score in ((a, outcome), (b, 1-outcome)):
                row['matches'] += 1
                row['wins' if score == 1 else 'losses' if score == 0 else 'ties'] += 1
            key = tuple(sorted((aid, bid)))
            counts[key] = counts.get(key, 0)+1
        ordered = sorted(rows.values(), key=lambda row: (row['matches'] == 0, -row['elo'], row['code']))
        previous, rank = None, None
        for i, row in enumerate(ordered):
            if row['matches'] == 0:
                row['rank'] = None
                continue
            if previous != row['elo']:
                rank = i+1
            previous = row['elo']
            row['rank'] = rank
        recorded = [v for v in self.active_votes() if v['choice'] != 'skip']
        stats = dict(recorded_votes=len(recorded),
                     excluded_votes=sum(not v.get('include_in_elo', True) for v in recorded),
                     imported_votes=sum(v.get('origin') == 'external_csv' for v in recorded),
                     votes=sum(counts.values()), pairs_covered=len(counts), total_pairs=78,
                     participants=len({v['participant_id'] for v in self.active_votes() if v['choice'] != 'skip'}),
                     sources_rated=sum(r['matches'] > 0 for r in ordered),
                     min_matches=min(r['matches'] for r in ordered))
        return ordered, stats

    def state(self, pid, *, include_pair=True):
        with self.lock:
            self.ensure_healthy()
            p = self.participant(pid)
            rows, stats = self.leaderboard()
            correlations = []
            rated = [r for r in rows if r['matches']]
            generated = [r for r in rated if r['id'] != self.spec['reference_id']]
            for judge in next(iter(self.spec['llm_ratings'].values())):
                calc = lambda subset: spearman([r['elo'] for r in subset], [self.spec['llm_ratings'][r['id']][judge] for r in subset])
                correlations.append(dict(judge=judge, rho=calc(rated), n=len(rated),
                    rho_generators=calc(generated), n_generators=len(generated),
                    warning=self.spec['llm_source']['caveat']))
            result = dict(study={k:self.spec[k] for k in ('id','title','excerpt_words','source_count','sample_count')} |
                          dict(protocol=RATING_PROMPT, response_policy=RESPONSE_POLICY, choices=list(ACTIVE_CHOICES)),
                          participant=dict(p, votes=sum(v['choice'] != 'skip' for v in self.active_votes(pid)), can_undo=any(v.get('origin') != 'external_csv' for v in self.active_votes(pid))),
                          stats=stats, rankings=[{k:v for k,v in row.items() if k != 'id'} | dict(label=self.banks[row['id']]['label'] if p['revealed'] else 'Source ' + row['code']) for row in rows],
                          correlations=correlations, revealed=p['revealed'])
            if include_pair:
                pair = self.next_pair(pid)
                result['pair'] = dict(id=pair['id'], **{side: {k:pair[side][k] for k in ('text','word_count','truncated')} for side in ('a','b')})
            return result

    def vote(self, body):
        with self.lock:
            self.ensure_healthy()
            pid, pair_id, choice = body.get('participant_id'), body.get('pair_id'), body.get('choice')
            request_id = body.get('request_id')
            if request_id is not None and (not isinstance(request_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', request_id)):
                raise ValueError('Request ID must contain 1–128 ASCII letters, digits, underscores or hyphens.')
            def acknowledged():
                result = self.state(pid)
                if request_id is not None:
                    result['acknowledged_request_id'] = request_id
                return result
            if request_id is not None and request_id in self.requests:
                if self.requests[request_id] != (pid, pair_id, choice):
                    raise Conflict('This request ID was already used for a different response.')
                # Request history survives Undo. A delayed retry must not cast it again.
                return acknowledged()
            self.participant(pid, body.get('name'))
            if choice not in ACTIVE_CHOICES:
                raise ValueError('Choose A or B, or skip this pair.')
            pair = self.pairs.get(pair_id)
            if pair is None or pair['participant_id'] != pid:
                raise ValueError('This comparison does not belong to this participant.')
            if pair['status'] != 'pending':
                prior = [v for v in self.active_votes(pid) if v['pair_id'] == pair_id]
                if prior and prior[-1]['choice'] == choice:
                    if request_id is not None:
                        # Bind a fresh client ID even when the comparison was already
                        # saved through a legacy client without request IDs.
                        self.append('request_acknowledged', dict(request_id=request_id,
                            participant_id=pid, pair_id=pair_id, choice=choice, vote_id=prior[-1]['id']))
                    return acknowledged()
                raise Conflict('This comparison was already answered. Refresh to continue.')
            assigned = datetime.fromisoformat(pair['assigned_at']).timestamp()
            payload = dict(participant_id=pid, participant_name=self.participants[pid]['name'], pair_id=pair_id,
                choice=choice, score_a=CHOICES[choice], response_policy=RESPONSE_POLICY, elapsed_ms=max(0, round((time.time()-assigned)*1000)),
                revealed_at_vote=self.participants[pid]['revealed'])
            if request_id is not None:
                payload['request_id'] = request_id
            self.append('vote', payload)
            return acknowledged()

    def undo(self, pid):
        with self.lock:
            self.ensure_healthy()
            self.participant(pid)
            active = self.active_votes(pid)
            if not active:
                raise Conflict('There is no response to undo.')
            if active[-1].get('origin') == 'external_csv':
                raise Conflict('Imported responses are preserved; correct them through an audited import update.')
            self.append('undo', dict(participant_id=pid, vote_id=active[-1]['id']))
            return self.state(pid)

    def reveal(self, pid, export=False):
        with self.lock:
            self.ensure_healthy()
            self.participant(pid)
            if export or not self.participants[pid]['revealed']:
                self.append('named_export' if export else 'reveal', dict(participant_id=pid))
            return self.state(pid, include_pair=not export)

    def export(self, format):
        with self.lock:
            self.ensure_healthy()
            rows, stats = self.leaderboard()
            if format == 'json':
                return canonical(dict(exported_at=now(), study=self.spec, code_map=self.codes, current_response_policy=RESPONSE_POLICY,
                    rankings=[dict(r, label=self.banks[r['id']]['label']) for r in rows], stats=stats,
                    events=self.events, event_chain_last_sha256=self.previous_sha,
                    votes=list(self.votes.values()), assignments=list(self.pairs.values()),
                    imports=[{k:v for k,v in item.items() if k not in ('records','source')} |
                        dict(source={k:v for k,v in item['source'].items() if k != 'raw_csv'})
                        for item in self.imports.values()]))
            columns = ['study_id','vote_id','sequence','at','participant_id','participant_name','pair_id','request_id','choice','score_a','undone','revealed_at_vote','elapsed_ms','excerpt_words','response_policy',
                       'origin','include_in_elo','exclusion_reason','import_id','import_row','original_round',
                       'blinding_status','judged_at','timestamp_meaning','exact_packet_match','data_quality_flags']
            for side in ('a', 'b'):
                columns += [side+'_'+k for k in ('bank_id','bank_sha256','sample_id','index','full_text_sha256','excerpt_sha256','word_count','truncated','text','exact_packet_match','flags','expected_packet_sha256','mapped_canonical_source')]
            output = io.StringIO(newline='')
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            for vote in self.votes.values():
                pair = self.pairs[vote['pair_id']]
                row = {k:vote[k] for k in columns if k in vote}
                row.update(study_id=self.spec['id'], vote_id=vote['id'], excerpt_words=pair['excerpt_words'],
                           origin=vote.get('origin', 'live'), include_in_elo=vote.get('include_in_elo', True),
                           timestamp_meaning=vote.get('timestamp_meaning', 'response_recorded_time'))
                for side in ('a', 'b'):
                    row.update({side+'_'+k:pair[side].get(k) for k in ('bank_id','bank_sha256','sample_id','index','full_text_sha256','excerpt_sha256','word_count','truncated','text','exact_packet_match','flags','expected_packet_sha256','mapped_canonical_source')})
                writer.writerow({k:json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k,v in row.items()})
            return output.getvalue().encode('utf-8')


class Handler(BaseHTTPRequestHandler):
    server_version = 'OWTPreference/1.0'

    def send(self, content, status=200, content_type='application/json; charset=utf-8', filename=None):
        if not isinstance(content, bytes):
            content = canonical(content)
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        if filename:
            self.send_header('Content-Disposition', 'attachment; filename="'+filename+'"')
        self.end_headers()
        self.wfile.write(content)

    def check_host(self):
        allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        if self.headers.get('Host') not in allowed:
            raise ValueError('Use the localhost URL for this study.')
        origin = self.headers.get('Origin')
        if origin and origin not in {'http://'+host for host in allowed}:
            raise ValueError('Cross-origin requests are not accepted.')

    def do_GET(self):
        try:
            self.check_host()
            url = urlparse(self.path)
            query = parse_qs(url.query)
            pid = query.get('participant_id', [''])[0]
            if url.path == '/api/session':
                return self.send(dict(mode='local', can_export=True))
            if url.path == '/api/health':
                healthy = self.server.study.unhealthy_reason is None
                result = dict(ok=healthy, study_id=self.server.study.spec['id'], samples=self.server.study.spec['sample_count'])
                if not healthy:
                    result['error'] = DURABILITY_MESSAGE
                return self.send(result, status=200 if healthy else 503)
            if url.path in ('/api/state', '/api/rankings'):
                return self.send(self.server.study.state(pid, include_pair=url.path == '/api/state'))
            if url.path == '/api/export':
                format = query.get('format', ['json'])[0]
                if format not in ('json','csv'):
                    raise ValueError('Export format must be json or csv.')
                if not pid:
                    raise ValueError('A participant ID is required to log access to the named export.')
                self.server.study.reveal(pid, export=True)
                return self.send(self.server.study.export(format), content_type='application/json; charset=utf-8' if format == 'json' else 'text/csv; charset=utf-8', filename='owt-human-preferences.'+format)
            files = {'/':'index.html','/index.html':'index.html','/app.js':'app.js','/app.css':'app.css'}
            if url.path in files:
                filename = files[url.path]
                ctype = {'html':'text/html','js':'text/javascript','css':'text/css'}[filename.rsplit('.',1)[1]]
                return self.send((ROOT / 'static' / filename).read_bytes(), content_type=ctype+'; charset=utf-8')
            return self.send(dict(error='Not found'), 404)
        except DurabilityError as error:
            self.send(dict(error=str(error), retry_after_restart=True), 503)
        except (ValueError, KeyError) as error:
            self.send(dict(error=str(error)), 400)
        except Exception:
            import traceback
            traceback.print_exc()
            self.send(dict(error='The request failed. Your saved votes remain on disk.'), 500)

    def do_POST(self):
        try:
            self.check_host()
            if self.headers.get('Content-Type','').split(';')[0] != 'application/json':
                raise ValueError('JSON content is required.')
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 8192:
                raise ValueError('Invalid request size.')
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError('Expected a JSON object.')
            path = urlparse(self.path).path
            if path == '/api/vote':
                result = self.server.study.vote(body)
            elif path == '/api/undo':
                result = self.server.study.undo(body.get('participant_id'))
            elif path == '/api/reveal':
                result = self.server.study.reveal(body.get('participant_id'))
            else:
                return self.send(dict(error='Not found'), 404)
            self.send(result)
        except DurabilityError as error:
            self.send(dict(error=str(error), retry_after_restart=True), 503)
        except Conflict as error:
            self.send(dict(error=str(error)), 409)
        except (ValueError, KeyError, TypeError) as error:
            self.send(dict(error=str(error)), 400)
        except Exception:
            import traceback
            traceback.print_exc()
            self.send(dict(error='Could not complete the request. Retry; duplicate responses are counted only once.'), 500)

    def log_message(self, fmt, *args):
        if args and str(args[1] if len(args)>1 else '') not in ('200','304'):
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--events', type=Path, default=ROOT / 'data' / 'Events')
    parser.add_argument('--backup-events', type=Path, help='Independent directory for durable copies of every event.')
    args = parser.parse_args()
    study = Study(args.data, args.events, backup_events=args.backup_events)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.study = study
    print(f'OWT preference study: http://127.0.0.1:{server.server_port} | {study.spec["sample_count"]} samples | events: {args.events}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        study.close()


if __name__ == '__main__':
    main()
