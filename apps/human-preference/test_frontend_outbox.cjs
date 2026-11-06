'use strict';
// Pure browser/API simulation. Never connects to the study or writes its journal.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { randomUUID } = require('node:crypto');
const source = fs.readFileSync(path.join(__dirname, 'static/app.js'), 'utf8');
const html = fs.readFileSync(path.join(__dirname, 'static/index.html'), 'utf8');
const PID = 'owt-human-preference-participant';
const OUTBOX = 'owt-human-preference-pending-vote';
const localId = '11111111-1111-4111-8111-111111111111';
const sharedId = '22222222-2222-4222-8222-222222222222';
const pendingVote = () => ({ participant_id: localId, pair_id: 'pair-old', choice: 'a', request_id: randomUUID() });
const state = (pid = localId, pairId = 'pair-old', votes = 0) => ({
  participant: { id: pid, votes, can_undo: votes > 0 },
  pair: { id: pairId, a: { text: `Text A ${pairId}`, word_count: 35 }, b: { text: `Text B ${pairId}`, word_count: 35 } },
  study: { excerpt_words: 35, source_count: 13, sample_count: 13312 },
  stats: { votes, total_pairs: 78 }, rankings: [], correlations: [], revealed: true
});
class ElementMock {
  constructor(id) { this.id = id; this.events = new Map(); this.children = []; this.hidden = false; this.disabled = false; this.textContent = ''; this.dataset = {}; this.classList = { toggle() {}, add() {} }; }
  addEventListener(name, handler) { this.events.set(name, [...(this.events.get(name) || []), handler]); }
  async emit(name, event = {}) { for (const handler of this.events.get(name) || []) await handler({ target: this, preventDefault() {}, ...event }); }
  setAttribute(name, value) { this[name] = value; }
  removeAttribute(name) { delete this[name]; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  closest() { return null; }
  focus() {}
}
async function flush() { for (let i = 0; i < 5; i++) await new Promise(resolve => setImmediate(resolve)); }
async function browser({ stored = new Map([[PID, localId]]), handler, storageFailure = false } = {}) {
  const ids = new Map([...html.matchAll(/id="([^"]+)"/g)].map(match => [match[1], new ElementMock(match[1])]));
  ids.get('export-links').hidden = true;
  const buttons = ['a', 'b'].map(choice => { const button = new ElementMock(choice); button.dataset.choice = choice; button.disabled = true; return button; });
  const tabs = new ElementMock('tabs');
  const connection = new ElementMock('connection');
  const document = new ElementMock('document');
  Object.assign(document, {
    hidden: false,
    getElementById: id => ids.get(id),
    querySelector: selector => ({ '.tabs': tabs, '.connection': connection }[selector]),
    querySelectorAll: selector => selector === '[data-choice]' ? buttons : [],
    createElement: () => new ElementMock(), createDocumentFragment: () => new ElementMock(),
    createTextNode: text => ({ textContent: text })
  });
  const window = new ElementMock('window');
  const calls = [];
  const timers = new Map();
  let timerId = 0;
  let failStorage = storageFailure;
  const localStorage = {
    getItem(key) { if (failStorage) throw new Error('Storage blocked'); return stored.has(key) ? stored.get(key) : null; },
    setItem(key, value) { if (failStorage) throw new Error('Storage blocked'); stored.set(key, value); },
    removeItem(key) { if (failStorage) throw new Error('Storage blocked'); stored.delete(key); }
  };
  const context = vm.createContext({
    document, window, Element: ElementMock, localStorage,
    navigator: { onLine: true, locks: { request: (_name, task) => Promise.resolve().then(task) } },
    crypto: { randomUUID }, AbortController, console,
    setTimeout(fn, delay) { const id = ++timerId; timers.set(id, { fn, delay }); return id; },
    clearTimeout(id) { timers.delete(id); }, setInterval() {},
    async fetch(url, options) {
      const body = options.body ? JSON.parse(options.body) : undefined;
      const call = { url, body, options };
      calls.push(call);
      let result;
      if (handler) result = await handler(call, { stored, calls });
      if (result === undefined) {
        if (url === '/api/session') result = { mode: 'local', can_export: true };
        else if (url.startsWith('/api/state?')) result = state();
        else throw new Error(`Unexpected request ${url}`);
      }
      if (result && result.httpStatus) return { ok: false, status: result.httpStatus, json: async () => ({ error: result.error }) };
      return { ok: true, status: 200, json: async () => result };
    }
  });
  vm.runInContext(source, context, { filename: 'static/app.js' });
  await flush();
  return {
    ids, calls, stored, buttons, timers, window, context,
    setStorageFailure(value) { failStorage = value; },
    async click(id) { const element = ids.get(id) || buttons.find(button => button.id === id); if (!element.disabled) await element.emit('click'); await flush(); },
    async fireDelay(delay) { const entry = [...timers.entries()].find(([, timer]) => timer.delay === delay); assert.ok(entry, `No ${delay}ms timer found`); timers.delete(entry[0]); entry[1].fn(); await flush(); }
  };
}

test('retains the existing local UUID, stores exact payload before network, and clears only on matching acknowledgement', async () => {
  const app = await browser({ handler(call, { stored }) {
    if (call.url === '/api/vote') {
      assert.deepEqual(JSON.parse(stored.get(OUTBOX)), call.body);
      assert.deepEqual(Object.keys(call.body).sort(), ['choice', 'pair_id', 'participant_id', 'request_id']);
      assert.equal(call.body.participant_id, localId);
      return { ...state(localId, 'pair-next', 1), acknowledged_request_id: call.body.request_id };
    }
  } });
  assert.equal(app.calls[0].url, '/api/session');
  assert.equal(app.stored.get(PID), localId);
  assert.equal(app.ids.get('export-links').hidden, false);
  await app.click('a');
  assert.equal(app.stored.has(OUTBOX), false);
  assert.match(app.ids.get('save-status').textContent, /^Saved/);
  assert.equal(app.ids.get('sample-a').textContent, 'Text A pair-next');
  assert.equal(app.buttons[0].disabled, false);
});

test('a lost response keeps the same comparison locked and retries the same request ID', async () => {
  let fail = true;
  const app = await browser({ handler(call) {
    if (call.url === '/api/vote') {
      if (fail) throw new Error('Disconnected after commit');
      return { ...state(localId, 'pair-next', 1), acknowledged_request_id: call.body.request_id };
    }
  } });
  await app.click('a');
  const queued = app.stored.get(OUTBOX);
  assert.ok(queued);
  assert.equal(app.ids.get('sample-a').textContent, 'Text A pair-old');
  assert.equal(app.buttons[0].disabled, true);
  assert.equal(app.ids.get('skip-button').disabled, true);
  assert.equal(app.ids.get('undo-button').disabled, true);
  assert.doesNotMatch(app.ids.get('save-status').textContent, /^Saved/);
  await app.click('b');
  assert.equal(app.calls.filter(call => call.url === '/api/vote').length, 1);
  fail = false;
  await app.click('retry-button');
  const votes = app.calls.filter(call => call.url === '/api/vote');
  assert.equal(votes.length, 2);
  assert.deepEqual(votes[0].body, votes[1].body);
  assert.equal(app.calls.filter(call => call.url.startsWith('/api/state?')).length, 1);
  assert.equal(app.stored.has(OUTBOX), false);
});

test('reload replays a persisted pending vote before requesting any next pair', async () => {
  const vote = pendingVote();
  const app = await browser({ stored: new Map([[PID, localId], [OUTBOX, JSON.stringify(vote)]]), handler(call) {
    if (call.url === '/api/vote') return { ...state(localId, 'pair-next', 1), acknowledged_request_id: vote.request_id };
  } });
  assert.deepEqual(app.calls.map(call => call.url), ['/api/session', '/api/vote']);
  assert.deepEqual(app.calls[1].body, vote);
  assert.equal(app.ids.get('sample-a').textContent, 'Text A pair-next');
  assert.equal(app.stored.has(OUTBOX), false);
});

test('incorrect or missing acknowledgement retains the outbox without advancing', async () => {
  const app = await browser({ handler(call) {
    if (call.url === '/api/vote') return { ...state(localId, 'pair-next', 1), acknowledged_request_id: 'wrong-request' };
  } });
  await app.click('a');
  assert.ok(app.stored.get(OUTBOX));
  assert.equal(app.ids.get('sample-a').textContent, 'Text A pair-old');
  assert.equal(app.buttons[0].disabled, true);
  assert.match(app.ids.get('error-text').textContent, /request ID/);
  assert.equal([...app.timers.values()].some(timer => timer.delay === 1000), false);
});

test('shared sessions use the cookie identity, omit names, hide exports, and preserve the local UUID', async () => {
  const app = await browser({ handler(call) {
    if (call.url === '/api/session') return { mode: 'shared', participant_id: sharedId, can_export: false };
    if (call.url.startsWith('/api/state?')) return state(sharedId);
    if (call.url === '/api/vote') {
      assert.equal(call.body.participant_id, sharedId);
      assert.equal('name' in call.body, false);
      return { ...state(sharedId, 'pair-next', 1), acknowledged_request_id: call.body.request_id };
    }
  } });
  assert.equal(app.stored.get(PID), localId);
  assert.equal(app.ids.get('export-links').hidden, true);
  assert.equal(app.ids.get('export-json').href, undefined);
  assert.ok(app.calls[1].url.includes(sharedId));
  await app.click('b');
  assert.equal(app.stored.has(OUTBOX), false);
  assert.equal(app.ids.get('connection-text').textContent, 'Connected');
  assert.equal(app.ids.has('participant-name'), false);
  assert.doesNotMatch(source, /Antonio|participant-name/);
});

test('a shared-session mismatch preserves the exact unsent choice and sends no vote', async () => {
  const vote = pendingVote();
  const raw = JSON.stringify(vote);
  const app = await browser({ stored: new Map([[PID, localId], [OUTBOX, raw]]), handler(call) {
    if (call.url === '/api/session') return { mode: 'shared', participant_id: sharedId, can_export: false };
  } });
  assert.deepEqual(app.calls.map(call => call.url), ['/api/session']);
  assert.equal(app.stored.get(OUTBOX), raw);
  assert.match(app.ids.get('error-text').textContent, /different anonymous session/);
  assert.equal(app.buttons[0].disabled, true);
});

test('unavailable storage disables voting instead of claiming a durable queue', async () => {
  const app = await browser({ storageFailure: true });
  assert.equal(app.buttons[0].disabled, true);
  assert.equal(app.calls.some(call => call.url === '/api/vote'), false);
  assert.match(app.ids.get('error-text').textContent, /cannot reliably store pending choices/);
  app.setStorageFailure(false);
  await app.click('retry-button');
  assert.equal(app.buttons[0].disabled, false);
});

test('failed local persistence at click time prevents submission', async () => {
  const app = await browser();
  app.setStorageFailure(true);
  await app.click('a');
  assert.equal(app.calls.some(call => call.url === '/api/vote'), false);
  assert.equal(app.buttons[0].disabled, true);
  assert.match(app.ids.get('save-status').textContent, /not submitted/);
});

test('automatic retries are exponential and bounded; online resumes the same vote', async () => {
  let fail = true;
  const app = await browser({ handler(call) {
    if (call.url === '/api/vote') {
      if (fail) throw new Error('Offline');
      return { ...state(localId, 'pair-next', 1), acknowledged_request_id: call.body.request_id };
    }
  } });
  await app.click('a');
  for (const delay of [1000, 2000, 4000, 8000, 16000, 30000]) await app.fireDelay(delay);
  assert.equal(app.calls.filter(call => call.url === '/api/vote').length, 7);
  assert.equal(app.timers.size, 0);
  assert.match(app.ids.get('save-status').textContent, /still pending/);
  const bodies = app.calls.filter(call => call.url === '/api/vote').map(call => call.body);
  assert.ok(bodies.every(body => JSON.stringify(body) === JSON.stringify(bodies[0])));
  fail = false;
  await app.window.emit('online');
  await flush();
  assert.equal(app.stored.has(OUTBOX), false);
});

test('skip uses the same durable path and undo remains functional after acknowledgement', async () => {
  const app = await browser({ handler(call) {
    if (call.url === '/api/vote') {
      assert.equal(call.body.choice, 'skip');
      return { ...state(localId, 'pair-next', 1), acknowledged_request_id: call.body.request_id };
    }
    if (call.url === '/api/undo') return state(localId, 'pair-restored', 0);
  } });
  await app.click('skip-button');
  assert.equal(app.stored.has(OUTBOX), false);
  assert.match(app.ids.get('save-status').textContent, /Skip saved/);
  await app.click('undo-button');
  assert.equal(app.calls.filter(call => call.url === '/api/undo').length, 1);
  assert.equal(app.ids.get('sample-a').textContent, 'Text A pair-restored');
});

test('unreadable existing outbox is preserved and never overwritten', async () => {
  const raw = '{broken-json';
  const app = await browser({ stored: new Map([[PID, localId], [OUTBOX, raw]]) });
  assert.equal(app.stored.get(OUTBOX), raw);
  assert.equal(app.buttons[0].disabled, true);
  assert.match(app.ids.get('error-text').textContent, /could not be read/);
  assert.equal(app.calls.some(call => call.url === '/api/vote'), false);
});

test('a storage failure after server acknowledgement retains the queue for safe deduplicated retry', async () => {
  let failClear = true;
  let app;
  app = await browser({ handler(call) {
    if (call.url === '/api/vote') {
      if (failClear) app.setStorageFailure(true);
      return { ...state(localId, 'pair-next', 1), acknowledged_request_id: call.body.request_id };
    }
  } });
  await app.click('a');
  assert.ok(app.stored.get(OUTBOX));
  assert.equal(app.buttons[0].disabled, true);
  assert.doesNotMatch(app.ids.get('save-status').textContent, /^Saved/);
  failClear = false;
  app.setStorageFailure(false);
  await app.click('retry-button');
  const submissions = app.calls.filter(call => call.url === '/api/vote');
  assert.deepEqual(submissions[0].body, submissions[1].body);
  assert.equal(app.stored.has(OUTBOX), false);
});

test('another tab’s queue is reconciled when its storage event arrives during a state request', async () => {
  let finishState;
  const app = await browser({ handler(call) {
    if (call.url.startsWith('/api/state?')) return new Promise(resolve => { finishState = resolve; });
    if (call.url === '/api/vote') throw new Error('Keep pending for inspection');
  } });
  const vote = pendingVote();
  app.stored.set(OUTBOX, JSON.stringify(vote));
  await app.window.emit('storage', { key: OUTBOX });
  finishState(state(localId, 'unrelated-next-pair'));
  await flush();
  assert.equal(app.buttons[0].disabled, true);
  assert.equal(app.ids.get('sample-a').textContent.includes('unrelated-next-pair'), false);
  assert.deepEqual(app.calls.find(call => call.url === '/api/vote').body, vote);
  assert.equal(app.stored.get(OUTBOX), JSON.stringify(vote));
});
