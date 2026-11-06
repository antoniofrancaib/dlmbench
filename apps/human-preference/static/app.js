'use strict';
(() => {
  const $ = (id) => document.getElementById(id);
  const PARTICIPANT_KEY = 'owt-human-preference-participant';
  const OUTBOX_KEY = 'owt-human-preference-pending-vote';
  const LOCK_NAME = 'owt-human-preference-outbox';
  const excerptWords = Number($('excerpt-size').textContent) || 35;
  const RETRY_DELAYS = [1000, 2000, 4000, 8000, 16000, 30000];
  const createId = () => {
    if (globalThis.crypto?.randomUUID) return crypto.randomUUID();
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, char => {
      const r = globalThis.crypto?.getRandomValues ? crypto.getRandomValues(new Uint8Array(1))[0] % 16 : Math.floor(Math.random() * 16);
      return (char === 'x' ? r : (r & 3) | 8).toString(16);
    });
  };
  class RecoveryError extends Error {}
  class StorageError extends RecoveryError {}
  const storageMessage = 'Voting is paused because this browser cannot reliably store pending choices. Enable browser storage, then choose Try again. Any pending choice has not been deliberately removed.';
  const storage = {
    get(key) {
      try { return localStorage.getItem(key); }
      catch { throw new StorageError(storageMessage); }
    },
    set(key, value) {
      try {
        localStorage.setItem(key, value);
        if (localStorage.getItem(key) !== value) throw new Error('Storage verification failed');
      } catch { throw new StorageError(storageMessage); }
    },
    remove(key) {
      try {
        localStorage.removeItem(key);
        if (localStorage.getItem(key) !== null) throw new Error('Storage verification failed');
      } catch { throw new StorageError(storageMessage); }
    },
    verify() {
      const key = `${OUTBOX_KEY}-check-${createId()}`;
      this.set(key, key);
      this.remove(key);
    }
  };
  const withOutboxLock = task => navigator.locks?.request ? navigator.locks.request(LOCK_NAME, task) : Promise.resolve().then(task);
  const voteButtons = [...document.querySelectorAll('[data-choice]')];
  let participantId = null;
  let session = null;
  let state = null;
  let pending = null;
  let storageReady = false;
  let busy = false;
  let pollRunning = false;
  let revision = 0;
  let activeView = 'comparison';
  let needsReload = false;
  let savedMessageTimer = null;
  let retryTimer = null;
  let retryAttempt = 0;
  let storageChanged = false;
  const number = value => Number(value || 0).toLocaleString();
  const query = () => `participant_id=${encodeURIComponent(participantId)}`;
  const sameVote = (a, b) => Boolean(a && b && ['participant_id', 'pair_id', 'choice', 'request_id'].every(key => a[key] === b[key]));

  function readPending() {
    const raw = storage.get(OUTBOX_KEY);
    if (raw === null) return null;
    try {
      const value = JSON.parse(raw);
      if (!value || Object.keys(value).length !== 4 ||
          !['participant_id', 'pair_id', 'request_id'].every(key => typeof value[key] === 'string' && value[key].length > 0) ||
          !['a', 'b', 'skip'].includes(value.choice)) throw new Error('Invalid pending choice');
      return value;
    } catch {
      throw new RecoveryError('A pending choice in this browser could not be read. It has been retained. Ask the study owner for recovery before clearing browser storage.');
    }
  }
  function restorePending() {
    const stored = readPending();
    if (pending && stored && !sameVote(pending, stored)) {
      throw new RecoveryError('Another tab has a different pending choice. Keep both tabs open and ask the study owner for recovery; neither choice will be overwritten here.');
    }
    if (pending && !stored) storage.set(OUTBOX_KEY, JSON.stringify(pending));
    else pending = stored;
  }
  function setConnection(online) {
    $('connection-text').textContent = online ? (session?.mode === 'shared' ? 'Connected' : 'Locally connected') : 'Connection interrupted';
    document.querySelector('.connection').classList.toggle('offline', !online);
  }
  function setError(error) {
    $('error').hidden = !error;
    $('error-text').textContent = error ? String(error.message || error) : '';
    if (error) setConnection(false);
  }
  function setStatus(message) {
    clearTimeout(savedMessageTimer);
    $('save-status').textContent = message;
  }
  async function request(path, body) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20000);
    try {
      const response = await fetch(path, {
        method: body === undefined ? 'GET' : 'POST',
        headers: body === undefined ? { 'Accept': 'application/json' } : { 'Accept': 'application/json', 'Content-Type': 'application/json' },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
        credentials: 'same-origin',
        cache: 'no-store'
      });
      let result;
      try { result = await response.json(); }
      catch { throw new Error('The study returned an unreadable response.'); }
      if (!response.ok || result.error) {
        const error = new Error(result.error || `Request failed (${response.status}).`);
        error.retryable = response.status === 408 || response.status === 429 || response.status >= 500;
        throw error;
      }
      return result;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('The study server took too long to respond.');
      if (error instanceof TypeError) throw new Error('Could not reach the study server.');
      throw error;
    } finally { clearTimeout(timeout); }
  }
  function syncControls() {
    const locked = busy || !storageReady || !session || needsReload || Boolean(pending);
    voteButtons.forEach(button => { button.disabled = locked || !state?.pair; });
    $('skip-button').disabled = locked || !state?.pair;
    $('undo-button').disabled = locked || !state?.participant?.can_undo;
    $('finish-button').disabled = locked || !state?.participant?.can_finish || state?.participant?.finished;
    $('retry-button').disabled = busy;
    $('samples').setAttribute('aria-busy', String(busy));
  }
  function showSaved(message) {
    setStatus(message);
    savedMessageTimer = setTimeout(() => {
      if (!pending && !busy && storageReady && !needsReload) $('save-status').textContent = 'Your choices are saved as you go.';
    }, 5500);
  }
  function showPending() {
    if (!pending) return;
    if (!state || state.pair?.id !== pending.pair_id) {
      for (const side of ['a', 'b']) {
        $(`sample-${side}`).textContent = 'Your previous comparison is waiting for confirmation. The next pair will appear after it is saved.';
        $(`sample-${side}-count`).textContent = '';
      }
    }
    setStatus('Choice kept in this browser · waiting for server confirmation.');
  }
  function cancelRetry() {
    clearTimeout(retryTimer);
    retryTimer = null;
  }
  function scheduleRetry(error) {
    if (!pending || !storageReady || error instanceof RecoveryError || error.retryable === false) return;
    if (navigator.onLine === false) {
      setStatus('Choice kept in this browser · waiting for a connection.');
      return;
    }
    if (retryAttempt >= RETRY_DELAYS.length) {
      setStatus('Choice still pending · use Try again when the connection is ready.');
      return;
    }
    const delay = RETRY_DELAYS[retryAttempt++];
    setStatus(`Choice kept in this browser · retrying in ${delay / 1000}s.`);
    retryTimer = setTimeout(() => { retryTimer = null; resume(); }, delay);
  }
  function cell(text, className) {
    const td = document.createElement('td');
    td.textContent = text;
    if (className) td.className = className;
    return td;
  }
  function renderRankings(data) {
    const fragment = document.createDocumentFragment();
    for (const source of data.rankings || []) {
      const matches = Number(source.matches || 0);
      const row = document.createElement('tr');
      if (!matches) row.className = 'unrated';
      row.append(cell(matches && source.rank != null ? String(source.rank).padStart(2, '0') : '—', 'rank-cell'));
      const label = cell('', 'source-cell');
      const dot = document.createElement('span');
      dot.className = 'source-code';
      dot.setAttribute('aria-hidden', 'true');
      const name = document.createElement('span');
      name.textContent = source.label || ('Source ' + source.code);
      label.append(dot, name);
      row.append(label, cell(matches ? Number(source.elo).toLocaleString(undefined, { maximumFractionDigits: 1, minimumFractionDigits: 1 }) : 'Unrated', 'numeric elo-cell'));
      row.append(cell(number(matches), 'numeric match-cell'));
      for (const key of ['wins', 'ties', 'losses']) row.append(cell(number(source[key]), 'numeric outcome-cell'));
      fragment.append(row);
    }
    $('ranking-rows').replaceChildren(fragment);
    $('ranking-votes').textContent = number(data.stats?.votes);
    const provenance = [];
    const participants = Number(data.stats?.participants);
    const imported = Number(data.stats?.imported_votes);
    const excluded = Number(data.stats?.excluded_votes);
    if (Number.isFinite(participants) && participants > 0) provenance.push(`${number(participants)} ${participants === 1 ? 'rater' : 'raters'}`);
    if (Number.isFinite(imported) && imported > 0) provenance.push(`${number(imported)} imported`);
    if (Number.isFinite(excluded) && excluded > 0) provenance.push(`${number(excluded)} excluded from Elo`);
    $('ranking-provenance').textContent = provenance.join(' · ') + (Number.isFinite(excluded) && excluded > 0 ? '. Flagged imported comparisons are retained in exports.' : '');
    $('ranking-provenance').hidden = provenance.length === 0;
    $('pairs-covered').textContent = number(data.stats?.pairs_covered);
    $('total-pairs').textContent = number(data.stats?.total_pairs || 78);
    $('identity-state').textContent = 'Model names';
    $('ranking-description').textContent = data.revealed ? 'Model names are shown here. Comparison cards stay anonymous.' : 'Loading model names…';
    $('ranking-updated').textContent = `Updated ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`;
    renderCorrelations(data.correlations || []);
  }
  function renderCorrelations(correlations) {
    const fragment = document.createDocumentFragment();
    for (const correlation of correlations) {
      const card = document.createElement('article');
      card.className = 'correlation-card';
      const title = document.createElement('h4');
      title.textContent = correlation.judge || 'LLM judge';
      const value = document.createElement('div');
      value.className = 'correlation-value';
      const rho = correlation.rho_generators;
      const available = rho !== null && rho !== undefined && Number.isFinite(Number(rho));
      if (available) {
        const symbol = document.createElement('span');
        symbol.className = 'rho';
        symbol.textContent = 'ρ';
        symbol.setAttribute('aria-label', 'Spearman rho');
        const score = document.createElement('span');
        score.textContent = Number(rho).toFixed(3);
        value.append(symbol, score);
      } else { value.classList.add('empty'); value.textContent = 'Awaiting ratings'; }
      const detail = document.createElement('p');
      const n = correlation.n_generators;
      detail.textContent = `${number(n)} / 12 generators rated`;
      if (!available) detail.append(document.createElement('br'), document.createTextNode('At least 3 rated generators and variation needed'));
      if (correlation.rho !== null && correlation.rho !== undefined && Number.isFinite(Number(correlation.rho))) {
        detail.append(document.createElement('br'), document.createTextNode(`Including reference: ρ ${Number(correlation.rho).toFixed(3)} (${number(correlation.n)} sources)`));
      }
      if (correlation.warning) {
        const warning = document.createElement('span');
        warning.textContent = String(correlation.warning);
        detail.append(document.createElement('br'), warning);
      }
      card.append(title, value, detail);
      fragment.append(card);
    }
    if (!correlations.length) {
      const placeholder = document.createElement('p');
      placeholder.className = 'correlation-caution';
      placeholder.textContent = 'Judge correlations will appear as ratings become available.';
      fragment.append(placeholder);
    }
    $('correlation-cards').replaceChildren(fragment);
  }
  function applyState(data, full = false) {
    // A pending choice owns the current comparison until its matching acknowledgement.
    if (full && (pending || storageChanged)) return;
    state = full ? data : { ...state, ...data, participant: data.participant || state?.participant, pair: state?.pair, study: data.study || state?.study };
    $('your-votes').textContent = number(state.participant?.votes);
    $('total-votes').textContent = number(state.stats?.votes);
    $('rankings-tab').hidden = session?.mode === 'shared' && !state.participant?.finished;
    $('finish-button').hidden = Boolean(state.participant?.finished);
    if (full) {
      needsReload = false;
      const pair = state.pair;
      for (const side of ['a', 'b']) {
        $(`sample-${side}`).textContent = pair?.[side]?.text ? pair[side].text + (pair[side].truncated ? ' …' : '') : 'Thank you. Your voting session is complete and your choices are saved.';
        $(`sample-${side}-count`).textContent = pair ? `${pair[side].word_count ?? state.study?.excerpt_words ?? excerptWords} WORDS` : '';
      }
      for (const body of document.querySelectorAll('.sample-body')) body.scrollTop = 0;
      $('excerpt-size').textContent = state.study?.excerpt_words ?? excerptWords;
      $('source-count').textContent = state.study?.source_count ?? 13;
      $('available-samples').textContent = `${number(state.study?.sample_count || 13312)} samples in the study`;
      if (!pair) showSaved('You’ve completed the available comparisons. Thank you.');
    }
    renderRankings(state);
    setConnection(true);
    syncControls();
  }
  function configureSession(data) {
    if (!data || !['local', 'shared'].includes(data.mode)) throw new RecoveryError('The study session could not be verified. Choose Try again.');
    if (data.mode === 'shared') {
      if (typeof data.participant_id !== 'string' || !data.participant_id) throw new RecoveryError('The anonymous session is missing. Enable cookies for this site, then choose Try again.');
      participantId = data.participant_id;
    } else {
      participantId = storage.get(PARTICIPANT_KEY);
      if (!participantId) {
        participantId = pending?.participant_id || createId();
        storage.set(PARTICIPANT_KEY, participantId);
      }
    }
    session = data;
    const canExport = data.mode === 'local' && data.can_export === true;
    $('export-links').hidden = !canExport;
    for (const format of ['json', 'csv']) {
      const link = $('export-' + format);
      if (canExport) link.href = `/api/export?format=${format}&${query()}`;
      else link.removeAttribute('href');
    }
    $('session-description').textContent = 'Anonymous feedback · no name needed';
    if (pending && pending.participant_id !== participantId) {
      throw new RecoveryError('Your pending choice belongs to a different anonymous session and has been kept in this browser. Restore the original session cookie, then choose Try again, or ask the study owner for recovery. Do not clear browser storage.');
    }
  }
  async function submitPending() {
    const vote = { ...pending };
    setStatus('Sending your pending choice…');
    const data = await request('/api/vote', vote);
    if (data.acknowledged_request_id !== vote.request_id) {
      throw new RecoveryError('The server did not confirm this choice’s request ID. The pending choice has been retained. Choose Try again after the study server is updated.');
    }
    await withOutboxLock(() => {
      const stored = readPending();
      if (stored && !sameVote(stored, vote)) throw new RecoveryError('The server confirmed a choice, but another pending choice is stored in this browser. Keep this tab open and ask the study owner for recovery.');
      storage.remove(OUTBOX_KEY);
      pending = null;
    });
    retryAttempt = 0;
    needsReload = false;
    applyState(data, true);
    const messages = { a: 'Saved · A preferred. Here’s the next pair.', b: 'Saved · B preferred. Here’s the next pair.', skip: 'Skip saved. No rating recorded.' };
    showSaved(messages[vote.choice]);
  }
  function finishWork() {
    busy = false;
    syncControls();
    if (storageChanged) {
      storageChanged = false;
      resume();
    }
  }
  async function resume(resetRetries = false) {
    if (busy) return;
    busy = true;
    cancelRetry();
    if (resetRetries) retryAttempt = 0;
    ++revision;
    syncControls();
    try {
      storage.verify();
      storageReady = true;
      await withOutboxLock(restorePending);
      showPending();
      // The session is always the first network request, including every retry.
      const sessionData = await request('/api/session');
      configureSession(sessionData);
      if (pending) await submitPending();
      else {
        const data = await request(`/api/state?${query()}`);
        applyState(data, true);
        setStatus('Your choices are saved as you go.');
      }
      setError(null);
      setConnection(true);
    } catch (error) {
      if (error instanceof StorageError) storageReady = false;
      needsReload = true;
      const retained = pending && !(error instanceof RecoveryError) ? ' Your pending choice is retained in this browser; retrying will send that same choice.' : '';
      setError(new Error(error.message + retained));
      if (pending) setStatus('Choice pending · the next comparison is locked until confirmation.');
      else if (!storageReady) setStatus('Voting is paused until browser storage is available.');
      else setStatus('Reconnect to continue.');
      scheduleRetry(error);
    } finally { finishWork(); }
  }
  async function vote(choice) {
    if (busy || !storageReady || !session || needsReload || pending || !state?.pair) return;
    busy = true;
    ++revision;
    syncControls();
    let queued = false;
    try {
      await withOutboxLock(() => {
        storage.verify();
        const stored = readPending();
        if (stored) { pending = stored; return; }
        const candidate = { participant_id: participantId, pair_id: state.pair.id, choice, request_id: createId() };
        // No network request is allowed before this exact payload is durably stored.
        storage.set(OUTBOX_KEY, JSON.stringify(candidate));
        pending = candidate;
      });
      queued = true;
      showPending();
    } catch (error) {
      if (error instanceof StorageError) storageReady = false;
      needsReload = true;
      setError(error);
      setStatus('Choice not submitted. Restore browser storage before continuing.');
    } finally { finishWork(); }
    if (queued) await resume(true);
  }
  async function undo() {
    if (busy || pending || needsReload || !storageReady || !session || !state?.participant?.can_undo) return;
    busy = true;
    ++revision;
    syncControls();
    setStatus('Undoing your last choice…');
    try {
      await withOutboxLock(restorePending);
      if (pending) throw new RecoveryError('A pending choice must be confirmed before undoing a choice. Choose Try again to recover it.');
      const expectedParticipant = participantId;
      configureSession(await request('/api/session'));
      if (participantId !== expectedParticipant) throw new RecoveryError('The anonymous session changed. Choose Try again before making another choice.');
      const data = await request('/api/undo', { participant_id: participantId, vote_id: state.participant.last_vote_id });
      applyState(data, true);
      setError(null);
      showSaved('Last choice undone. Rankings have been recalculated.');
    } catch (error) {
      if (error instanceof StorageError) storageReady = false;
      needsReload = true;
      setError(error);
      setStatus('Undo not confirmed. Choose Try again to check the saved state.');
    } finally { finishWork(); }
  }
  async function pollRankings() {
    if (session?.mode === 'shared' && !state?.participant?.finished) return;
    if (busy || pending || needsReload || !session || pollRunning || activeView !== 'rankings' || document.hidden || !state) return;
    pollRunning = true;
    const token = revision;
    try {
      // Opening rankings reveals model names; comparison payloads omit source identity.
      const data = state.revealed
        ? await request(`/api/rankings?${query()}`)
        : await request('/api/reveal', { participant_id: participantId });
      if (token !== revision || busy || pending) return;
      applyState(data, false);
      setError(null);
    } catch (error) { if (token === revision && !busy && !pending) setError(error); }
    finally { pollRunning = false; }
  }
  function selectView(view, focus = false) {
    if (view === 'rankings' && session?.mode === 'shared' && !state?.participant?.finished) return;
    activeView = view;
    for (const item of ['comparison', 'rankings']) {
      const selected = item === view;
      $(item + '-view').hidden = !selected;
      $(item + '-tab').classList.toggle('active', selected);
      $(item + '-tab').setAttribute('aria-selected', String(selected));
      $(item + '-tab').tabIndex = selected ? 0 : -1;
    }
    if (focus) $(view + '-tab').focus();
    if (view === 'rankings') pollRankings();
  }
  $('comparison-tab').addEventListener('click', () => selectView('comparison'));
  $('rankings-tab').addEventListener('click', () => selectView('rankings'));
  document.querySelector('.tabs').addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    selectView(event.key === 'Home' ? 'comparison' : event.key === 'End' ? 'rankings' : activeView === 'comparison' ? 'rankings' : 'comparison', true);
  });
  voteButtons.forEach(button => button.addEventListener('click', () => vote(button.dataset.choice)));
  $('skip-button').addEventListener('click', () => vote('skip'));
  $('undo-button').addEventListener('click', undo);
  $('finish-button').addEventListener('click', async () => {
    if (busy || pending || needsReload || !state?.participant?.can_finish) return;
    busy = true; ++revision; syncControls();
    try {
      await withOutboxLock(restorePending);
      if (pending) throw new RecoveryError('Confirm your pending choice before finishing.');
      const data = await request('/api/finish', {});
      applyState(data, true); setError(null); showSaved('Thank you. All your choices are saved.');
    } catch (error) { needsReload = true; setError(error); }
    finally { finishWork(); }
    if (state?.participant?.finished) selectView('rankings');
  });
  $('retry-button').addEventListener('click', () => resume(true));
  for (const format of ['json', 'csv']) $('export-' + format).addEventListener('click', event => {
    if (session?.mode !== 'local' || !session.can_export) { event.preventDefault(); return; }
    setTimeout(() => { if (!busy && !pending) resume(); }, 700);
  });
  document.addEventListener('keydown', event => {
    if (event.defaultPrevented || event.repeat || event.ctrlKey || event.metaKey || event.altKey || activeView !== 'comparison') return;
    if (event.target instanceof Element && event.target.closest('input,textarea,select,[contenteditable]:not([contenteditable="false"])')) return;
    const choices = { '1': 'a', '2': 'b', 'a': 'a', 'b': 'b' };
    if (choices[event.key]) { event.preventDefault(); vote(choices[event.key]); }
  });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) pollRankings(); });
  window.addEventListener('online', () => { if (!busy) resume(true); });
  window.addEventListener('storage', event => {
    if (![OUTBOX_KEY, PARTICIPANT_KEY, null].includes(event.key)) return;
    // Reconcile another tab even if its storage event arrives during a request.
    storageChanged = true;
    needsReload = true;
    syncControls();
    if (!busy) { storageChanged = false; resume(); }
  });
  setInterval(pollRankings, 3000);
  resume(true);
})();
