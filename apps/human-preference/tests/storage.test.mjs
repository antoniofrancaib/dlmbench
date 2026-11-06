import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {Miniflare,convertV4MiniflareOptions} from 'miniflare';
import dataset from '../generated/dataset.js';
import {Engine,id} from '../src/engine.js';

test('two-copy acknowledgement survives backup outage and a full runtime restart',async()=>{
 const dir=await mkdtemp(join(tmpdir(),'owt-storage-test-'));
 const config={workers:[{name:'test-worker',modules:true,scriptPath:new URL('../dist/worker.js',import.meta.url).pathname,compatibilityDate:'2026-10-13',compatibilityFlags:['nodejs_compat'],durableObjects:{STUDY:{className:'PreferenceStudy',useSQLite:true}},d1Databases:{BACKUP:'test-backup'},bindings:{STUDY_ID:dataset.spec.id,STUDY_KEY:'test',PUBLIC_ORIGIN:'http://localhost',ADMIN_TOKEN:'owner',SESSION_SECRET:'session',BACKUP_TOKEN:'backup'}}],resourcePersistencePath:dir};
 let mf=new Miniflare(convertV4MiniflareOptions(config));
 const call=(path,init={})=>mf.dispatchFetch('http://localhost'+path,init);
 try{
  let db=await mf.getD1Database('BACKUP');
  await db.exec('CREATE TABLE journal(study_key TEXT NOT NULL,sequence INTEGER NOT NULL,event_id TEXT NOT NULL,raw TEXT NOT NULL,raw_sha TEXT NOT NULL,PRIMARY KEY(study_key,sequence));');
  const e=new Engine(dataset);e.append('study_started',{codes:Object.fromEntries(e.ids.map((b,i)=>[b,String(i+1)])),spec_sha256:dataset.specSha});e.append('response_policy_changed',{policy:'binary-v1'});
  const imported=await call('/api/admin/import',{method:'POST',headers:{Authorization:'Bearer owner','Content-Type':'application/json'},body:JSON.stringify({event_raw:e.raws})});assert.equal(imported.status,200,await imported.text());
  const session=await call('/api/session'),cookie=session.headers.get('Set-Cookie').split(';')[0];
  const state=await(await call('/api/state',{headers:{Cookie:cookie}})).json();
  const vote={pair_id:state.pair.id,choice:'a',request_id:id()},init={method:'POST',headers:{Cookie:cookie,Origin:'http://localhost','Content-Type':'application/json'},body:JSON.stringify(vote)};
  await db.exec('ALTER TABLE journal RENAME TO offline_journal');
  assert.equal((await call('/api/vote',init)).status,503);
  await db.exec('ALTER TABLE offline_journal RENAME TO journal');
  const saved=await(await call('/api/vote',init)).json();assert.equal(saved.participant.votes,1);assert.equal(saved.acknowledged_request_id,vote.request_id);
  const stats=await(await call('/api/admin/stats',{headers:{Authorization:'Bearer owner'}})).json();assert.equal(stats.responses,1);assert.equal(stats.pending_backups,0);
  await mf.dispose();mf=new Miniflare(convertV4MiniflareOptions(config));
  const replay=await(await call('/api/vote',init)).json();assert.equal(replay.participant.votes,1);assert.equal(replay.acknowledged_request_id,vote.request_id);assert.equal(replay.pair.id,saved.pair.id);
  const end=await(await call('/api/admin/stats',{headers:{Authorization:'Bearer owner'}})).json();assert.equal(end.responses,1);assert.equal(end.last_event_sha256,stats.last_event_sha256);
  db=await mf.getD1Database('BACKUP');const mirrored=await db.prepare('SELECT COUNT(*) AS n FROM journal').first();assert.equal(mirrored.n,end.events);
 }finally{await mf.dispose();await rm(dir,{recursive:true,force:true});}
});
