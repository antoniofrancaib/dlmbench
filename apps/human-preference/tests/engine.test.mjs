import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,readdirSync} from 'node:fs';
import dataset from '../generated/dataset.js';
import {Engine,id,sha} from '../src/engine.js';
const pid=()=> 'anon_'+id();
function fresh(){const e=new Engine(dataset);e.append('study_started',{codes:Object.fromEntries(e.ids.map((id,i)=>[id,String(i+1).padStart(2,'0')])),spec_sha256:dataset.specSha});e.append('response_policy_changed',{policy:'binary-v1'});return e;}
test('balanced scheduling, anonymous responses and persisted replay',()=>{
 const e=fresh(),p=pid(),pairs=new Set(),seen=new Set();
 for(let i=0;i<78;i++){const s=e.state(p),raw=e.pairs.get(s.pair.id);pairs.add([raw.a.bank_id,raw.b.bank_id].sort().join('|'));for(const side of ['a','b']){assert.equal(s.pair[side].bank_id,undefined);const key=raw[side].bank_id+':'+raw[side].index;assert.ok(!seen.has(key));seen.add(key);}assert.deepEqual(s.rankings,[]);e.vote(p,{pair_id:s.pair.id,choice:i%2?'a':'b',request_id:id()});}
 assert.equal(pairs.size,78);const restored=new Engine(dataset,e.raws);assert.deepEqual(restored.export().votes,e.export().votes);assert.deepEqual(restored.leaderboard(),e.leaderboard());
});
test('request retries survive restart and undo without duplicate or revived votes',()=>{
 let e=fresh(),p=pid(),s=e.state(p),body={pair_id:s.pair.id,choice:'a',request_id:id()};e.vote(p,body);e=new Engine(dataset,e.raws);e.vote(p,body);assert.equal(e.votes.size,1);assert.throws(()=>e.vote(p,{...body,choice:'b'}));const vote_id=e.active(p)[0].id;e.undo(p,{vote_id});e.undo(p,{vote_id});e.vote(p,body);assert.equal(e.active(p).length,0);assert.equal(e.votes.size,1);e.vote(p,{...body,choice:'b',request_id:id()});assert.equal(e.active(p).length,1);
});
test('finish is idempotent, reveals rankings, and closes voting',()=>{
 const e=fresh(),p=pid(),s=e.state(p);assert.throws(()=>e.finish(p));e.vote(p,{pair_id:s.pair.id,choice:'a',request_id:id()});let done=e.finish(p);assert.equal(done.pair,null);assert.equal(done.rankings.length,13);assert.equal(done.participant.finished,true);const n=e.events.length;e.finish(p);assert.equal(e.events.length,n);assert.throws(()=>e.vote(p,{pair_id:s.pair.id,choice:'b',request_id:id()}));assert.throws(()=>e.undo(p,{vote_id:e.active(p)[0].id}));
});
test('sessions cannot submit or undo another participant response',()=>{
 const e=fresh(),a=pid(),b=pid(),s=e.state(a);e.state(b);assert.throws(()=>e.vote(b,{pair_id:s.pair.id,choice:'a',request_id:id()}));e.vote(a,{pair_id:s.pair.id,choice:'a',request_id:id()});assert.throws(()=>e.undo(b,{vote_id:e.active(a)[0].id}));
});
test('corruption and wrong study are rejected before replay',()=>{
 const e=fresh();assert.throws(()=>new Engine(dataset,[e.raws[0].replace('study_started','study_deleted')]));assert.throws(()=>new Engine({...dataset,spec:{...dataset.spec,id:'different'}},e.raws));
});
test('recovered collection has identical responses, exclusions and Elo', {skip:!process.env.RECOVERY_DIR},()=>{
 const root=process.env.RECOVERY_DIR,files=readdirSync(root+'/Human-Preference/data/Events').filter(x=>x.endsWith('.json')).sort(),raws=files.map(x=>readFileSync(root+'/Human-Preference/data/Events/'+x,'utf8'));
 const e=new Engine(dataset,raws),reference=JSON.parse(readFileSync(root+'/full-export.json'));
 assert.equal(e.events.length,561);assert.equal(e.votes.size,323);assert.equal(e.leaderboard().stats.votes,301);assert.equal(e.leaderboard().stats.excluded_votes,4);assert.equal(e.previous,reference.event_chain_last_sha256);
 for(const r of e.leaderboard().rankings){const old=reference.rankings.find(x=>x.id===r.id);assert.equal(r.rank,old.rank);assert.ok(Math.abs(r.elo-old.elo)<1e-10);assert.equal(r.matches,old.matches);}
 assert.deepEqual(raws,e.raws);assert.equal(sha(e.raws.at(-1)),reference.event_chain_last_sha256);
});
