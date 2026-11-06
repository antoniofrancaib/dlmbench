import test from 'node:test';
import assert from 'node:assert/strict';
import {loadStudy} from '../scripts/study-data.mjs';
import {Engine,id} from '../src/engine.js';
const old=loadStudy(new URL('../study',import.meta.url).pathname),long=loadStudy(new URL('../study-100',import.meta.url).pathname);
function seed(data){const e=new Engine(data);e.append('study_started',{codes:Object.fromEntries(e.ids.map((id,i)=>[id,String(i+1)])),spec_sha256:data.specSha});e.append('response_policy_changed',{policy:'binary-v1'});return e;}
test('100-word study preserves every original sample identity, full-text hash and 35-word prefix',()=>{
 assert.notEqual(old.spec.id,long.spec.id);assert.equal(long.spec.excerpt_words,100);assert.deepEqual(Object.keys(old.banks),Object.keys(long.banks));
 let shorter=0;
 for(const [id,b] of Object.entries(long.banks)){
  assert.equal(b.source_sha256,old.banks[id].source_sha256);
  for(const [i,s] of b.samples.entries()){
   const previous=old.banks[id].samples[i];
   for(const k of ['index','sample_id','full_text_sha256'])assert.equal(s[k],previous[k]);
   assert.equal(s.text.split(' ').slice(0,35).join(' '),previous.text);
   assert.ok(s.word_count<=100);if(s.word_count<100){shorter++;assert.equal(s.truncated,false);}
  }
 }
 assert.equal(shorter,28);assert.equal(long.spec.sample_count,13312);
});
test('collections reject one another’s journal and record the correct length and cohort',()=>{
 const a=seed(old),b=seed(long);assert.throws(()=>new Engine(long,a.raws));assert.throws(()=>new Engine(old,b.raws));
 for(const e of [a,b]){
  const pid='anon_'+id(),s=e.state(pid),p=e.pairs.get(s.pair.id);
  assert.equal(s.study.excerpt_words,e.spec.excerpt_words);assert.equal(p.excerpt_words,e.spec.excerpt_words);assert.equal(p.cohort,e.cohort);
  assert.equal(e.leaderboard().stats.votes,0);e.vote(pid,{pair_id:p.id,choice:'a',request_id:id()});
  assert.equal(e.active(pid)[0].cohort,e.cohort);assert.equal(e.finish(pid).rankings.length,13);
 }
 assert.equal(a.active()[0].cohort,'hosted-2026-11');assert.equal(b.active()[0].cohort,'hosted-prefix100-2026-11');
});
