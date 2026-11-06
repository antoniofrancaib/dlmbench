import {createHash,randomBytes} from 'node:crypto';
export const sha = x => createHash('sha256').update(x).digest('hex');
const sorted = x => Array.isArray(x) ? x.map(sorted) : x && typeof x==='object' ? Object.fromEntries(Object.keys(x).sort().map(k=>[k,sorted(x[k])])) : x;
export const canonical = x => JSON.stringify(sorted(x));
export const id = () => randomBytes(16).toString('hex');
export class StudyError extends Error {constructor(message,status=400){super(message);this.status=status;}}
export const choices={a:1,b:0,both_good:.5,both_bad:.5,skip:null};
// Preserve original byte strings, including Python's numeric representation.
export function verifyRaw(raw,sequence,previous,studyId) {
  const e=JSON.parse(raw);
  const without=raw.trimEnd().replace(/,"event_sha256":"[a-f0-9]{64}"/,'');
  if(e.sequence!==sequence || e.previous_sha256!==previous || e.study_id!==studyId || e.event_sha256!==sha(without)) throw Error('Invalid event hash chain at '+sequence);
  return e;
}
function rng(seed) {
  let [a,b,c,d]=seed.match(/.{8}/g).map(x=>parseInt(x,16)>>>0);
  const next=()=>{let t=(b<<9)>>>0;let v=Math.imul(b,5);v=Math.imul((v<<7)|(v>>>25),9)>>>0;c^=a;d^=b;b^=c;a^=d;c^=t;d=((d<<11)|(d>>>21))>>>0;return v;};
  return n=>{const limit=Math.floor(4294967296/n)*n;let v;do{v=next();}while(v>=limit);return v%n;};
}
function ranks(a){return a.map(x=>1+a.filter(y=>y<x).length+(a.filter(y=>y===x).length-1)/2);}
function rho(a,b){if(a.length<3)return null;const x=ranks(a),y=ranks(b),mean=(a.length+1)/2;const xx=x.reduce((s,v)=>s+(v-mean)**2,0),yy=y.reduce((s,v)=>s+(v-mean)**2,0);return xx&&yy?x.reduce((s,v,i)=>s+(v-mean)*(y[i]-mean),0)/Math.sqrt(xx*yy):null;}

export class Engine {
  constructor(dataset,raws=[],persist=()=>{}) {
    this.dataset=dataset;this.spec=dataset.spec;this.cohort=this.spec.cohort||'hosted-2026-11';this.banks=dataset.banks;this.ids=Object.keys(this.banks).sort();
    this.events=[];this.raws=[];this.previous=null;this.participants=new Map();this.pairs=new Map();this.votes=new Map();this.requests=new Map();this.imports=[];this.codes={};this.persist=persist;
    for(const raw of raws){const e=verifyRaw(raw,this.events.length+1,this.previous,this.spec.id);this.apply(e);this.events.push(e);this.raws.push(raw);this.previous=sha(raw);}
    if(raws.length && this.events[0].payload.spec_sha256!==dataset.specSha)throw Error('Frozen study hash mismatch');
  }
  append(type,payload) {
    const e={sequence:this.events.length+1,id:id(),study_id:this.spec.id,at:new Date().toISOString(),type,payload,previous_sha256:this.previous};
    e.event_sha256=sha(canonical(e));const raw=canonical(e)+'\n';
    this.persist(e,raw,sha(raw));this.apply(e);this.events.push(e);this.raws.push(raw);this.previous=sha(raw);return e;
  }
  apply(e){const p=e.payload;switch(e.type){
    case 'study_started':this.codes=p.codes;break;
    case 'participant':this.participants.set(p.id,{...this.participants.get(p.id),id:p.id,name:p.name,revealed:this.participants.get(p.id)?.revealed||false});break;
    case 'pair_assigned':this.pairs.set(p.id,{...p,status:'pending',assigned_at:e.at});break;
    case 'vote':this.votes.set(e.id,{...p,id:e.id,at:e.at,sequence:e.sequence,undone:false,response_policy:p.response_policy||'four-outcome-v1'});this.pairs.get(p.pair_id).status='answered';if(p.request_id)this.requests.set(p.request_id,{participant_id:p.participant_id,pair_id:p.pair_id,choice:p.choice});break;
    case 'request_acknowledged':this.requests.set(p.request_id,{participant_id:p.participant_id,pair_id:p.pair_id,choice:p.choice});break;
    case 'undo':{const v=this.votes.get(p.vote_id);v.undone=true;for(const pair of this.pairs.values())if(pair.participant_id===p.participant_id&&pair.status==='pending')pair.status='superseded';this.pairs.get(v.pair_id).status='pending';break;}
    case 'reveal':case 'named_export':this.participants.get(p.participant_id).revealed=true;break;
    case 'participant_finished':Object.assign(this.participants.get(p.participant_id),{finished:true,revealed:true,finished_at:e.at});break;
    case 'external_import':this.imports.push(p);this.participants.set(p.participant.id,{...p.participant,revealed:false,origin:'external_csv'});p.records.forEach((r,i)=>{this.pairs.set(r.pair.id,{...r.pair,status:'answered'});this.votes.set(r.vote.id,{...r.vote,at:e.at,sequence:e.sequence,sequence_in_event:i+1,undone:false,timestamp_meaning:'import_time'});});break;
    case 'response_policy_changed':this.responsePolicy=p.policy;break;
    case 'hosting_migrated':break;
    default:throw Error('Unknown journal event '+e.type);
  }}
  participant(pid){if(!/^anon_[a-f0-9]{32}$/.test(pid))throw new StudyError('Invalid session.',401);if(!this.participants.has(pid))this.append('participant',{id:pid,name:'Anonymous',cohort:this.cohort});return this.participants.get(pid);}
  active(pid){return [...this.votes.values()].filter(v=>!v.undone&&(!pid||v.participant_id===pid));}
  next(pid){
    if(this.participants.get(pid).finished)return null;
    const past=[...this.pairs.values()].filter(p=>p.participant_id===pid),pending=past.filter(p=>p.status==='pending');if(pending.length)return pending.at(-1);
    const pc=new Map(),sc=Object.fromEntries(this.ids.map(i=>[i,0])),seen=Object.fromEntries(this.ids.map(i=>[i,new Map()]));
    for(const p of past){let key=[p.a.bank_id,p.b.bank_id].sort().join('|');pc.set(key,(pc.get(key)||0)+1);for(const s of [p.a,p.b]){sc[s.bank_id]++;seen[s.bank_id].set(s.index,(seen[s.bank_id].get(s.index)||0)+1);}}
    let options=[],lowest=[Infinity,Infinity];for(let i=0;i<this.ids.length;i++)for(let j=i+1;j<this.ids.length;j++){const a=this.ids[i],b=this.ids[j],priority=[pc.get(a+'|'+b)||0,sc[a]+sc[b]];if(priority[0]<lowest[0]||(priority[0]===lowest[0]&&priority[1]<lowest[1])){lowest=priority;options=[];}if(priority[0]===lowest[0]&&priority[1]===lowest[1])options.push([a,b]);}
    const seed=id(),random=rng(seed),selected=[...options[random(options.length)]];if(random(2))selected.reverse();
    const sides=selected.map(bank=>{const samples=this.banks[bank].samples,min=Math.min(...samples.map(s=>seen[bank].get(s.index)||0)),eligible=samples.filter(s=>(seen[bank].get(s.index)||0)===min);return {...eligible[random(eligible.length)],bank_id:bank,bank_sha256:this.banks[bank].source_sha256};});
    const p={id:id(),participant_id:pid,seed,sampler_version:'balanced-xoshiro128ss-v1',a:sides[0],b:sides[1],revealed_at_assignment:false,excerpt_words:this.spec.excerpt_words,response_policy:'binary-v1',cohort:this.cohort};
    this.append('pair_assigned',p);return this.pairs.get(p.id);
  }
  leaderboard(){
    const rows=Object.fromEntries(this.ids.map(id=>[id,{id,code:this.codes[id],label:this.banks[id].label,elo:this.spec.elo.initial,matches:0,wins:0,ties:0,losses:0}])),pairs=new Set();let count=0;
    for(const v of this.active()){const s=choices[v.choice];if(s==null||v.include_in_elo===false)continue;const p=this.pairs.get(v.pair_id),a=rows[p.a.bank_id],b=rows[p.b.bank_id],expected=1/(1+10**Math.max(-300,Math.min(300,(b.elo-a.elo)/400))),delta=this.spec.elo.k*(s-expected);a.elo+=delta;b.elo-=delta;for(const [r,score]of [[a,s],[b,1-s]]){r.matches++;r[score===1?'wins':score===0?'losses':'ties']++;}pairs.add([a.id,b.id].sort().join('|'));count++;}
    const rankings=Object.values(rows).sort((a,b)=>(!a.matches)-(!b.matches)||b.elo-a.elo||a.code.localeCompare(b.code));let last=null,rank=null;rankings.forEach((r,i)=>{if(r.elo!==last)rank=i+1;r.rank=r.matches?rank:null;last=r.elo;});
    const recorded=this.active().filter(v=>v.choice!=='skip');
    return {rankings,stats:{recorded_votes:recorded.length,excluded_votes:recorded.filter(v=>v.include_in_elo===false).length,imported_votes:recorded.filter(v=>v.origin==='external_csv').length,votes:count,pairs_covered:pairs.size,total_pairs:this.ids.length*(this.ids.length-1)/2,participants:new Set(recorded.map(v=>v.participant_id)).size,sources_rated:rankings.filter(r=>r.matches).length,min_matches:Math.min(...rankings.map(r=>r.matches))}};
  }
  state(pid,includePair=true){
    const p=this.participant(pid),pair=includePair?this.next(pid):null,board=this.leaderboard(),active=this.active(pid),last=active.at(-1),rated=board.rankings.filter(r=>r.matches),generated=rated.filter(r=>r.id!==this.spec.reference_id);
    const result={study:{id:this.spec.id,title:this.spec.title,excerpt_words:this.spec.excerpt_words,source_count:this.ids.length,sample_count:this.spec.sample_count,response_policy:'binary-v1'},participant:{id:pid,votes:active.filter(v=>v.choice!=='skip').length,can_undo:!p.finished&&!!last&&last.origin!=='external_csv',can_finish:active.length>0,finished:!!p.finished},stats:board.stats,revealed:!!p.finished,rankings:p.finished?board.rankings.map(({id,...r})=>r):[],correlations:p.finished?Object.keys(Object.values(this.spec.llm_ratings)[0]).map(judge=>({judge,rho:rho(rated.map(r=>r.elo),rated.map(r=>this.spec.llm_ratings[r.id][judge])),n:rated.length,rho_generators:rho(generated.map(r=>r.elo),generated.map(r=>this.spec.llm_ratings[r.id][judge])),n_generators:generated.length,warning:this.spec.llm_source.caveat})):[]};
    result.participant.last_vote_id=last?.id||null;
    if(includePair)result.pair=pair?{id:pair.id,...Object.fromEntries(['a','b'].map(s=>[s,Object.fromEntries(['text','word_count','truncated'].map(k=>[k,pair[s][k]]))]))}:null;
    return result;
  }
  vote(pid,body){
    const {pair_id,choice,request_id}=body;
    if(typeof request_id!=='string'||!/^[A-Za-z0-9_-]{1,128}$/.test(request_id))throw new StudyError('A request ID is required.');
    const prior=this.requests.get(request_id);
    if(prior){if(prior.participant_id!==pid||prior.pair_id!==pair_id||prior.choice!==choice)throw new StudyError('This request ID already belongs to another choice.',409);return {...this.state(pid),acknowledged_request_id:request_id};}
    const p=this.participant(pid);if(p.finished)throw new StudyError('This voting session is finished.',409);
    if(!['a','b','skip'].includes(choice))throw new StudyError('Choose A, B, or Skip.');
    const pair=this.pairs.get(pair_id);if(!pair||pair.participant_id!==pid)throw new StudyError('This comparison does not belong to your session.',403);
    if(pair.status!=='pending'){
      const old=this.active(pid).filter(v=>v.pair_id===pair_id).at(-1);if(!old||old.choice!==choice)throw new StudyError('This comparison was already answered.',409);
      this.append('request_acknowledged',{participant_id:pid,pair_id,choice,request_id,vote_id:old.id});
    }else this.append('vote',{participant_id:pid,participant_name:'Anonymous',pair_id,choice,score_a:choices[choice],request_id,response_policy:'binary-v1',elapsed_ms:Math.max(0,Date.now()-Date.parse(pair.assigned_at)),revealed_at_vote:false,cohort:this.cohort});
    return {...this.state(pid),acknowledged_request_id:request_id};
  }
  undo(pid,body){const p=this.participant(pid);if(p.finished)throw new StudyError('This session is finished.',409);const v=this.votes.get(body.vote_id);if(!v||v.participant_id!==pid||v.origin==='external_csv')throw new StudyError('Cannot undo this response.',409);if(v.undone)return this.state(pid);if(this.active(pid).at(-1)?.id!==v.id)throw new StudyError('Only the most recent response can be undone.',409);this.append('undo',{participant_id:pid,vote_id:v.id});return this.state(pid);}
  finish(pid){const p=this.participant(pid);if(!this.active(pid).length)throw new StudyError('Save at least one comparison before finishing.');if(!p.finished)this.append('participant_finished',{participant_id:pid,cohort:this.cohort});return this.state(pid);}
  export(){return {exported_at:new Date().toISOString(),study:this.spec,code_map:this.codes,current_response_policy:'binary-v1',...this.leaderboard(),events:this.events,event_raw:this.raws,event_chain_last_sha256:this.previous,votes:[...this.votes.values()],assignments:[...this.pairs.values()],imports:this.imports};}
}
