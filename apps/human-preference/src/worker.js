import {DurableObject} from 'cloudflare:workers';
import {createHmac,timingSafeEqual,createHash} from 'node:crypto';
import dataset from '../generated/dataset.js';
import assets from '../generated/assets.js';
import {Engine,StudyError,sha,id} from './engine.js';

function equal(a,b){return timingSafeEqual(createHash('sha256').update(String(a)).digest(),createHash('sha256').update(String(b)).digest());}
const headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff','X-Robots-Tag':'noindex, nofollow','Referrer-Policy':'no-referrer','Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"};
function response(data,status=200,extra={}){return new Response(typeof data==='string'?data:JSON.stringify(data),{status,headers:{...headers,'Content-Type':'application/json; charset=utf-8',...extra}});}
function signature(pid,secret){return createHmac('sha256',secret).update(pid).digest('hex');}
function identity(request,env){const cookie=request.headers.get('Cookie')||'',value=cookie.split(';').map(s=>s.trim()).find(s=>s.startsWith('__Host-owt-session='))?.split('=').slice(1).join('=');if(!value)return null;const [pid,sig]=value.split('.');return /^anon_[a-f0-9]{32}$/.test(pid)&&sig&&equal(sig,signature(pid,env.SESSION_SECRET))?pid:null;}
function csv(exported){
  const base=['study_id','vote_id','sequence','at','participant_id','participant_name','pair_id','request_id','choice','score_a','undone','revealed_at_vote','elapsed_ms','excerpt_words','response_policy','origin','include_in_elo','exclusion_reason','import_id','import_row','original_round','blinding_status','judged_at','timestamp_meaning','exact_packet_match','data_quality_flags','cohort'];
  const sideFields=['bank_id','bank_sha256','sample_id','index','full_text_sha256','excerpt_sha256','word_count','truncated','text','exact_packet_match','flags','expected_packet_sha256','mapped_canonical_source'];
  const cols=[...base,...['a','b'].flatMap(side=>sideFields.map(k=>side+'_'+k))],pairs=new Map(exported.assignments.map(p=>[p.id,p]));
  const quote=v=>'"'+(v==null?'':typeof v==='object'?JSON.stringify(v):String(v)).replaceAll('"','""')+'"';
  return [cols.join(','),...exported.votes.map(v=>{const p=pairs.get(v.pair_id),row={...v,study_id:exported.study.id,vote_id:v.id,excerpt_words:p.excerpt_words,origin:v.origin||'live',include_in_elo:v.include_in_elo!==false,timestamp_meaning:v.timestamp_meaning||'response_recorded_time'};for(const s of ['a','b'])for(const k of sideFields)row[s+'_'+k]=p[s][k];return cols.map(k=>quote(row[k])).join(',');})].join('\r\n')+'\r\n';
}

export class PreferenceStudy extends DurableObject {
  constructor(ctx,env){
    super(ctx,env);this.env=env;this.ctx=ctx;this.queue=Promise.resolve();
    ctx.storage.sql.exec('CREATE TABLE IF NOT EXISTS events(sequence INTEGER PRIMARY KEY,id TEXT UNIQUE NOT NULL,raw TEXT NOT NULL,raw_sha TEXT NOT NULL,mirrored INTEGER NOT NULL DEFAULT 0)');
    ctx.storage.sql.exec('CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)');
    this.reload();
  }
  reload(){const raws=this.ctx.storage.sql.exec('SELECT raw FROM events ORDER BY sequence').toArray().map(r=>r.raw);this.engine=new Engine(dataset,raws,(e,raw,digest)=>this.ctx.storage.sql.exec('INSERT INTO events(sequence,id,raw,raw_sha) VALUES(?,?,?,?)',e.sequence,e.id,raw,digest));}
  ready(){return this.ctx.storage.sql.exec("SELECT value FROM meta WHERE key='ready'").toArray()[0]?.value==='true';}
  async mirror(){
    for(;;){const rows=this.ctx.storage.sql.exec('SELECT sequence,id,raw,raw_sha FROM events WHERE mirrored=0 ORDER BY sequence LIMIT 40').toArray();if(!rows.length)return;
      await this.env.BACKUP.batch(rows.map(r=>this.env.BACKUP.prepare('INSERT INTO journal(study_key,sequence,event_id,raw,raw_sha) VALUES(?,?,?,?,?) ON CONFLICT(study_key,sequence) DO UPDATE SET raw=CASE WHEN journal.raw_sha=excluded.raw_sha THEN journal.raw ELSE NULL END').bind(this.env.STUDY_KEY,r.sequence,r.id,r.raw,r.raw_sha)));
      this.ctx.storage.transactionSync(()=>{for(const r of rows)this.ctx.storage.sql.exec('UPDATE events SET mirrored=1 WHERE sequence=? AND raw_sha=?',r.sequence,r.raw_sha);});
    }
  }
  fetch(request){const task=this.queue.then(()=>this.handle(request));this.queue=task.catch(()=>{});return task;}
  async handle(request){
    try{
      const url=new URL(request.url),path=url.pathname,pid=request.headers.get('X-Study-Participant');
      if(path==='/api/admin/import'){
        const body=await request.json();if(!Array.isArray(body.event_raw)||!body.event_raw.length)throw new StudyError('Original event bytes are required.');
        const incoming=new Engine(dataset,body.event_raw); // Verify entire chain before any write.
        const existing=this.engine.raws;if(existing.some((raw,i)=>incoming.raws[i]!==raw))throw new StudyError('Import would replace or omit existing events.',409);
        if(this.ready()&&existing.length!==incoming.raws.length)throw new StudyError('This collection is already active.',409);
        this.ctx.storage.transactionSync(()=>{for(let i=existing.length;i<incoming.raws.length;i++){const e=incoming.events[i],raw=incoming.raws[i];this.ctx.storage.sql.exec('INSERT INTO events(sequence,id,raw,raw_sha) VALUES(?,?,?,?)',e.sequence,e.id,raw,sha(raw));}});
        this.reload();await this.mirror();this.ctx.storage.sql.exec("INSERT INTO meta VALUES('ready','true') ON CONFLICT(key) DO UPDATE SET value='true'");return response(this.summary());
      }
      if(!this.ready())return response({error:'The study is being prepared. Please return shortly.'},503);
      await this.mirror();
      if(path==='/api/health')return response({ok:true,samples:13312,study_id:dataset.spec.id});
      if(path==='/api/admin/stats')return response(this.summary());
      if(path==='/api/admin/export'){
        const exp=this.engine.export(),format=url.searchParams.get('format');return format==='csv'?response(csv(exp),200,{'Content-Type':'text/csv; charset=utf-8','Content-Disposition':'attachment; filename="owt-responses.csv"'}):response(exp,200,{'Content-Disposition':'attachment; filename="owt-full-export.json"'});
      }
      let result;
      if(path==='/api/state')result=this.engine.state(pid);
      else if(path==='/api/rankings'){if(!this.engine.participant(pid).finished)throw new StudyError('Finish your voting session to view rankings.',403);result=this.engine.state(pid,false);}
      else if(path==='/api/vote')result=this.engine.vote(pid,await request.json());
      else if(path==='/api/undo')result=this.engine.undo(pid,await request.json());
      else if(path==='/api/finish')result=this.engine.finish(pid);
      else throw new StudyError('Not found.',404);
      await this.mirror();return response(result);
    }catch(e){if(e instanceof StudyError)return response({error:e.message},e.status);return response({error:'Saving could not be confirmed. Your pending choice stays in this browser; please retry.'},503);}
  }
  summary(){return {...this.engine.leaderboard(),events:this.engine.events.length,responses:this.engine.votes.size,last_event_sha256:this.engine.previous,pending_backups:this.ctx.storage.sql.exec('SELECT COUNT(*) AS n FROM events WHERE mirrored=0').toArray()[0].n,ready:this.ready()};}
}

export default {
  async fetch(request,env){
    const url=new URL(request.url),path=url.pathname;
    if(url.origin!==env.PUBLIC_ORIGIN)return response({error:'Use the published study address.'},403);
    const origin=request.headers.get('Origin');if(origin&&origin!==url.origin)return response({error:'Cross-origin requests are not accepted.'},403);
    const asset=path==='/'?'/index.html':path==='/admin'?'/admin.html':path;
    if(request.method==='GET'&&Object.hasOwn(assets,asset))return response(assets[asset],200,{'Content-Type':asset.endsWith('.js')?'text/javascript; charset=utf-8':asset.endsWith('.css')?'text/css; charset=utf-8':'text/html; charset=utf-8'});
    if(path==='/robots.txt')return response('User-agent: *\nDisallow: /\n',200,{'Content-Type':'text/plain'});
    if(!env.SESSION_SECRET||!env.ADMIN_TOKEN)return response({error:'The study is being prepared.'},503);
    let pid=identity(request,env);
    if(path==='/api/session'&&request.method==='GET'){pid=pid||'anon_'+id();return response({mode:'shared',participant_id:pid,can_export:false},200,{'Set-Cookie':`__Host-owt-session=${pid}.${signature(pid,env.SESSION_SECRET)}; Path=/; Max-Age=31536000; Secure; HttpOnly; SameSite=Lax`});}
    const admin=path.startsWith('/api/admin/');
    if(admin){
      const token=request.headers.get('Authorization')||'';
      const owner=equal(token,'Bearer '+env.ADMIN_TOKEN),backup=env.BACKUP_TOKEN&&equal(token,'Bearer '+env.BACKUP_TOKEN)&&request.method==='GET'&&path==='/api/admin/export';
      if(!owner&&!backup)return response({error:'Owner access is required.'},401);
      if(!((request.method==='GET'&&['/api/admin/stats','/api/admin/export'].includes(path))||(request.method==='POST'&&path==='/api/admin/import')))return response({error:'Not found.'},404);
    }else{
      const get=request.method==='GET'&&['/api/state','/api/rankings','/api/health'].includes(path),post=request.method==='POST'&&['/api/vote','/api/undo','/api/finish'].includes(path);
      if(!get&&!post)return response({error:'Not found.'},404);
      if(post&&origin!==url.origin)return response({error:'The study origin is required.'},403);
      if(!pid&&path!=='/api/health')return response({error:'Reload to reconnect your anonymous session.'},401);
    }
    if(request.method==='POST'){
      if(!request.headers.get('Content-Type')?.startsWith('application/json'))return response({error:'JSON is required.'},400);
      const body=await request.text();if(body.length>(admin?20_000_000:8192))return response({error:'Request too large.'},413);
      try{const d=JSON.parse(body);if(!d||typeof d!=='object'||Array.isArray(d))throw Error();}catch{return response({error:'Invalid JSON.'},400);}
      request=new Request(request.url,{method:'POST',headers:request.headers,body});
    }
    const forwarded=new Request(request);forwarded.headers.delete('X-Study-Participant');if(pid)forwarded.headers.set('X-Study-Participant',pid);
    return env.STUDY.get(env.STUDY.idFromName(env.STUDY_KEY),{locationHint:'weur'}).fetch(forwarded);
  }
};
