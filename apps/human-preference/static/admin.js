'use strict';
const $=id=>document.getElementById(id);
const fragment=new URLSearchParams(location.hash.slice(1));
if(fragment.has('key')){sessionStorage.setItem('owt-owner-key',fragment.get('key'));history.replaceState(null,'',location.pathname);}
let key=sessionStorage.getItem('owt-owner-key')||'';
async function get(path){const r=await fetch(path,{headers:{Authorization:'Bearer '+key},cache:'no-store'});if(!r.ok)throw Error((await r.json()).error||'Request failed');return r;}
async function update(){try{const d=await(await get('/api/admin/stats')).json();$('login').hidden=true;$('dashboard').hidden=false;$('status').textContent=`${d.responses} saved responses · ${d.stats.votes} scored judgments · ${d.stats.participants} participant IDs · ${d.stats.pairs_covered}/78 pairs`;const rows=d.rankings.map(r=>{const tr=document.createElement('tr');for(const text of [r.rank,r.label,r.elo.toFixed(1),r.matches]){const td=document.createElement('td');td.textContent=text;tr.append(td);}return tr;});$('rows').replaceChildren(...rows);$('backup').textContent=d.pending_backups?`${d.pending_backups} events awaiting their second copy`:`All ${d.events} events have a verified second copy. Updated ${new Date().toLocaleTimeString()}.`;}catch(e){$('status').textContent=e.message;}}
$('login').addEventListener('submit',e=>{e.preventDefault();key=$('key').value;sessionStorage.setItem('owt-owner-key',key);$('key').value='';update();});
for(const format of ['json','csv'])$(format).addEventListener('click',async()=>{try{const r=await get('/api/admin/export?format='+format),u=URL.createObjectURL(await r.blob()),a=document.createElement('a');a.href=u;a.download='owt-human-preferences-'+new Date().toISOString().slice(0,10)+'.'+format;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}catch(e){$('status').textContent=e.message;}});
if(key)update();setInterval(()=>{if(key&&!document.hidden)update();},5000);
