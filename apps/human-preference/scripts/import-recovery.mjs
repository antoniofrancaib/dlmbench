import {readFileSync,readdirSync} from 'node:fs';
const [url,secretsFile,recovery]=process.argv.slice(2);
if(!url||!secretsFile||!recovery)throw Error('Usage: node scripts/import-recovery.mjs URL SECRET_JSON RECOVERY_DIRECTORY');
const token=JSON.parse(readFileSync(secretsFile)).ADMIN_TOKEN;
const root=recovery+'/Human-Preference/data/Events';
const event_raw=readdirSync(root).filter(n=>n.endsWith('.json')).sort().map(n=>readFileSync(root+'/'+n,'utf8'));
const r=await fetch(url+'/api/admin/import',{method:'POST',headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},body:JSON.stringify({event_raw})});
const data=await r.json();if(!r.ok)throw Error(JSON.stringify(data));
console.log(JSON.stringify({events:data.events,responses:data.responses,stats:data.stats,pending_backups:data.pending_backups,last_event_sha256:data.last_event_sha256}));
