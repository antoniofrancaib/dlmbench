import {writeFileSync} from 'node:fs';
import {randomInt} from 'node:crypto';
import {loadStudy} from './study-data.mjs';
import {Engine} from '../src/engine.js';
const [directory,output]=process.argv.slice(2);
if(!directory||!output)throw Error('Usage: node scripts/seed-study.mjs STUDY_DIRECTORY OUTPUT_JSON');
const data=loadStudy(directory),e=new Engine(data),ids=[...e.ids];
for(let i=ids.length-1;i>0;i--){const j=randomInt(i+1);[ids[i],ids[j]]=[ids[j],ids[i]];}
e.append('study_started',{spec_sha256:data.specSha,codes:Object.fromEntries(ids.map((id,i)=>[id,String(i+1).padStart(2,'0')])),cohort:data.spec.cohort});
e.append('response_policy_changed',{policy:'binary-v1'});
writeFileSync(output,JSON.stringify({event_raw:e.raws},null,2)+'\n',{flag:'wx',mode:0o600});
console.log('Created empty collection seed for '+data.spec.id);
