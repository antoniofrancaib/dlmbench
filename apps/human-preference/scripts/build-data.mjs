import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
const hash = x => createHash('sha256').update(x).digest('hex');
const raw = readFileSync(new URL('../study/study.json', import.meta.url));
const spec = JSON.parse(raw);
const banks = {};
for (const b of spec.banks) {
  const bytes = readFileSync(new URL('../study/'+b.file, import.meta.url));
  if (hash(bytes) !== b.sha256) throw Error('Bank hash mismatch: '+b.id);
  banks[b.id] = JSON.parse(bytes);
  if (banks[b.id].samples.length !== b.count) throw Error('Sample count mismatch');
  for (const s of banks[b.id].samples) if(hash(s.text)!==s.excerpt_sha256) throw Error('Excerpt mismatch');
}
mkdirSync(new URL('../generated/',import.meta.url),{recursive:true});
writeFileSync(new URL('../generated/dataset.js',import.meta.url),'export default '+JSON.stringify({spec,specSha:hash(raw),banks})+';\n');
const assets=Object.fromEntries(['index.html','app.js','app.css','admin.html','admin.js'].map(name=>['/'+name,readFileSync(new URL('../static/'+name,import.meta.url),'utf8')]));
writeFileSync(new URL('../generated/assets.js',import.meta.url),'export default '+JSON.stringify(assets)+';\n');
console.log('Verified 13 banks and 13,312 excerpt hashes.');
