import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
const hash = x => createHash('sha256').update(x).digest('hex');
export function loadStudy(directory) {
  const raw=readFileSync(resolve(directory,'study.json')),spec=JSON.parse(raw),banks={};
  for(const b of spec.banks){
    const bytes=readFileSync(resolve(directory,b.file));
    if(hash(bytes)!==b.sha256)throw Error('Bank hash mismatch: '+b.id);
    const bank=banks[b.id]=JSON.parse(bytes);
    if(bank.id!==b.id||bank.source_sha256!==b.source_sha256||bank.samples.length!==b.count)throw Error('Bank identity or count mismatch');
    for(const s of bank.samples){
      if(hash(s.text)!==s.excerpt_sha256)throw Error('Excerpt mismatch');
      if(s.word_count!==s.text.split(' ').length||s.word_count>spec.excerpt_words)throw Error('Excerpt length mismatch');
    }
  }
  if(Object.keys(banks).length!==spec.source_count||Object.values(banks).reduce((n,b)=>n+b.samples.length,0)!==spec.sample_count)throw Error('Study counts mismatch');
  return {spec,specSha:hash(raw),banks};
}
