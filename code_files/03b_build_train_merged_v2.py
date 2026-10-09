"""Build v2 labels, adding the previously omitted train batch 001."""
from pathlib import Path
import subprocess, sys, re
import pandas as pd

P=Path(__file__).resolve().parents[1]; L=P/'data'/'labels'; R=P/'temp_code'/'labeller_out'; H=P/'temp_code'/'holdout_ids.csv'
SAFE={'government','govt','sarkari','sarakari','ganapathi','kalyana','overhead','children'}
VALID=re.compile(r'^(O|[BI]-(TOWN|LANDMARK|LOCALITY|RELATION|PINCODE))$')

def orphan(g):
    g=g.sort_values('word_index',key=lambda s:s.astype(int)).copy(); prev='O'
    for i,t in g.tag.items():
        if t.startswith('I-') and prev not in {f'B-{t[2:]}',f'I-{t[2:]}'}: g.at[i,'tag']='B-'+t[2:]
        prev=g.at[i,'tag']
    return g
def bio(g):
    p='O'
    for t in g.sort_values('word_index',key=lambda s:s.astype(int)).tag:
        if not VALID.fullmatch(t) or (t.startswith('I-') and p not in {f'B-{t[2:]}',f'I-{t[2:]}'}): return False
        p=t
    return True
def main():
    ref=pd.read_csv(L/'train_bio.csv',dtype=str).fillna(''); batch=pd.read_csv(P/'temp_code'/'labeller_batches'/'batch_001.csv',dtype=str).fillna('')
    # The existing merge tool produces the requested first-pass batch output.
    subprocess.run([sys.executable,str(P/'temp_code'/'merge_changes.py'),str(P/'temp_code'/'labeller_batches'/'batch_001.csv'),str(R/'changes_001.txt'),str(R/'out_batch001.csv')],check=True)
    d=pd.read_csv(R/'out_batch001.csv',dtype=str).fillna(''); d=d.sort_values(['address_id','word_index'],key=lambda s:s.astype(int) if s.name=='word_index' else s)
    for _,g in d.groupby('address_id',sort=False):
        ix=g.index.tolist()
        for a,b in zip(ix,ix[1:]):
            if d.at[b,'tag']=='B-LANDMARK' and d.at[a,'tag']=='O' and d.at[a,'word'].lower() in SAFE: d.at[a,'tag']='B-LANDMARK'
    d.to_csv(R/'out2_batch001.csv',index=False)
    subset=ref[ref.address_id.isin(set(batch.address_id))]; subset_path=R/'batch001_reference.csv'; subset.to_csv(subset_path,index=False)
    subprocess.run([sys.executable,str(P/'temp_code'/'check_ai_labels.py'),'--reference',str(subset_path),'--ai',str(R/'out2_batch001.csv'),'--output-dir',str(R/'check2_batch001')],check=True)
    hold=set(pd.read_csv(H,dtype=str).address_id); base=ref[~ref.address_id.isin(hold)].copy(); ai={}
    rejected=[]
    for n in [*range(1,15),'batch001']:
        c=R/f'check2_{n:03d}' if isinstance(n,int) else R/'check2_batch001'
        ac=pd.read_csv(c/'accepted_ai_labels.csv',dtype=str).fillna(''); rj=pd.read_csv(c/'rejected_ai_labels.csv',dtype=str).fillna('')
        ai.update({(x.address_id,x.word_index):x.tag for x in ac.itertuples(index=False)})
        rejected.append((n,rj))
    # Invalid-I-only rows retain AI after repair only when protected tags still agree.
    special=[]
    for n,rj in rejected:
        ids=rj.loc[rj.rejection_reasons.eq('invalid_i_transition'),'address_id'].unique()
        if len(ids):
            source=pd.read_csv(R/(f'out2_{n:03d}.csv' if isinstance(n,int) else 'out2_batch001.csv'),dtype=str).fillna('')
            for aid in ids:
                g=orphan(source[source.address_id.eq(aid)]); b=base[base.address_id.eq(aid)]
                protected=['TOWN','LOCALITY','PINCODE']; ok=True
                for x,y in zip(g.tag,b.tag):
                    if any(t in y for t in protected) and x!=y: ok=False
                print(f'invalid_i_transition {aid}: '+('AI repaired' if ok else 'fallback'))
                if ok:
                    ai.update({(x.address_id,x.word_index):x.tag for x in g.itertuples(index=False)}); special.append(aid)
    merged=base.copy(); merged['tag']=[ai.get((x.address_id,x.word_index),x.tag) for x in merged.itertuples(index=False)]
    merged=pd.concat([orphan(g) for _,g in merged.groupby('address_id',sort=False)],ignore_index=True)
    words=merged[['address_id','word_index','word']].sort_values(['address_id','word_index'],key=lambda s:s.astype(int) if s.name=='word_index' else s).reset_index(drop=True)
    basewords=base[['address_id','word_index','word']].sort_values(['address_id','word_index'],key=lambda s:s.astype(int) if s.name=='word_index' else s).reset_index(drop=True)
    checks=(merged.address_id.nunique()==2075,not merged.duplicated(['address_id','word_index']).any(),not merged.address_id.isin(hold).any(),words.equals(basewords),all(bio(g) for _,g in merged.groupby('address_id')))
    print('addresses',merged.address_id.nunique(),'checks',checks,'special',special)
    if not all(checks): raise ValueError('v2 checks failed')
    merged.to_csv(L/'train_bio_merged_v2.csv',index=False)
if __name__=='__main__': main()
