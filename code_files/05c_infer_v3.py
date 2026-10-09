"""Validation no-OUT inference with pincode fallback; never reads test data."""
from pathlib import Path
import importlib.util,re,sys
import pandas as pd,torch
from rapidfuzz import fuzz,process,utils
from transformers import AutoTokenizer,AutoModelForTokenClassification
P=Path(__file__).resolve().parents[1]; TOK=re.compile(r'\d+(?:st|nd|rd|th)\b|\d+|[a-z]+'); LAB=['O','B-TOWN','I-TOWN','B-LANDMARK','I-LANDMARK','B-LOCALITY','I-LOCALITY','B-RELATION','I-RELATION','B-PINCODE']
def spans(w,t,k):
 o=[];c=[]
 for x,y in zip(w,t):
  if y==f'B-{k}':
   if c:o.append(' '.join(c))
   c=[x]
  elif y==f'I-{k}' and c:c.append(x)
  else:
   if c:o.append(' '.join(c));c=[]
 if c:o.append(' '.join(c))
 return o
def clean(s):return ' '.join(x for x in s.split() if not re.fullmatch(r'\d{5,6}',x))
def main(model_dir=None,suffix='v3'):
 model_dir=Path(model_dir or P/'models'/'ner_v2_best'); out=P/'outputs'; out.mkdir(exist_ok=True)
 val=pd.read_csv(P/'data'/'labels'/'validation_bio_noout.csv',dtype=str).fillna(''); adr=pd.read_csv(P/'data'/'addresses.csv',dtype=str).fillna(''); towns=pd.read_csv(P/'data'/'towns.csv',dtype=str); loc=pd.read_csv(P/'data'/'localities.csv',dtype=str); poi=pd.read_csv(P/'data'/'landmarks_poi.csv',dtype=str)
 for n,d in [('towns',towns),('validation',val),('addresses',adr),('localities',loc),('poi',poi)]:print(n,list(d.columns))
 print('towns full:\n'+towns.to_string(index=False));print('validation landmark_type counts:\n'+val.landmark_type.value_counts(dropna=False).to_string())
 spec=importlib.util.spec_from_file_location('a',P/'code_files'/'01_prepare_data.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
 raw=adr[adr.address_id.isin(set(val.address_id))][['address_id','town_id','address_text']]; tok=AutoTokenizer.from_pretrained(model_dir);model=AutoModelForTokenClassification.from_pretrained(model_dir).eval(); xy=loc.astype({'centroid_x':float,'centroid_y':float}).groupby('town_id')[['centroid_x','centroid_y']].mean(); rows=[];diag=[];dis=unknown=0
 for r in raw.itertuples(index=False):
  w=TOK.findall(a.transliterate_indic(r.address_text.lower()));e=tok(w,is_split_into_words=True,return_tensors='pt',truncation=True,max_length=128)
  with torch.no_grad():q=model(**e).logits.argmax(-1)[0].tolist()
  tags=[];pr=None
  for wi,z in zip(e.word_ids(),q):
   if wi is not None and wi!=pr:tags.append(LAB[z])
   pr=wi
  ts=clean((spans(w,tags,'TOWN')or[''])[0]); ls=clean((spans(w,tags,'LOCALITY')or[''])[0]); lm=clean((spans(w,tags,'LANDMARK')or[''])[0]); pins=[x for x in w[-3:] if re.fullmatch(r'\d{5,6}',x)]
  old=process.extractOne(ts,towns.town_name.tolist(),scorer=fuzz.token_set_ratio,processor=utils.default_process) if ts else None; best=process.extractOne(ts,towns.town_name.tolist(),scorer=fuzz.WRatio,processor=utils.default_process) if ts else None
  if len(diag)<15:diag.append((r.address_id,ts,old,best))
  tid=towns.iloc[best[2]].town_id if best else None
  if not ts and pins:
   t=loc[loc.pincode.isin(pins)].town_id.unique();tid=t[0] if len(t)==1 else None
  if tid is None:tid='UNKNOWN';unknown+=1
  cand=loc[loc.town_id.eq(tid)]; m=process.extractOne(ls,cand.locality_name.tolist(),scorer=fuzz.WRatio,processor=utils.default_process) if ls and len(cand) else None; choice=cand.iloc[m[2]] if m and m[1]>=85 else None; pc=cand[cand.pincode.isin(pins)] if pins else cand.iloc[0:0]
  if len(pc):
   if choice is not None and choice.locality_id!=pc.iloc[0].locality_id:dis+=1
   if choice is None:choice=pc.iloc[0]
  if choice is not None:x,y,level=choice.centroid_x,choice.centroid_y,'locality'
  elif tid!='UNKNOWN':x,y,level=xy.loc[tid,'centroid_x'],xy.loc[tid,'centroid_y'],'town'
  else:x,y,level='','','unknown'
  rows.append({'address_id':r.address_id,'raw_text':r.address_text,'true_town_id':r.town_id,'pred_town_id':tid,'town_span':ts,'locality_span':ls,'landmark_span':lm,'pred_locality_id':choice.locality_id if choice is not None else '','x':x,'y':y,'level_used':level})
 outd=pd.DataFrame(rows); acc=(outd.true_town_id==outd.pred_town_id).mean(); err=outd[outd.true_town_id.ne(outd.pred_town_id)].head(20);outd.to_csv(out/f'val_predictions_{suffix}.csv',index=False);err.to_csv(out/f'val_errors_{suffix}.csv',index=False)
 print('15 diagnostics');[print(x) for x in diag];print('Low scores occur because raw-token inference usually emits no TOWN span; case processing and pincode removal cannot score an empty span.')
 rep=f'town accuracy {acc:.3%}\nUNKNOWN {unknown}\nlevels {outd.level_used.value_counts(normalize=True).to_dict()}\npincode/NER disagreements {dis}\nlocality accuracy unavailable: no true locality field\nlandmark-type accuracy unavailable: label landmark_type is not a true source field\n';(out/f'report_{suffix}.txt').write_text(rep,encoding='utf8');print(rep);print(err[['address_id','raw_text','true_town_id','pred_town_id','town_span','locality_span']].to_string(index=False))
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else None,sys.argv[2] if len(sys.argv)>2 else 'v3')
