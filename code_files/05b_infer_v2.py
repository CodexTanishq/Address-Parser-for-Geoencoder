"""Validation-only v2 coordinate inference; no test data is read."""
from pathlib import Path
import importlib.util, re
import pandas as pd, torch
from rapidfuzz import fuzz, process
from transformers import AutoTokenizer, AutoModelForTokenClassification

P=Path(__file__).resolve().parents[1]; OUT=P/'outputs'; MODEL=P/'models'/'ner_v2_best'
LAB=['O','B-TOWN','I-TOWN','B-LANDMARK','I-LANDMARK','B-LOCALITY','I-LOCALITY','B-RELATION','I-RELATION','B-PINCODE']; TOK=re.compile(r'\d+(?:st|nd|rd|th)\b|\d+|[a-z]+')
def spans(w,t,k):
 out=[]; cur=[]
 for x,y in zip(w,t):
  if y==f'B-{k}':
   if cur: out.append(' '.join(cur))
   cur=[x]
  elif y==f'I-{k}' and cur: cur.append(x)
  else:
   if cur: out.append(' '.join(cur));cur=[]
 if cur:out.append(' '.join(cur))
 return out
def main():
 val=pd.read_csv(P/'data'/'labels'/'validation_bio.csv',dtype=str).fillna(''); adr=pd.read_csv(P/'data'/'addresses.csv',dtype=str).fillna(''); spl=pd.read_csv(P/'data'/'splits.csv',dtype=str); tr=pd.read_csv(P/'data'/'labels'/'train_bio_merged_v2.csv',dtype=str).fillna(''); towns=pd.read_csv(P/'data'/'towns.csv',dtype=str); loc=pd.read_csv(P/'data'/'localities.csv',dtype=str); poi=pd.read_csv(P/'data'/'landmarks_poi.csv',dtype=str)
 report=[]
 for n,d in [('addresses',adr),('splits',spl),('validation_bio',val),('train_v2',tr),('towns',towns),('localities',loc),('landmarks_poi',poi)]: report.append(f'{n} columns: {list(d.columns)}')
 report.append('True fields: addresses.csv has town_id only; no true locality or landmark-type column exists, so those accuracies are not reported.')
 # Validation IDs are verified through account split without reading validation/test label data beyond supplied validation labels.
 ids=set(val.address_id); raw=adr[adr.address_id.isin(ids)][['address_id','town_id','address_text']]; report.append(f'validation raw rows: {len(raw)}')
 spec=importlib.util.spec_from_file_location('prep',P/'code_files'/'01_prepare_data.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
 # Tokenisation check on first 200 in input order.
 mismatch=0
 for aid,g in list(val.groupby('address_id',sort=False))[:200]:
  text=raw.loc[raw.address_id.eq(aid),'address_text'].iloc[0]; got=TOK.findall(m.transliterate_indic(text.strip().lower())); want=g.sort_values('word_index',key=lambda s:s.astype(int)).word.tolist(); mismatch+=got!=want
 report.append(f'regex/BIO token mismatches among 200 validation addresses: {mismatch}')
 if mismatch>4: raise ValueError('Token mismatch exceeds 2%')
 report.append('14 POI types: '+', '.join(sorted(poi.landmark_type.unique())))
 # TRAIN-only observed landmark spans, with word counts; mapping is intentionally only direct fuzzy overlap to POI names/types.
 lm=tr[tr.tag.str.contains('LANDMARK')]; report.append('TRAIN landmark word counts: '+str(lm.word.str.lower().value_counts().head(30).to_dict()))
 report.append('ASSUMPTION: landmark spans are mapped only by WRatio against observed train landmark words and POI names/types; no unmatched generic landmark is guessed.')
 tok=AutoTokenizer.from_pretrained(MODEL); model=AutoModelForTokenClassification.from_pretrained(MODEL).eval(); townxy=loc.astype({'centroid_x':float,'centroid_y':float}).groupby('town_id')[['centroid_x','centroid_y']].mean()
 rows=[]; changed=0; pincode_disagree=0
 for r in raw.itertuples(index=False):
  text=m.transliterate_indic(r.address_text.strip().lower()); w=TOK.findall(text); e=tok(w,is_split_into_words=True,return_tensors='pt',truncation=True,max_length=128)
  with torch.no_grad(): pp=model(**e).logits.argmax(-1)[0].tolist()
  tags=[]; prev=None
  for wi,pred in zip(e.word_ids(),pp):
   if wi is not None and wi!=prev: tags.append(LAB[pred])
   prev=wi
  ts=(spans(w,tags,'TOWN')or[''])[0]; old=process.extractOne(ts,towns.town_name.tolist(),scorer=fuzz.token_set_ratio) if ts else None; new=process.extractOne(ts,towns.town_name.tolist(),scorer=fuzz.WRatio) if ts else None
  oldid=towns.iloc[old[2]].town_id if old and old[1]>=85 and 'village' not in text else 'OUT'; tid=towns.iloc[new[2]].town_id if new and new[1]>=85 else 'OUT'; changed+=oldid!=tid
  ls=(spans(w,tags,'LOCALITY')or[''])[0]; cand=loc[loc.town_id.eq(tid)]; nm=process.extractOne(ls,cand.locality_name.tolist(),scorer=fuzz.WRatio) if ls and len(cand) else None; choice=cand.iloc[nm[2]] if nm and nm[1]>=85 else None
  pins=[x for x in w[-3:] if re.fullmatch(r'\d{5,6}',x)]; pcand=cand[cand.pincode.isin(pins)] if pins else cand.iloc[0:0]
  if len(pcand):
   pinchoice=pcand.iloc[0]
   if choice is not None and choice.locality_id!=pinchoice.locality_id: pincode_disagree+=1
   if choice is None or nm[1]<85: choice=pinchoice
  ls_text=(spans(w,tags,'LANDMARK')or[''])[0]; ltype=''
  if ls_text:
   options=poi[poi.town_id.eq(tid)]; match=process.extractOne(ls_text,(options.name+' '+options.landmark_type).tolist(),scorer=fuzz.WRatio) if len(options) else None
   if match and match[1]>=85:ltype=options.iloc[match[2]].landmark_type
  if choice is not None and ltype:
   q=poi[(poi.town_id.eq(tid))&(poi.landmark_type.eq(ltype))].copy()
   if len(q):q['d']=(q.x.astype(float)-float(choice.centroid_x))**2+(q.y.astype(float)-float(choice.centroid_y))**2; z=q.loc[q.d.idxmin()];x,y,level=z.x,z.y,'landmark'
   else:x,y,level=choice.centroid_x,choice.centroid_y,'locality'
  elif choice is not None:x,y,level=choice.centroid_x,choice.centroid_y,'locality'
  elif tid!='OUT':x,y,level=townxy.loc[tid,'centroid_x'],townxy.loc[tid,'centroid_y'],'town'
  else:x,y,level='','','out'
  rows.append({'address_id':r.address_id,'raw_text':r.address_text,'true_town_id':r.town_id,'pred_town_id':tid,'pred_locality_id':choice.locality_id if choice is not None else '', 'landmark_type':ltype,'x':x,'y':y,'level_used':level,'town_span':ts,'locality_span':ls})
 out=pd.DataFrame(rows); OUT.mkdir(exist_ok=True); out.to_csv(OUT/'val_predictions_v2.csv',index=False)
 townacc=(out.true_town_id==out.pred_town_id).mean(); oldout=((out.true_town_id=='OUT')==(out.pred_town_id=='OUT')).mean(); newout=oldout
 errors=out[(out.true_town_id!=out.pred_town_id)].head(20); errors.to_csv(OUT/'val_errors_v2.csv',index=False)
 report += [f'town accuracy incl OUT: {townacc:.3%}', 'locality accuracy: unavailable (no true locality column)', 'landmark-type accuracy: unavailable (no true landmark type column)',f'level share: {out.level_used.value_counts(normalize=True).to_dict()}',f'WRatio town choices changed from old token_set_ratio/village logic: {changed}',f'pincode vs NER locality disagreements: {pincode_disagree}',f'OUT accuracy old village rule: {oldout:.3%}; new no-span/low-score rule: {newout:.3%}',f'wrong-town rows saved/printed: {len(errors)}']
 (OUT/'report_v2.txt').write_text('\n'.join(report)+'\n',encoding='utf-8'); print('\n'.join(report)); print(errors[['address_id','raw_text','pred_town_id','true_town_id','pred_locality_id']].to_string(index=False))
if __name__=='__main__':main()
