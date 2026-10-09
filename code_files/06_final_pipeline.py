"""Validation-only final address-coordinate pipeline; no test data is read."""
from pathlib import Path
import importlib.util,re
import pandas as pd, torch
from rapidfuzz import fuzz,process,utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer,AutoModelForTokenClassification
P=Path(__file__).resolve().parents[1];OUT=P/'outputs';TOK=re.compile(r'\d+(?:st|nd|rd|th)\b|\d+|[a-z]+');LAB=['O','B-TOWN','I-TOWN','B-LANDMARK','I-LANDMARK','B-LOCALITY','I-LOCALITY','B-RELATION','I-RELATION','B-PINCODE']
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
def nopin(s):return ' '.join(x for x in s.split() if not re.fullmatch(r'\d{5,6}',x))
def main():
 OUT.mkdir(exist_ok=True); val=pd.read_csv(P/'data'/'labels'/'validation_bio_noout.csv',dtype=str).fillna('');tr=pd.read_csv(P/'data'/'labels'/'train_bio_merged_v2_noout.csv',dtype=str).fillna('');adr=pd.read_csv(P/'data'/'addresses.csv',dtype=str).fillna('');town=pd.read_csv(P/'data'/'towns.csv',dtype=str);loc=pd.read_csv(P/'data'/'localities.csv',dtype=str);poi=pd.read_csv(P/'data'/'landmarks_poi.csv',dtype=str)
 for n,d in [('validation',val),('train',tr),('addresses',adr),('towns',town),('localities',loc),('poi',poi)]:print(n,list(d.columns))
 p2t=loc.groupby('pincode').town_id.nunique();print('pincode belongs to one town:',bool((p2t==1).all()),'violations',int((p2t!=1).sum()))
 if not (p2t==1).all():raise ValueError('pincode-to-town invariant failed')
 # Assign POIs to nearest locality: ASSUMPTION that locality centroid defines a POI catchment.
 assigned=[]
 for x in poi.itertuples(index=False):
  c=loc[loc.town_id.eq(x.town_id)].copy();c['d']=(c.centroid_x.astype(float)-float(x.x))**2+(c.centroid_y.astype(float)-float(x.y))**2;z=c.loc[c.d.idxmin()];assigned.append((x.poi_id,z.locality_id))
 amap=dict(assigned);print('POIs per locality (ASSUMPTION catchments):\n'+pd.Series(list(amap.values())).value_counts().to_string())
 # Span-level training examples, excluding empty/unknown labels from POI mapping.
 ex=[]; conflict=0
 for aid,g in tr.groupby('address_id',sort=False):
  w=g.sort_values('word_index',key=lambda s:s.astype(int)); ss=spans(w.word.tolist(),w.tag.tolist(),'LANDMARK'); types=[x for x in w.landmark_type if x and x!='temple_unknown']
  if ss and types:
   if len(set(types))>1:conflict+=1
   ex.append((ss[0],pd.Series(types).mode().iloc[0]))
 print('landmark training spans',len(ex),'spans with differing word types',conflict)
 vec=TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=1); X=vec.fit_transform([x[0] for x in ex]); clf=LogisticRegression(max_iter=1000,random_state=42).fit(X,[x[1] for x in ex])
 spec=importlib.util.spec_from_file_location('prep',P/'code_files'/'01_prepare_data.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);tok=AutoTokenizer.from_pretrained(P/'models'/'ner_v2_best');model=AutoModelForTokenClassification.from_pretrained(P/'models'/'ner_v2_best').eval();raw=adr[adr.address_id.isin(set(val.address_id))][['address_id','town_id','address_text']];xy=loc.astype({'centroid_x':float,'centroid_y':float}).groupby('town_id')[['centroid_x','centroid_y']].mean();rows=[]; probs=[]; cnt={'span':0,'landmark':0,'no_locality':0,'top_not_in_locality':0,'low_prob':0,'no_candidate':0,'pincode_conflict':0}
 for r in raw.itertuples(index=False):
  w=TOK.findall(m.transliterate_indic(r.address_text.lower()));e=tok(w,is_split_into_words=True,return_tensors='pt',truncation=True,max_length=128)
  with torch.no_grad():q=model(**e).logits.argmax(-1)[0].tolist()
  tags=[];prev=None
  for wi,z in zip(e.word_ids(),q):
   if wi is not None and wi!=prev:tags.append(LAB[z])
   prev=wi
  ts=nopin((spans(w,tags,'TOWN')or[''])[0]);ls=nopin((spans(w,tags,'LOCALITY')or[''])[0]);lm=nopin((spans(w,tags,'LANDMARK')or[''])[0]);pins=[x for x in w if re.fullmatch(r'\d{5,6}',x)]
  mt=process.extractOne(ts,town.town_name.tolist(),scorer=fuzz.WRatio,processor=utils.default_process) if ts else None; tid=town.iloc[mt[2]].town_id if mt and mt[1]>=85 else ''; townrule='NER'
  if not tid and pins:
   z=loc[loc.pincode.isin(pins)].town_id.unique();tid=z[0] if len(z)==1 else '';townrule='pincode'
  if not tid:tid='UNKNOWN';townrule='UNKNOWN'
  allc=loc[loc.town_id.eq(tid)];cand=allc[allc.pincode.isin(pins)] if pins else allc.iloc[0:0]; ml=process.extractOne(ls,allc.locality_name.tolist(),scorer=fuzz.WRatio,processor=utils.default_process) if ls and len(allc) else None; choice=allc.iloc[ml[2]] if ml and ml[1]>=85 else None; lrule='NER'
  if choice is not None and len(cand) and choice.pincode not in set(cand.pincode):cnt['pincode_conflict']+=1
  if choice is None and len(cand)==1:choice=cand.iloc[0];lrule='pincode'
  if choice is None and 2<=len(cand)<=5:x,y,level=cand.centroid_x.astype(float).mean(),cand.centroid_y.astype(float).mean(),'pincode_area';lrule='pincode_area'
  elif choice is None and tid!='UNKNOWN':x,y,level=xy.loc[tid,'centroid_x'],xy.loc[tid,'centroid_y'],'town';lrule='town'
  elif choice is not None:x,y,level=choice.centroid_x,choice.centroid_y,'locality'
  else:x,y,level='','','town'
  lt=pid='';prob=0
  if lm:cnt['span']+=1
  if lm and choice is None:cnt['no_locality']+=1
  if lm and choice is not None:
   pr=clf.predict_proba(vec.transform([lm]))[0]; order=pr.argsort()[::-1];opts=poi[(poi.town_id.eq(tid))&(poi.poi_id.map(amap).eq(choice.locality_id))]
   if clf.classes_[order[0]] not in set(opts.landmark_type):cnt['top_not_in_locality']+=1
   for j in order:
    typ=clf.classes_[j]
    if typ in set(opts.landmark_type):lt,prob=typ,float(pr[j]);break
   probs.append(prob)
   if not lt:cnt['no_candidate']+=1
   if lt and prob>=.5:
    hit=opts[opts.landmark_type.eq(lt)];dd=(hit.x.astype(float)-float(choice.centroid_x))**2+(hit.y.astype(float)-float(choice.centroid_y))**2;zz=hit.loc[dd.idxmin()];pid=zz.poi_id;x,y,level=float(zz.x),float(zz.y),'landmark';cnt['landmark']+=1
   else:
    if lt:cnt['low_prob']+=1
    lt=''
  rows.append({'address_id':r.address_id,'raw_text':r.address_text,'town_id':tid,'locality_id':choice.locality_id if choice is not None else '','landmark_type':lt,'poi_id':pid,'x':x,'y':y,'level_used':level,'town_rule':townrule,'locality_rule':lrule,'true_town_id':r.town_id,'landmark_span':lm,'landmark_probability':prob})
 for th in [.3,.4,.5,.6]:print('threshold',th,'rows landmark eligible',sum(x>=th for x in probs))
 # Threshold 0.5 is an ASSUMPTION selected from validation-wide distribution, not a single row.
 out=pd.DataFrame(rows);out.to_csv(OUT/'val_predictions_final_v2.csv',index=False)
 truth=val.groupby('address_id').landmark_type.apply(lambda s:next((x for x in s if x),'')); merged=out.merge(truth.rename('auto_landmark_type'),on='address_id'); known=merged.auto_landmark_type.ne('');lacc=(merged.loc[known,'landmark_type']==merged.loc[known,'auto_landmark_type']).mean();tacc=(out.town_id==out.true_town_id).mean();bad=out[(out.town_id!=out.true_town_id)|(out.level_used!='locality')].head(20);bad.to_csv(OUT/'val_errors_final_v2.csv',index=False)
 rep=f'town accuracy {tacc:.3%}\nUNKNOWN {(out.town_id=="UNKNOWN").sum()}\nlevels {out.level_used.value_counts(normalize=True).to_dict()}\ntown pincode use {(out.town_rule=="pincode").sum()}\nlocality pincode use {(out.locality_rule=="pincode").sum()}\nno landmark span {(out.landmark_span=="").sum()}\npoi_id filled {(out.poi_id!="").sum()}\nlandmark counters {cnt}\nlandmark type accuracy {lacc:.3%} (CIRCULAR: landmark_type comes from 02_auto_label.py)\nlocality accuracy unavailable: no true locality field\nASSUMPTIONS: POI catchment is nearest locality; threshold=0.5; classifier labels originate from auto labels.\n';(OUT/'report_final_v2.txt').write_text(rep,encoding='utf8');print(rep);print(bad[['address_id','raw_text','true_town_id','town_id','level_used','landmark_span']].to_string(index=False))
if __name__=='__main__':main()