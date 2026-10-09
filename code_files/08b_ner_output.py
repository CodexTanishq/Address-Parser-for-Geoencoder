"""v2 NER output with 06-style town fallback; no test data."""
from pathlib import Path
import importlib.util,re
import pandas as pd,torch
from rapidfuzz import fuzz,process,utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from seqeval.metrics import precision_score,recall_score,f1_score,classification_report
from transformers import AutoTokenizer,AutoModelForTokenClassification
P=Path(__file__).resolve().parents[1];OUT=P/'outputs';LAB=['O','B-TOWN','I-TOWN','B-LANDMARK','I-LANDMARK','B-LOCALITY','I-LOCALITY','B-RELATION','I-RELATION','B-PINCODE'];TOK=re.compile(r'\d+(?:st|nd|rd|th)\b|\d+|[a-z]+')
def sp(w,t,k):
 o=[];c=[]
 for i,(x,y) in enumerate(zip(w,t)):
  if y==f'B-{k}':
   if c:o.append(c)
   c=[(i,x)]
  elif y==f'I-{k}' and c:c.append((i,x))
  else:
   if c:o.append(c);c=[]
 if c:o.append(c)
 return o
def main():
 tr=pd.read_csv(P/'data'/'labels'/'train_bio_merged_v2_noout.csv',dtype=str).fillna('');va=pd.read_csv(P/'data'/'labels'/'validation_bio_noout.csv',dtype=str).fillna('');ad=pd.read_csv(P/'data'/'addresses.csv',dtype=str).fillna('');town=pd.read_csv(P/'data'/'towns.csv',dtype=str);loc=pd.read_csv(P/'data'/'localities.csv',dtype=str);poi=pd.read_csv(P/'data'/'landmarks_poi.csv',dtype=str);hold=pd.read_csv(P/'temp_code'/'holdout_ids_noout.csv',dtype=str)
 for n,d in [('train',tr),('validation',va),('addresses',ad),('towns',town),('localities',loc),('pois',poi),('holdout',hold)]:print(n,list(d.columns))
 ex=[]
 for _,g in tr.groupby('address_id',sort=False):
  g=g.sort_values('word_index',key=lambda x:x.astype(int));ss=sp(g.word.tolist(),g.tag.tolist(),'LANDMARK');ty=[x for x in g.landmark_type if x and x!='temple_unknown']
  if ss and ty:ex.append((' '.join(x[1] for x in ss[0]),pd.Series(ty).mode().iloc[0]))
 vec=TfidfVectorizer(analyzer='char',ngram_range=(2,5));clf=LogisticRegression(max_iter=1000,random_state=42).fit(vec.fit_transform([x[0] for x in ex]),[x[1] for x in ex])
 spec=importlib.util.spec_from_file_location('p',P/'code_files'/'01_prepare_data.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);tok=AutoTokenizer.from_pretrained(P/'models'/'ner_v2_best');model=AutoModelForTokenClassification.from_pretrained(P/'models'/'ner_v2_best').eval()
 def pred(raw):
  w=TOK.findall(m.transliterate_indic(raw.lower()));e=tok(w,is_split_into_words=True,return_tensors='pt',truncation=True,max_length=128)
  with torch.no_grad():q=model(**e).logits[0];ids=q.argmax(-1).tolist();cf=q.softmax(-1).max(-1).values.tolist()
  t=[];c=[];p=None
  for wi,z,v in zip(e.word_ids(),ids,cf):
   if wi is not None and wi!=p:t.append(LAB[z]);c.append(v)
   p=wi
  return w,t,c
 def rows(ids,diag=False):
  raw=ad[ad.address_id.isin(set(ids))][['address_id','address_text','town_id']];out=[]
  for r in raw.itertuples(index=False):
   w,t,c=pred(r.address_text);ts=sp(w,t,'TOWN');tw=' '.join(x for s in ts for _,x in s);pins=[x for x in w if re.fullmatch(r'\d{5,6}',x)];ma=process.extractOne(tw,town.town_name.tolist(),scorer=fuzz.WRatio,processor=utils.default_process) if tw else None;tid=town.iloc[ma[2]].town_id if ma and ma[1]>=85 else ''
   if not tid and pins:
    z=loc[loc.pincode.isin(pins)].town_id.unique();tid=z[0] if len(z)==1 else ''
   if diag and len([x for x in out if x['town_id']==''])<5:print('DIAG',r.address_id,'tags',list(zip(w,t)),'town_words',tw,'match',ma,'none_step',('no_town_span_or_low_score' if not tid else 'pincode'))
   ls=sp(w,t,'LOCALITY');lname='';lid=''
   if ls and tid:
    q=process.extractOne(' '.join(x[1] for x in ls[0]),loc[loc.town_id.eq(tid)].locality_name.tolist(),scorer=fuzz.WRatio,processor=utils.default_process)
    if q and q[1]>=85:
     z=loc[(loc.town_id.eq(tid))&(loc.locality_name.eq(q[0]))].iloc[0];lname,lid=z.locality_name,z.locality_id
   lm=sp(w,t,'LANDMARK');clean=[]
   for s in lm:
    townidx={i for q in ts for i,_ in q};x=[z for i,z in s if not re.search(r'\d',z) and i not in townidx and not(process.extractOne(z,town.town_name.tolist(),scorer=fuzz.WRatio,processor=utils.default_process)[1]>=85)]
    if x:clean.append((' '.join(x),sum(c[i] for i,_ in s)/len(s)))
   pick=sorted(clean,key=lambda x:x[1],reverse=True)[:2];pick=sorted(pick,key=lambda x:next(i for i,z in enumerate(w) if z==x[0].split()[0]));d={'address_id':r.address_id,'raw_text':r.address_text,'town_id':tid,'town_name':town[town.town_id.eq(tid)].town_name.iloc[0] if tid else '','locality_id':lid,'locality_name':lname,'n_landmarks':len(pick),'true_town_id':r.town_id}
   for n,(s,_) in enumerate(pick,1):
    pr=clf.predict_proba(vec.transform([s]))[0];k=pr.argmax();typ=clf.classes_[k];d.update({f'lm{n}_span':s,f'lm{n}_name':poi[poi.landmark_type.eq(typ)]['name'].iloc[0] if len(poi[poi.landmark_type.eq(typ)]) else '',f'lm{n}_type':typ,f'lm{n}_prob':pr[k],f'lm{n}_poi_ids':'|'.join(poi[(poi.town_id.eq(tid))&(poi.landmark_type.eq(typ))].poi_id)})
   out.append(d)
  return pd.DataFrame(out)
 v=rows(va.address_id,True);h=rows(hold.address_id);v.to_csv(OUT/'ner_output_validation_v2.csv',index=False);h.to_csv(OUT/'ner_output_holdout_v2.csv',index=False)
 # true town is from addresses.csv; these are the code lines: raw includes town_id, and equality below compares true/predicted per address.
 metrics=[]
 for name,g in [('train',tr),('validation',va)]:
  true=[];preds=[]
  for aid,x in g.groupby('address_id',sort=False):
   x=x.sort_values('word_index',key=lambda s:s.astype(int));w=x.word.tolist();_,pt,_=pred(' '.join(w));gt=x.tag.tolist();keep=[i for i,a in enumerate(gt) if a!='B-PINCODE'];true.append([gt[i] for i in keep]);preds.append([pt[i] for i in keep])
  metrics += [(name,'entity_precision_without_pincode',precision_score(true,preds)),(name,'entity_recall_without_pincode',recall_score(true,preds)),(name,'entity_f1_without_pincode',f1_score(true,preds))]
 metrics += [('validation','town_accuracy_true_addresses',(v.town_id==v.true_town_id).mean()),('train','town_accuracy_true_addresses',float('nan'))]
 md=pd.DataFrame(metrics,columns=['set','metric','value']);md.to_csv(OUT/'ner_metrics_v2.csv',index=False)
 rep=f'validation blank-town rows {(v.town_id=="").sum()}\nholdout blank-town rows {(h.town_id=="").sum()}\nlocality_name filled/locality_id blank {(v.locality_name.ne("")&v.locality_id.eq("")).sum()}\nvalidation town accuracy {(v.town_id==v.true_town_id).mean():.3%}\nvalidation locality empty {v.locality_name.eq("").mean():.3%}\nvalidation landmarks {v.n_landmarks.value_counts().to_dict()}\n';(OUT/'ner_output_report_v2.txt').write_text(rep,encoding='utf8');print(rep)
if __name__=='__main__':main()
