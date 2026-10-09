"""One-time final no-OUT test inference using the fixed v2-best model."""
from pathlib import Path
import importlib.util,re
import pandas as pd,torch
from rapidfuzz import fuzz,process,utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer,AutoModelForTokenClassification

P=Path(__file__).resolve().parents[1];OUT=P/'outputs';TOK=re.compile(r'\d+(?:st|nd|rd|th)\b|\d+|[a-z]+');LAB=['O','B-TOWN','I-TOWN','B-LANDMARK','I-LANDMARK','B-LOCALITY','I-LOCALITY','B-RELATION','I-RELATION','B-PINCODE']
def spans(w,t,k):
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
 test_ids=pd.read_csv(P/'data'/'test_noout_ids.csv',dtype=str);addresses=pd.read_csv(P/'data'/'addresses.csv',dtype=str).fillna('');train=pd.read_csv(P/'data'/'labels'/'train_bio_merged_v2_noout.csv',dtype=str).fillna('');towns=pd.read_csv(P/'data'/'towns.csv',dtype=str);loc=pd.read_csv(P/'data'/'localities.csv',dtype=str);poi=pd.read_csv(P/'data'/'landmarks_poi.csv',dtype=str)
 for n,d in [('test_noout_ids',test_ids),('addresses',addresses),('train',train),('towns',towns),('localities',loc),('pois',poi)]:print(n,list(d.columns))
 raw=addresses[addresses.address_id.isin(set(test_ids.address_id))][['address_id','address_text','town_id']]
 if len(raw)!=len(test_ids):raise ValueError('test address missing')
 ex=[]
 for _,g in train.groupby('address_id',sort=False):
  g=g.sort_values('word_index',key=lambda s:s.astype(int));ss=spans(g.word.tolist(),g.tag.tolist(),'LANDMARK');ty=[x for x in g.landmark_type if x and x!='temple_unknown']
  if ss and ty:ex.append((' '.join(x[1] for x in ss[0]),pd.Series(ty).mode().iloc[0]))
 vec=TfidfVectorizer(analyzer='char',ngram_range=(2,5));clf=LogisticRegression(max_iter=1000,random_state=42).fit(vec.fit_transform([x[0] for x in ex]),[x[1] for x in ex])
 spec=importlib.util.spec_from_file_location('prep',P/'code_files'/'01_prepare_data.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);tok=AutoTokenizer.from_pretrained(P/'models'/'ner_v2_best');model=AutoModelForTokenClassification.from_pretrained(P/'models'/'ner_v2_best').eval();rows=[];town_pin=0
 for r in raw.itertuples(index=False):
  w=TOK.findall(m.transliterate_indic(r.address_text.lower()));e=tok(w,is_split_into_words=True,return_tensors='pt',truncation=True,max_length=128)
  with torch.no_grad():q=model(**e).logits[0];ids=q.argmax(-1).tolist();cf=q.softmax(-1).max(-1).values.tolist()
  t=[];c=[];prev=None
  for wi,z,v in zip(e.word_ids(),ids,cf):
   if wi is not None and wi!=prev:t.append(LAB[z]);c.append(v)
   prev=wi
  ts=spans(w,t,'TOWN');tw=' '.join(x for s in ts for _,x in s if not re.fullmatch(r'\d{5,6}',x));pins=[x for x in w if re.fullmatch(r'\d{5,6}',x)];ma=process.extractOne(tw,towns.town_name.tolist(),scorer=fuzz.WRatio,processor=utils.default_process) if tw else None;tid=towns.iloc[ma[2]].town_id if ma and ma[1]>=85 else ''
  if not tid and pins:
   z=loc[loc.pincode.isin(pins)].town_id.unique()
   if len(z)==1:tid=z[0];town_pin+=1
  ls=spans(w,t,'LOCALITY');lname=lid=''
  if ls and tid:
   q=process.extractOne(' '.join(x[1] for x in ls[0]),loc[loc.town_id.eq(tid)].locality_name.tolist(),scorer=fuzz.WRatio,processor=utils.default_process)
   if q and q[1]>=85:z=loc[(loc.town_id.eq(tid))&(loc.locality_name.eq(q[0]))].iloc[0];lname,lid=z.locality_name,z.locality_id
  lm=spans(w,t,'LANDMARK');clean=[]
  townidx={i for s in ts for i,_ in s}
  for s in lm:
   x=[z for i,z in s if not re.search(r'\d',z) and i not in townidx and not(process.extractOne(z,towns.town_name.tolist(),scorer=fuzz.WRatio,processor=utils.default_process)[1]>=85)]
   if x:clean.append((' '.join(x),sum(c[i] for i,_ in s)/len(s)))
  picked=sorted(clean,key=lambda x:x[1],reverse=True)[:2];picked=sorted(picked,key=lambda x:next(i for i,z in enumerate(w) if z==x[0].split()[0]));d={'address_id':r.address_id,'raw_text':r.address_text,'town_id':tid,'town_name':towns[towns.town_id.eq(tid)].town_name.iloc[0] if tid else '','locality_id':lid,'locality_name':lname,'n_landmarks':len(picked),'true_town_id':r.town_id}
  for n,(s,_) in enumerate(picked,1):
   pr=clf.predict_proba(vec.transform([s]))[0];k=pr.argmax();typ=clf.classes_[k];d.update({f'lm{n}_span':s,f'lm{n}_name':poi[poi.landmark_type.eq(typ)]['name'].iloc[0] if len(poi[poi.landmark_type.eq(typ)]) else '',f'lm{n}_type':typ,f'lm{n}_prob':pr[k],f'lm{n}_poi_ids':'|'.join(poi[(poi.town_id.eq(tid))&(poi.landmark_type.eq(typ))].poi_id)})
  for n in (1,2):
   for key in ['span','name','type','prob','poi_ids']:d.setdefault(f'lm{n}_{key}','')
  rows.append(d)
 out=pd.DataFrame(rows);out.to_csv(OUT/'test_ner_output.csv',index=False);report=f'number of test rows {len(out)}\ntown accuracy {(out.town_id==out.true_town_id).mean():.3%}\nblank-town rows {(out.town_id=="").sum()}\ntown pincode fallback rows {town_pin}\nlocality empty share {out.locality_name.eq("").mean():.3%}\nrows with 0/1/2 landmarks {out.n_landmarks.value_counts().to_dict()}\nlocality and landmark cannot be scored: no true coordinates\n';(OUT/'test_report.txt').write_text(report,encoding='utf8');print(report)
if __name__=='__main__':main()
