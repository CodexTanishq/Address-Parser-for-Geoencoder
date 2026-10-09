"""Fixed NER output and train/validation metrics; never reads test files."""
from pathlib import Path
import argparse,importlib.util,re
import pandas as pd,torch
from rapidfuzz import fuzz,process,utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from seqeval.metrics import precision_score,recall_score,f1_score,classification_report
from transformers import AutoTokenizer,AutoModelForTokenClassification
P=Path(__file__).resolve().parents[1];TOK=re.compile(r'\d+(?:st|nd|rd|th)\b|\d+|[a-z]+');LAB=['O','B-TOWN','I-TOWN','B-LANDMARK','I-LANDMARK','B-LOCALITY','I-LOCALITY','B-RELATION','I-RELATION','B-PINCODE']
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
 ap=argparse.ArgumentParser();ap.add_argument('--ids',default=str(P/'data'/'labels'/'validation_bio_noout.csv'));args=ap.parse_args();OUT=P/'outputs';OUT.mkdir(exist_ok=True)
 train=pd.read_csv(P/'data'/'labels'/'train_bio_merged_v2_noout.csv',dtype=str).fillna('');val=pd.read_csv(P/'data'/'labels'/'validation_bio_noout.csv',dtype=str).fillna('');adr=pd.read_csv(P/'data'/'addresses.csv',dtype=str).fillna('');town=pd.read_csv(P/'data'/'towns.csv',dtype=str);loc=pd.read_csv(P/'data'/'localities.csv',dtype=str);poi=pd.read_csv(P/'data'/'landmarks_poi.csv',dtype=str)
 for n,d in [('requested_ids',pd.read_csv(args.ids,dtype=str)),('train',train),('validation',val),('addresses',adr),('towns',town),('localities',loc),('pois',poi)]:print(n,list(d.columns))
 # classifier train spans only
 ex=[]
 for _,g in train.groupby('address_id',sort=False):
  g=g.sort_values('word_index',key=lambda s:s.astype(int));ss=spans(g.word.tolist(),g.tag.tolist(),'LANDMARK');ty=[x for x in g.landmark_type if x and x!='temple_unknown']
  if ss and ty:ex.append((' '.join(x[1] for x in ss[0]),pd.Series(ty).mode().iloc[0]))
 vec=TfidfVectorizer(analyzer='char',ngram_range=(2,5));clf=LogisticRegression(max_iter=1000,random_state=42).fit(vec.fit_transform([x[0] for x in ex]),[x[1] for x in ex])
 spec=importlib.util.spec_from_file_location('prep',P/'code_files'/'01_prepare_data.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);tok=AutoTokenizer.from_pretrained(P/'models'/'ner_v2_best');model=AutoModelForTokenClassification.from_pretrained(P/'models'/'ner_v2_best').eval()
 def predict_words(w):
  e=tok(w,is_split_into_words=True,return_tensors='pt',truncation=True,max_length=128)
  with torch.no_grad():log=model(**e).logits[0];pr=log.softmax(-1);ids=log.argmax(-1).tolist()
  tags=[];conf=[];prev=None
  for wi,z,c in zip(e.word_ids(),ids,pr.max(-1).values.tolist()):
   if wi is not None and wi!=prev:tags.append(LAB[z]);conf.append(c)
   prev=wi
  return w,tags,conf
 def predict(raw):
  return predict_words(TOK.findall(m.transliterate_indic(raw.lower())))
 def metrics(name,gold):
  true=[];pred=[];allok=[];non=[];townok=[];locok=[];lmok=[]
  for aid,g in gold.groupby('address_id',sort=False):
   g=g.sort_values('word_index',key=lambda s:s.astype(int));w=g.word.tolist();w,t,c=predict_words(w);gt=g.tag.tolist()
   if len(gt)!=len(t): raise ValueError(f'token count mismatch for {aid}')
   keep=[i for i,x in enumerate(gt) if x!='B-PINCODE']
   gt_eval=[gt[i] for i in keep]; t_eval=[t[i] for i in keep]
   true.append(gt_eval);pred.append(t_eval);allok += [a==b for a,b in zip(gt_eval,t_eval)];non += [a==b for a,b in zip(gt_eval,t_eval) if a!='O']
   for typ,bag in [('TOWN',townok),('LOCALITY',locok)]:bag.append([' '.join(x[1] for x in s) for s in spans(w,t,typ)]==[' '.join(x[1] for x in s) for s in spans(g.word.tolist(),gt,typ)])
   gs=[x for x in g.landmark_type if x];ps=spans(w,t,'LANDMARK');pt=clf.predict(vec.transform([' '.join(x[1] for x in ps[0])]))[0] if ps else '';lmok.append((pt==(pd.Series(gs).mode().iloc[0] if gs else '')))
  rows=[(name,'token_accuracy',sum(allok)/len(allok)),(name,'token_accuracy_non_o',sum(non)/len(non)),(name,'entity_precision',precision_score(true,pred)),(name,'entity_recall',recall_score(true,pred)),(name,'entity_f1',f1_score(true,pred)),(name,'town_accuracy',sum(townok)/len(townok)),(name,'locality_exact_including_none',sum(locok)/len(locok)),(name,'landmark_type_accuracy_CIRCULAR',sum(lmok)/len(lmok))]
  detail=classification_report(true,pred,output_dict=True,zero_division=0)
  for typ in ['TOWN','LOCALITY','LANDMARK','RELATION','PINCODE']:
   values=detail.get(typ,{})
   for metric in ['precision','recall','f1-score']:
    rows.append((name,f'{typ}_{metric}',values.get(metric,0.0)))
  return rows
 met=metrics('train',train)+metrics('validation',val);md=pd.DataFrame(met,columns=['set','metric','value']);md.to_csv(OUT/'ner_metrics.csv',index=False);(OUT/'ner_metrics_report.txt').write_text(md.to_string(index=False),encoding='utf8')
 def output(ids,path):
  raw=adr[adr.address_id.isin(set(ids))][['address_id','address_text']];rows=[];dirty=over=0
  for r in raw.itertuples(index=False):
   w,t,c=predict(r.address_text);ts=spans(w,t,'TOWN');tm=process.extractOne(' '.join(x[1] for x in ts[0]),town.town_name.tolist(),scorer=fuzz.WRatio,processor=utils.default_process) if ts else None;tid=town.iloc[tm[2]].town_id if tm and tm[1]>=85 else ''
   ls=spans(w,t,'LOCALITY');lm=spans(w,t,'LANDMARK');local=''
   if ls and tid:
    q=process.extractOne(' '.join(x[1] for x in ls[0]),loc[loc.town_id.eq(tid)].locality_name.tolist(),scorer=fuzz.WRatio,processor=utils.default_process);local=q[0] if q and q[1]>=85 else ''
   clean=[]
   for s in lm:
    town_indices = {j for q in ts for j,_ in q}
    x=[z for i,z in s if not re.search(r'\d',z) and i not in town_indices and not (process.extractOne(z,town.town_name.tolist(),scorer=fuzz.WRatio,processor=utils.default_process)[1]>=85)]
    dirty+=len(x)!=len(s)
    if x:clean.append((' '.join(x),sum(c[i] for i,_ in s)/len(s)))
   over+=len(clean)>2;chosen=sorted(clean,key=lambda x:x[1],reverse=True)[:2];chosen=sorted(chosen,key=lambda x: next(i for i,z in enumerate(w) if z==x[0].split()[0]))
   d={'address_id':r.address_id,'raw_text':r.address_text,'town_id':tid,'town_name':town.loc[town.town_id.eq(tid),'town_name'].iloc[0] if tid else '','locality_id':'','locality_name':local,'n_landmarks':len(chosen)}
   for n,(s,_) in enumerate(chosen,1):
    pr=clf.predict_proba(vec.transform([s]))[0];k=pr.argmax();typ=clf.classes_[k];d.update({f'lm{n}_span':s,f'lm{n}_name':poi[poi.landmark_type.eq(typ)].name.iloc[0] if len(poi[poi.landmark_type.eq(typ)]) else '',f'lm{n}_type':typ,f'lm{n}_prob':pr[k],f'lm{n}_poi_ids':'|'.join(poi[poi.town_id.eq(tid)&poi.landmark_type.eq(typ)].poi_id)})
   rows.append(d)
  pd.DataFrame(rows).to_csv(path,index=False);return len(rows),dirty,over,pd.Series([x['n_landmarks'] for x in rows]).value_counts().to_dict(),pd.Series([not x['locality_name'] for x in rows]).mean()
 vr=output(val.address_id,OUT/'ner_output_validation.csv');hold=pd.read_csv(P/'temp_code'/'holdout_ids_noout.csv',dtype=str);hr=output(hold.address_id,OUT/'ner_output_holdout.csv');report=f'validation rows {vr[0]}\nholdout rows {hr[0]}\nvalidation locality empty share {vr[4]:.3%}\nvalidation landmarks 0/1/2 {vr[3]}\ncleaned spans containing digit/town word: 0 (removed; {vr[1]} spans changed)\nrows >2 spans before cap {vr[2]}\nvalidation entity F1 {md[(md.set=="validation")&(md.metric=="entity_f1")].value.iloc[0]} vs 0.8579\n';(OUT/'ner_output_report.txt').write_text(report,encoding='utf8');print(report)
if __name__=='__main__':main()
