"""Five-fold grouped train-only NER check. Do not run until explicitly requested."""
from pathlib import Path
import shutil,time,importlib.util
import pandas as pd,numpy as np,torch
from sklearn.model_selection import GroupKFold
from seqeval.metrics import precision_score,recall_score,f1_score,classification_report
from rapidfuzz import fuzz,process,utils
from transformers import AutoTokenizer,AutoModelForTokenClassification,DataCollatorForTokenClassification,Trainer,TrainingArguments,set_seed

P=Path(__file__).resolve().parents[1]
DATA=P/'data'/'labels'/'train_bio_merged_v2_noout.csv'
HOLD=P/'temp_code'/'holdout_ids_noout.csv'
ADDRESSES=P/'data'/'addresses.csv'

def main():
    labels=pd.read_csv(DATA,dtype=str);hold=set(pd.read_csv(HOLD,dtype=str).address_id)
    addresses=pd.read_csv(ADDRESSES,dtype=str,usecols=['address_id','account_id','town_id'])
    towns=pd.read_csv(P/'data'/'towns.csv',dtype=str)
    for name,table in [('train_labels',labels),('holdout',pd.read_csv(HOLD,dtype=str)),('addresses',addresses)]:print(name,list(table.columns))
    ids=[x for x in labels.address_id.drop_duplicates() if x not in hold]
    print('addresses after holdout exclusion:',len(ids))
    if len(ids)!=1920:raise ValueError('expected 1920 already-holdout-excluded train addresses')
    groups=addresses.set_index('address_id').loc[ids,'account_id'].to_numpy()
    splitter=GroupKFold(n_splits=5)
    settings=P/'code_files'/'04_train_ner.py';spec=importlib.util.spec_from_file_location('base',settings);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    metrics=[];tmp=P/'temp_code'/'kfold_tmp';tmp.mkdir(exist_ok=True)
    for fold,(train_index,test_index) in enumerate(splitter.split(ids,groups=groups),1):
        train_accounts=set(groups[train_index]);test_accounts=set(groups[test_index])
        print(f'fold {fold}: train addresses {len(train_index)}, test addresses {len(test_index)}, shared accounts {len(train_accounts&test_accounts)}')
        if train_accounts&test_accounts:raise ValueError('account leakage in fold')
        started=time.time();set_seed(42)
        train_ids=[ids[i] for i in train_index];test_ids=[ids[i] for i in test_index]
        train_rows=[];test_rows=[]
        for aid in train_ids:
            g=labels[labels.address_id.eq(aid)].sort_values('word_index',key=lambda s:s.astype(int));rng=np.random.default_rng(42+len(train_rows));words=[base.typo(x,__import__('random').Random(42+len(train_rows))) if __import__('random').Random(42+len(train_rows)).random()<.1 else x for x in g.word];train_rows.append((words,g.tag.tolist()))
        for aid in test_ids:
            g=labels[labels.address_id.eq(aid)].sort_values('word_index',key=lambda s:s.astype(int));test_rows.append((g.word.tolist(),g.tag.tolist()))
        fold_dir=tmp/f'fold_{fold}';tokenizer=AutoTokenizer.from_pretrained(base.MODEL_NAME);model=AutoModelForTokenClassification.from_pretrained(base.MODEL_NAME,num_labels=len(base.LABELS),id2label=dict(enumerate(base.LABELS)),label2id=base.LABEL_TO_ID)
        args=TrainingArguments(output_dir=str(fold_dir),learning_rate=3e-5,per_device_train_batch_size=16,per_device_eval_batch_size=16,num_train_epochs=5,fp16=True,seed=42,report_to=[],save_strategy='no',eval_strategy='no')
        trainer=Trainer(model=model,args=args,train_dataset=base.BioDataset(train_rows,tokenizer),data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer))
        trainer.train();pred=trainer.predict(base.BioDataset(test_rows,tokenizer));p=np.argmax(pred.predictions,-1);true=[];guess=[];allok=[];non=[]
        for pp,ll in zip(p,pred.label_ids):
            keep=ll!=-100;gt=[base.LABELS[x] for x in ll[keep]];pr=[base.LABELS[x] for x in pp[keep]];true.append(gt);guess.append(pr);allok += [a==b for a,b in zip(gt,pr)];non += [a==b for a,b in zip(gt,pr) if a!='O']
        report=classification_report(true,guess,output_dict=True,zero_division=0)
        values={'token_accuracy':sum(allok)/len(allok),'token_accuracy_non_o':sum(non)/len(non),'entity_precision_without_pincode':precision_score(true,guess),'entity_recall_without_pincode':recall_score(true,guess),'entity_f1_without_pincode':f1_score(true,guess)}
        for entity in ['TOWN','LOCALITY','LANDMARK','RELATION']:
            for metric in ['precision','recall','f1-score']:values[f'{entity}_{metric}']=report.get(entity,{}).get(metric,0.0)
        for metric,value in values.items():metrics.append((fold,metric,value))
        town_ok=[]; locality_ok=[]
        for aid, predicted_tags, gold_tags in zip(test_ids, guess, true):
            raw_words=test_rows[test_ids.index(aid)][0]
            def entity_words(tags, entity):
                current=[]; found=[]
                for word,tag in zip(raw_words,tags):
                    if tag==f'B-{entity}':
                        if current: found.append(' '.join(current))
                        current=[word]
                    elif tag==f'I-{entity}' and current: current.append(word)
                    else:
                        if current: found.append(' '.join(current));current=[]
                if current: found.append(' '.join(current))
                return found
            town_span=(entity_words(predicted_tags,'TOWN') or [''])[0]
            match=process.extractOne(town_span,towns.town_name.tolist(),scorer=fuzz.WRatio,processor=utils.default_process) if town_span else None
            predicted_town=towns.iloc[match[2]].town_id if match and match[1]>=85 else ''
            true_town=addresses.loc[addresses.address_id.eq(aid),'town_id'].iloc[0]
            town_ok.append(predicted_town==true_town)
            locality_ok.append(entity_words(predicted_tags,'LOCALITY')==entity_words(gold_tags,'LOCALITY'))
        metrics.append((fold,'town_accuracy_true_addresses',sum(town_ok)/len(town_ok)))
        metrics.append((fold,'locality_exact_including_none',sum(locality_ok)/len(locality_ok)))
        elapsed=time.time()-started;metrics.append((fold,'seconds',elapsed));print(f'fold {fold} completed in {elapsed:.1f}s')
        del trainer,model;torch.cuda.empty_cache();shutil.rmtree(fold_dir)
    result=pd.DataFrame(metrics,columns=['fold','metric','value']);P.joinpath('outputs').mkdir(exist_ok=True);result.to_csv(P/'outputs'/'kfold_metrics.csv',index=False)
    summary=result.groupby('metric').value.agg(['mean','std']);text='Final model is ner_v2_best, not a fold model. Folds use train labels only; the 90 holdout rows were excluded.\n\n'+summary.to_string();(P/'outputs'/'kfold_report.txt').write_text(text,encoding='utf8');print(text)
if __name__=='__main__':main()
