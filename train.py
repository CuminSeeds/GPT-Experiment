"""顺序运行真实预训练与监督微调；测试集只在验证集选定模型后评估。"""
from pathlib import Path
from dataclasses import asdict
import argparse, json, math, platform, random, time
import numpy as np
import torch
from torch.nn import functional as F
from model import Config, TinyGPT, PAD, encode, generate, load_model, save_model
from prepare_data import DATA, prompt_for

ROOT = Path(__file__).resolve().parent
RESULTS, WEIGHTS = ROOT/'results', ROOT/'weights'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SETTINGS = dict(seed=42, split_seed=2026, pretrain_steps=1800, pretrain_lr=.001,
                finetune_epochs=20, finetune_lr=.0003, batch_size=32,
                eval_every=100, optimizer='Adam', betas=[.9,.999], eps=1e-8,
                weight_decay=0, selection='minimum validation cross entropy')


def write(name, obj):
    (RESULTS/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


@torch.no_grad()
def lm_eval(model, ids):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    c = model.cfg.context
    starts = list(range(0,len(ids)-1,c))
    for offset in range(0,len(starts),32):
        group = starts[offset:offset+32]
        x = torch.full((len(group),c), PAD, dtype=torch.long)
        y = torch.full((len(group),c), -100, dtype=torch.long)
        for j,start in enumerate(group):
            size = min(c,len(ids)-start-1)
            x[j,:size],y[j,:size] = ids[start:start+size],ids[start+1:start+size+1]
        x,y=x.to(DEVICE),y.to(DEVICE)
        logits=model(x)
        total_loss += F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100,reduction='sum').item()
        correct += ((logits.argmax(-1)==y)&(y!=-100)).sum().item()
        count += (y!=-100).sum().item()
    return {'loss':total_loss/count, 'accuracy':correct/count, 'tokens':count}


def pretrain(cfg=None):
    seed_all(42)
    cfg = cfg or Config()
    model=TinyGPT(cfg).to(DEVICE)
    # 初始化检查只使用合成字符串，不读取测试数据。
    model.eval()
    a=torch.tensor([encode('hello there')],device=DEVICE); b=a.clone();b[:,7:]=42
    with torch.no_grad():
        delta=(model(a)[:,:7]-model(b)[:,:7]).abs().max().item()
    assert delta == 0 and encode('hello') == [104,101,108,108,111]
    env={'python':platform.python_version(),'torch':torch.__version__,'device':DEVICE,
         'gpu':torch.cuda.get_device_name(0) if DEVICE=='cuda' else None,
         'parameters':sum(p.numel() for p in model.parameters()),'config':asdict(cfg),
         'settings':SETTINGS,'causal_prefix_max_difference':delta}
    write('environment.json',env)
    save_model(model,WEIGHTS/'initial.pt')
    train_ids=torch.tensor(encode((DATA/'lm_train.txt').read_text(encoding='utf-8')),dtype=torch.long)
    val_ids=torch.tensor(encode((DATA/'lm_val.txt').read_text(encoding='utf-8')),dtype=torch.long)
    # 曲线的训练损失用固定的前 49152 个训练字符，评估时关闭 Dropout。
    train_eval=train_ids[:49152]
    history=[]; best=float('inf'); best_step=0
    rng=torch.Generator().manual_seed(42)
    opt=torch.optim.Adam(model.parameters(),lr=SETTINGS['pretrain_lr'])
    t0=time.perf_counter()
    initial_train,initial_val=lm_eval(model,train_eval),lm_eval(model,val_ids)
    history.append({'step':0,'train':initial_train,'val':initial_val})
    print('pretrain',json.dumps(history[-1]),flush=True)
    for step in range(1,SETTINGS['pretrain_steps']+1):
        model.train()
        starts=torch.randint(len(train_ids)-cfg.context,(SETTINGS['batch_size'],),generator=rng)
        x=torch.stack([train_ids[s:s+cfg.context] for s in starts]).to(DEVICE)
        y=torch.stack([train_ids[s+1:s+cfg.context+1] for s in starts]).to(DEVICE)
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x).flatten(0,1),y.flatten())
        loss.backward(); opt.step()
        if step%SETTINGS['eval_every']==0:
            tr,va=lm_eval(model,train_eval),lm_eval(model,val_ids)
            row={'step':step,'train':tr,'val':va}
            history.append(row);write('pretrain_history.json',history)
            if va['loss']<best:
                best,best_step=va['loss'],step
                save_model(model,WEIGHTS/'pretrained.pt')
            print('pretrain',json.dumps(row),flush=True)
    elapsed=time.perf_counter()-t0
    best_model=load_model(WEIGHTS/'pretrained.pt',DEVICE)
    test_ids=torch.tensor(encode((DATA/'lm_test.txt').read_text(encoding='utf-8')),dtype=torch.long)
    initial=load_model(WEIGHTS/'initial.pt',DEVICE)
    result={'best_step':best_step,'seconds':elapsed,'initial_train':initial_train,'initial_val':initial_val,
            'initial_test':lm_eval(initial,test_ids),'train':lm_eval(best_model,train_eval),
            'val':lm_eval(best_model,val_ids),'test':lm_eval(best_model,test_ids),
            'train_evaluation_note':'固定前49152个训练字符；验证和测试集全部评估；每段192字符独立上下文',
            'processed_training_tokens':SETTINGS['pretrain_steps']*SETTINGS['batch_size']*cfg.context}
    write('pretrain_metrics.json',result)
    samples=[]
    for p in ['ROMEO:\n','First Citizen:\n','The king']:
        samples.append({'prompt':p,'seed':2026,'new_characters':240,'temperature':.8,
                        'before':generate(initial,p),'after':generate(best_model,p)})
    write('generation_samples.json',samples)
    print('PRETRAIN_DONE',json.dumps(result),flush=True)


def sms_tensors(rows, context=192):
    prompts=[encode(prompt_for(r['text'],context)) for r in rows]
    x=torch.full((len(rows),context),PAD,dtype=torch.long)
    positions=torch.tensor([len(p)-1 for p in prompts],dtype=torch.long)
    for i,p in enumerate(prompts): x[i,:len(p)]=torch.tensor(p)
    labels=torch.tensor([r['label'] for r in rows],dtype=torch.long)
    return x,positions,labels


def batch_logits(model,tensors,indices):
    x,pos,labels=[v[indices].to(DEVICE) for v in tensors]
    # 只计算答案字符的监督损失，不要求复制短信正文。
    logits=model(x)[torch.arange(len(indices),device=DEVICE),pos]
    return logits,labels


def scores(labels,pred):
    tp=sum(y==1 and p==1 for y,p in zip(labels,pred))
    tn=sum(y==0 and p==0 for y,p in zip(labels,pred))
    fp=sum(y==0 and p==1 for y,p in zip(labels,pred))
    fn=sum(y==1 and p==0 for y,p in zip(labels,pred))
    precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn)
    f1=2*tp/max(1,2*tp+fp+fn)
    ham_f1=2*tn/max(1,2*tn+fp+fn)
    return {'accuracy':(tp+tn)/len(labels),'spam_precision':precision,'spam_recall':recall,
            'spam_f1':f1,'macro_f1':(f1+ham_f1)/2,'tp':tp,'tn':tn,'fp':fp,'fn':fn,'n':len(labels)}


@torch.no_grad()
def sms_eval(model,tensors,return_predictions=False):
    model.eval(); total=0.;pred=[];labels=[];raw=[]
    for start in range(0,len(tensors[0]),32):
        ids=torch.arange(start,min(start+32,len(tensors[0])))
        logits,y=batch_logits(model,tensors,ids)
        target=y+ord('0')
        total+=F.cross_entropy(logits,target,reduction='sum').item()
        pred.extend(logits[:,[ord('0'),ord('1')]].argmax(-1).cpu().tolist())
        labels.extend(y.cpu().tolist());raw.extend(logits.argmax(-1).cpu().tolist())
    result=scores(labels,pred)
    result.update(loss=total/len(labels),raw_valid_fraction=sum(p in (48,49) for p in raw)/len(raw),
                  raw_accuracy=sum(p==y+48 for p,y in zip(raw,labels))/len(raw))
    if return_predictions: return result,pred,raw
    return result


def finetune():
    seed_all(43)
    rows={k:json.loads((DATA/f'sms_{k}.json').read_text(encoding='utf-8')) for k in ('train','val')}
    ts={k:sms_tensors(v) for k,v in rows.items()}
    model=load_model(WEIGHTS/'pretrained.pt',DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=SETTINGS['finetune_lr'])
    rng=torch.Generator().manual_seed(43)
    history=[{'epoch':0,'train':sms_eval(model,ts['train']),'val':sms_eval(model,ts['val'])}]
    best=float('inf');best_epoch=0;t0=time.perf_counter()
    print('finetune',json.dumps(history[-1]),flush=True)
    for epoch in range(1,SETTINGS['finetune_epochs']+1):
        model.train()
        order=torch.randperm(len(rows['train']),generator=rng)
        for start in range(0,len(order),SETTINGS['batch_size']):
            ids=order[start:start+SETTINGS['batch_size']]
            logits,y=batch_logits(model,ts['train'],ids)
            loss=F.cross_entropy(logits,y+ord('0'))
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        tr,va=sms_eval(model,ts['train']),sms_eval(model,ts['val'])
        history.append({'epoch':epoch,'train':tr,'val':va});write('finetune_history.json',history)
        if va['loss']<best:
            best,best_epoch=va['loss'],epoch
            save_model(model,WEIGHTS/'finetuned.pt')
        print('finetune',json.dumps(history[-1]),flush=True)
    elapsed=time.perf_counter()-t0
    # 到此模型已按验证集选定，随后才加载测试集。
    test_rows=json.loads((DATA/'sms_test.json').read_text(encoding='utf-8'))
    test_ts=sms_tensors(test_rows)
    best_model=load_model(WEIGHTS/'finetuned.pt',DEVICE)
    pretrained=load_model(WEIGHTS/'pretrained.pt',DEVICE)
    result,pred,raw=sms_eval(best_model,test_ts,True)
    labels=[r['label'] for r in test_rows]
    metrics={'best_epoch':best_epoch,'seconds':elapsed,
             'train':sms_eval(best_model,ts['train']),'val':sms_eval(best_model,ts['val']),
             'test':result,'before_test':sms_eval(pretrained,test_ts),
             'majority_test':scores(labels,[0]*len(labels)),
             'decision_rule':'选答案位置输出的字符0和1中分数较大者；同时记录不限制词表时的原始首字符准确率'}
    write('finetune_metrics.json',metrics)
    predictions=[dict(index=i,**r,prediction=pred[i],raw_character=chr(raw[i]) if raw[i]<128 else '?') for i,r in enumerate(test_rows)]
    write('test_predictions.json',predictions)
    # 展示样例的选择规则固定：测试集顺序中每类前3条，不按预测是否正确筛选。
    examples=[]
    for label in (0,1): examples.extend([r for r in predictions if r['label']==label][:3])
    write('classification_examples.json',examples)
    print('FINETUNE_DONE',json.dumps(metrics),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pretrain','finetune','all'],default='all')
    args=p.parse_args(); RESULTS.mkdir(exist_ok=True);WEIGHTS.mkdir(exist_ok=True)
    if args.stage in ('pretrain','all'): pretrain()
    if args.stage in ('finetune','all'): finetune()
