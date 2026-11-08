"""高频单词/字符混合分词，比较随机词向量与训练集 CBOW 词向量初始化。"""
from pathlib import Path
from collections import Counter
from dataclasses import asdict
import argparse, hashlib, json, re, time
import torch
from torch import nn
from torch.nn import functional as F
import train
from model import Config, TinyGPT, save_model, load_model

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'token_embedding'
DATA=ROOT/'data'
PAD,UNK=128,129

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

class MixedTokenizer:
    """130个基础编号加382个高频完整单词；数字和标点仍逐字符编码。"""
    pattern=re.compile(r'[A-Za-z]+|[^A-Za-z]')
    def __init__(self,words):
        self.words=words
        self.lookup={word:i+130 for i,word in enumerate(words)}
        self.vocab_size=130+len(words)
    @classmethod
    def fit(cls,text):
        counts=Counter(re.findall(r'[A-Za-z]{2,}',text))
        return cls([word for word,count in sorted(counts.items(),key=lambda p:(-p[1],p[0]))[:382]])
    def encode(self,text):
        result=[]
        for piece in self.pattern.findall(text):
            if piece in self.lookup:result.append(self.lookup[piece])
            else:result.extend(ord(c) if ord(c)<128 else UNK for c in piece)
        return result
    def decode(self,ids):
        return ''.join(self.words[i-130] if i>=130 else chr(i) if i<128 else '?' for i in ids)

@torch.no_grad()
def generate_mixed(model,tokenizer,prompt,n=120,seed=2026,temperature=.8):
    device=next(model.parameters()).device
    ids=torch.tensor([tokenizer.encode(prompt)],device=device)
    rng=torch.Generator(device=device).manual_seed(seed)
    allowed=[10]+list(range(32,127))+list(range(130,tokenizer.vocab_size))
    model.eval()
    for _ in range(n):
        logits=model(ids[:,-model.cfg.context:])[:,-1]/temperature
        banned=torch.ones(tokenizer.vocab_size,dtype=torch.bool,device=device);banned[allowed]=False
        logits[:,banned]=float('-inf')
        ids=torch.cat([ids,torch.multinomial(logits.softmax(-1),1,generator=rng)],1)
    return tokenizer.decode(ids[0].tolist())

def cbow_vectors(ids,vocab_size):
    path=OUT/'cbow_vectors.pt'
    if path.exists():return torch.load(path,map_location='cpu',weights_only=True)['vectors']
    train.seed_all(44)
    embedding=nn.Embedding(vocab_size,128).to(train.DEVICE)
    output=nn.Linear(128,vocab_size).to(train.DEVICE)
    nn.init.normal_(embedding.weight,0,.02)
    nn.init.normal_(output.weight,0,.02);nn.init.zeros_(output.bias)
    opt=torch.optim.Adam(list(embedding.parameters())+list(output.parameters()),lr=.001)
    tokens=ids.to(train.DEVICE);rng=torch.Generator().manual_seed(44)
    history=[];start=time.perf_counter()
    for epoch in range(1,6):
        order=torch.randperm(len(ids)-4,generator=rng)+2
        total=0.;count=0
        for j in range(0,len(order),512):
            centers=order[j:j+512].to(train.DEVICE)
            neighbors=centers[:,None]+torch.tensor([-2,-1,1,2],device=train.DEVICE)
            logits=output(embedding(tokens[neighbors]).mean(1))
            loss=F.cross_entropy(logits,tokens[centers])
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            total+=loss.item()*len(centers);count+=len(centers)
        row={'epoch':epoch,'loss':total/count};history.append(row)
        print('CBOW',json.dumps(row),flush=True)
    vectors=embedding.weight.detach().cpu()
    vectors=vectors*(.02/vectors.std())
    torch.save({'vectors':vectors},path)
    write(OUT/'cbow_history.json',{'history':history,'seconds':time.perf_counter()-start,
          'epochs':5,'window_each_side':2,'batch_size':512,'learning_rate':.001,'seed':44,
          'data':'lm_train.txt only','embedding_width':128,'initialization_std':float(vectors.std())})
    return vectors

def pretrain(tokenizer,variant,ids,vectors=None):
    train.seed_all(42)
    cfg=Config(vocab_size=tokenizer.vocab_size,position_encoding='rope',heads=4)
    model=TinyGPT(cfg).to(train.DEVICE)
    if vectors is not None:
        with torch.no_grad():model.embedding.weight.copy_(vectors.to(train.DEVICE))
    model.eval();a=torch.arange(11,device=train.DEVICE)[None];b=a.clone();b[:,7:]=42
    with torch.no_grad():delta=(model(a)[:,:7]-model(b)[:,:7]).abs().max().item()
    assert delta==0
    train.write('environment.json',{'config':asdict(cfg),'parameters':sum(p.numel() for p in model.parameters()),
          'variant':variant,'settings':train.SETTINGS,'causal_prefix_max_difference':delta,
          'embedding_initialization':'CBOW training vectors, rescaled to std .02' if vectors is not None else 'normal std .02'})
    save_model(model,train.WEIGHTS/'initial.pt')
    train_ids=ids
    val_ids=torch.tensor(tokenizer.encode((DATA/'lm_val.txt').read_text(encoding='utf-8')))
    train_eval=train_ids[:49152]
    history=[{'step':0,'train':train.lm_eval(model,train_eval),'val':train.lm_eval(model,val_ids)}]
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    rng=torch.Generator().manual_seed(42);best=float('inf');best_step=0;t0=time.perf_counter()
    for step in range(1,1801):
        model.train();starts=torch.randint(len(train_ids)-192,(32,),generator=rng)
        x=torch.stack([train_ids[s:s+192] for s in starts]).to(train.DEVICE)
        y=torch.stack([train_ids[s+1:s+193] for s in starts]).to(train.DEVICE)
        loss=F.cross_entropy(model(x).flatten(0,1),y.flatten())
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step%100==0:
            row={'step':step,'train':train.lm_eval(model,train_eval),'val':train.lm_eval(model,val_ids)}
            history.append(row);train.write('pretrain_history.json',history)
            if row['val']['loss']<best:
                best=row['val']['loss'];best_step=step;save_model(model,train.WEIGHTS/'pretrained.pt')
            print(variant,'PRETRAIN',json.dumps(row),flush=True)
    elapsed=time.perf_counter()-t0
    selected=load_model(train.WEIGHTS/'pretrained.pt',train.DEVICE)
    test_ids=torch.tensor(tokenizer.encode((DATA/'lm_test.txt').read_text(encoding='utf-8')))
    metrics={'best_step':best_step,'seconds':elapsed,'train':train.lm_eval(selected,train_eval),
             'val':train.lm_eval(selected,val_ids),'test':train.lm_eval(selected,test_ids),
             'unit':'mixed word/character tokens, not characters','processed_training_tokens':1800*32*192}
    train.write('pretrain_metrics.json',metrics)
    train.write('generation_samples.json',[{'prompt':p,'new_tokens':120,'seed':2026,
               'after':generate_mixed(selected,tokenizer,p)} for p in ['ROMEO:\n','The king']])

def summarize():
    models=[]
    for name,folder in [('character_random',ROOT/'improvements/rope_h4'),
                        ('mixed_random',OUT/'mixed_random'),('mixed_cbow',OUT/'mixed_cbow')]:
        if (folder/'results/finetune_metrics.json').exists():
            models.append({'name':name,'parameters':628226 if name=='character_random' else 726400,
                'vocab_size':130 if name=='character_random' else 512,
                'pretrain':read(folder/'results/pretrain_metrics.json'),
                'finetune':read(folder/'results/finetune_metrics.json')})
    write(OUT/'comparison.json',{'models':models,'reference':'RoPE, 4 heads, width 128, 3 blocks',
          'sms_inputs':'same cleaned first 176 characters before tokenization',
          'pretrain_metric_unit':'each model token; character and mixed losses/accuracies are not directly comparable',
          'selection':'minimum validation loss within each model; no test-based selection'})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['all','mixed_random','mixed_cbow'],default='all')
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    text=(DATA/'lm_train.txt').read_text(encoding='utf-8')
    tokenizer=MixedTokenizer.fit(text)
    assert tokenizer.vocab_size==512
    for sample in [text[:5000],'Hello unseenWORD123!\nLabel:0','Message: Are we meeting tonight?\nLabel:']:
        assert tokenizer.decode(tokenizer.encode(sample))==sample
    assert tokenizer.encode('0')==[48] and tokenizer.encode('1')==[49]
    write(OUT/'tokenizer.json',{'words':tokenizer.words,'base_vocab':130,'vocab_size':512,
           'vocabulary_fitted_on':'lm_train.txt only','pattern':tokenizer.pattern.pattern})
    ids=torch.tensor(tokenizer.encode(text),dtype=torch.long)
    sms=json.loads((DATA/'sms_train.json').read_text(encoding='utf-8'))
    sms_lengths=[len(tokenizer.encode(train.prompt_for(row['text']))) for row in sms]
    write(OUT/'settings.json',{'training':train.SETTINGS,'word2vec':'CBOW, window 2, 128 dimensions, 5 epochs, train corpus only',
       'lm_train_characters':len(text),'lm_train_tokens':len(ids),'sms_train_mean_tokens':sum(sms_lengths)/len(sms_lengths),
       'sms_train_max_tokens':max(sms_lengths),'tokenization_example':{'text':'I love learning',
       'ids':tokenizer.encode('I love learning'),'pieces':[tokenizer.decode([i]) for i in tokenizer.encode('I love learning')]},
       'data_sha256':{n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in ['lm_train.txt','lm_val.txt','lm_test.txt','sms_train.json','sms_val.json','sms_test.json']}})
    train.encode=tokenizer.encode
    for name in ['mixed_random','mixed_cbow'] if args.variant=='all' else [args.variant]:
        folder=OUT/name;train.RESULTS=folder/'results';train.WEIGHTS=folder/'weights'
        train.RESULTS.mkdir(parents=True,exist_ok=True);train.WEIGHTS.mkdir(parents=True,exist_ok=True)
        print('VARIANT_START',name,flush=True)
        if not (train.RESULTS/'pretrain_metrics.json').exists():
            vectors=cbow_vectors(ids,512) if name=='mixed_cbow' else None
            pretrain(tokenizer,name,ids,vectors)
        if not (train.RESULTS/'finetune_metrics.json').exists():train.finetune()
        print('VARIANT_DONE',name,flush=True)
        summarize()

if __name__=='__main__':main()
