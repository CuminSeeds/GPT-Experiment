"""加载混合分词实验权重，支持 CPU 分类和续写。"""
import argparse,json
import torch
from model import load_model
from prepare_data import clean_sms,prompt_for
from token_embedding_experiment import ROOT,OUT,MixedTokenizer,read,generate_mixed

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=['mixed_random','mixed_cbow'],default='mixed_cbow')
    p.add_argument('--mode',choices=['spam','generate'],default='spam')
    p.add_argument('--text',default='Are we meeting tonight?')
    p.add_argument('--prompt',default='The king');p.add_argument('--length',type=int,default=120)
    a=p.parse_args();torch.set_num_threads(4)
    tokenizer=MixedTokenizer(read(OUT/'tokenizer.json')['words'])
    checkpoint=OUT/a.variant/'weights'/('finetuned.pt' if a.mode=='spam' else 'pretrained.pt')
    model=load_model(checkpoint,'cpu')
    if a.mode=='generate': print(generate_mixed(model,tokenizer,a.prompt,n=a.length));return
    prompt=prompt_for(clean_sms(a.text))
    with torch.no_grad():
        logits=model(torch.tensor([tokenizer.encode(prompt)]))[0,-1]
        label=int(logits[[48,49]].argmax())
    print(json.dumps({'label':label,'category':'spam' if label else 'ham',
          'raw_first_token':tokenizer.decode([int(logits.argmax())])},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
