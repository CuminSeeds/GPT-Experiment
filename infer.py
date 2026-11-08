"""保存模型后的本地推理。只需 CPU；分类沿用训练时的提示与字符答案。"""
from pathlib import Path
import argparse, json, time
import torch
from model import encode, generate, load_model
from prepare_data import clean_sms, prompt_for

ROOT=Path(__file__).resolve().parent


@torch.no_grad()
def classify(model,text):
    prompt=prompt_for(clean_sms(text),model.cfg.context)
    dev=next(model.parameters()).device
    x=torch.tensor([encode(prompt)],device=dev)
    logits=model(x)[0,-1]
    prob=torch.softmax(logits[[48,49]],dim=-1)
    label=int(prob.argmax().item())
    raw=int(logits.argmax().item())
    return {'label':label,'category':'spam' if label else 'ham',
            'spam_probability_among_labels':float(prob[1]),
            'raw_first_character':chr(raw) if raw<128 else '?',
            'input_characters_used':len(prompt)-16}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['spam','generate'],default='spam')
    p.add_argument('--text',default='Are we still meeting for dinner tonight?')
    p.add_argument('--prompt',default='ROMEO:\n');p.add_argument('--length',type=int,default=240)
    p.add_argument('--checkpoint',type=Path)
    a=p.parse_args();torch.set_num_threads(4)
    checkpoint=a.checkpoint or ROOT/'weights'/('finetuned.pt' if a.mode=='spam' else 'pretrained.pt')
    m=load_model(checkpoint)
    if a.mode=='spam': print(json.dumps(classify(m,a.text),ensure_ascii=False,indent=2))
    else: print(generate(m,a.prompt,n=a.length))
