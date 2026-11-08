"""检查词表来源、可逆分词、模型选择、初始化对照和全部测试预测的 CPU 重载。"""
import hashlib,json
from pathlib import Path
import torch
import train
from model import load_model
from token_embedding_experiment import OUT,DATA,MixedTokenizer,read,write

def main():
    torch.set_num_threads(4);train.DEVICE='cpu'
    tokenizer=MixedTokenizer(read(OUT/'tokenizer.json')['words'])
    assert tokenizer.words==MixedTokenizer.fit((DATA/'lm_train.txt').read_text(encoding='utf-8')).words
    settings=read(OUT/'settings.json')
    for name,sha in settings['data_sha256'].items(): assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==sha
    for split in ['train','val','test']:
        text=(DATA/f'lm_{split}.txt').read_text(encoding='utf-8')
        assert tokenizer.decode(tokenizer.encode(text))==text
        for row in read(DATA/f'sms_{split}.json'):
            prompt=train.prompt_for(row['text'])
            assert tokenizer.decode(tokenizer.encode(prompt))==prompt
            assert len(tokenizer.encode(prompt))<=192
    train.encode=tokenizer.encode
    rows=read(DATA/'sms_test.json');tensors=train.sms_tensors(rows)
    results={}
    for name in ['mixed_random','mixed_cbow']:
        folder=OUT/name
        ft=read(folder/'results/finetune_metrics.json')
        hist=read(folder/'results/finetune_history.json')
        assert ft['best_epoch']==min(hist[1:],key=lambda r:r['val']['loss'])['epoch']
        lm=read(folder/'results/pretrain_metrics.json')
        lm_hist=read(folder/'results/pretrain_history.json')
        assert lm['best_step']==min(lm_hist[1:],key=lambda r:r['val']['loss'])['step']
        model=load_model(folder/'weights/finetuned.pt','cpu')
        metrics,pred,raw=train.sms_eval(model,tensors,True)
        saved=read(folder/'results/test_predictions.json')
        assert pred==[r['prediction'] for r in saved]
        for key in ['accuracy','spam_precision','spam_recall','spam_f1']:assert abs(metrics[key]-ft['test'][key])<1e-12
        assert len(pred)==517 and sum(p.numel() for p in model.parameters())==726400
        results[name]={'cpu_test_matches':len(pred),'metrics':metrics,'selected_epoch':ft['best_epoch']}
    a=torch.load(OUT/'mixed_random/weights/initial.pt',weights_only=True,map_location='cpu')['state']
    b=torch.load(OUT/'mixed_cbow/weights/initial.pt',weights_only=True,map_location='cpu')['state']
    assert all(torch.equal(a[k],b[k]) for k in a if k!='embedding.weight')
    assert not torch.equal(a['embedding.weight'],b['embedding.weight'])
    vectors=torch.load(OUT/'cbow_vectors.pt',weights_only=True)['vectors']
    assert torch.equal(vectors,b['embedding.weight'])
    verification={'roundtrip_all_data':True,'vocabulary_train_only':True,
          'same_nonembedding_initial_parameters':True,'original_data_unchanged':True,'models':results}
    write(OUT/'verification.json',verification)
    print(json.dumps(verification,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
