"""核对保存后模型与本地 HTTP 服务的结果，并保存真实部署记录。"""
from pathlib import Path
import json, subprocess, sys, time, urllib.request, urllib.error
import torch
from model import load_model
from infer import classify
from prepare_data import DATA

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    model=load_model(ROOT/'weights'/'finetuned.pt')
    examples=json.loads((ROOT/'results'/'classification_examples.json').read_text(encoding='utf-8'))
    checks=[]
    for e in examples:
        result=classify(model,e['text'])
        assert result['label']==e['prediction']
        checks.append({'test_index':e['index'],'saved_prediction':e['prediction'],'cpu_prediction':result['label']})
    # 全部 CPU 测试预测与训练结束的记录核对，避免只核对展示样本。
    predictions=json.loads((ROOT/'results'/'test_predictions.json').read_text(encoding='utf-8'))
    cpu_predictions=[classify(model,r['text'])['label'] for r in predictions]
    matches=sum(p==r['prediction'] for p,r in zip(cpu_predictions,predictions))
    assert matches==len(predictions)
    port=18765
    proc=subprocess.Popen([sys.executable,'-X','utf8',str(ROOT/'serve.py'),'--port',str(port)],
                          stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                          creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    records=[]
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as r:
                    health=json.load(r)
                break
            except (urllib.error.URLError,TimeoutError): time.sleep(.2)
        else:raise RuntimeError('服务启动失败')
        for e in examples:
            data=json.dumps({'text':e['text']}).encode('utf-8')
            req=urllib.request.Request(f'http://127.0.0.1:{port}/predict',data=data,headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=10) as r: result=json.load(r)
            assert result['label']==e['prediction']
            records.append({'test_index':e['index'],'text':e['text'],'true_label':e['label'],'response':result})
    finally:
        proc.terminate();proc.wait(timeout=10)
    result={'health':health,'cpu_test_matches':matches,'test_count':len(predictions),'reload_checks':checks,'http_requests':records}
    (ROOT/'results'/'deployment_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
