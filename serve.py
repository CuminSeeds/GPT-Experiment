"""简易本地 HTTP 部署：python serve.py，然后访问 http://127.0.0.1:8765 。"""
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import argparse, json, time
import torch
from infer import classify
from model import load_model

PAGE='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>短信分类实验</title>
<body><h2>小型 GPT 短信分类</h2><p>输入英文短信，判断为普通短信或垃圾短信。</p>
<textarea id="text" rows="5" cols="65">Are we meeting for dinner tonight?</textarea><br>
<button onclick="run()">判断</button><p id="result"></p>
<script>async function run(){let r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:document.getElementById('text').value})});let d=await r.json();document.getElementById('result').textContent=d.category==='spam'?'垃圾短信':'普通短信';}</script></body></html>'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765)
    a=parser.parse_args();torch.set_num_threads(4)
    model=load_model(Path(__file__).resolve().parent/'weights'/'finetuned.pt')
    class Handler(BaseHTTPRequestHandler):
        def respond(self,obj,status=200):
            raw=json.dumps(obj,ensure_ascii=False).encode('utf-8')
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.end_headers();self.wfile.write(raw)

        def do_GET(self):
            if self.path=='/health':return self.respond({'status':'ready','device':'cpu'})
            if self.path!='/':return self.respond({'error':'not found'},404)
            self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
            self.end_headers();self.wfile.write(PAGE.encode('utf-8'))

        def do_POST(self):
            if self.path!='/predict':return self.respond({'error':'not found'},404)
            size=int(self.headers.get('Content-Length','0'))
            if size>20000:return self.respond({'error':'input too long'},400)
            try:
                data=json.loads(self.rfile.read(size)); text=data['text']
                if not isinstance(text,str):raise ValueError('text must be string')
                start=time.perf_counter(); result=classify(model,text)
                result['inference_ms']=(time.perf_counter()-start)*1000
                self.respond(result)
            except (ValueError,KeyError): self.respond({'error':'invalid text'},400)
    print(f'Ready: http://127.0.0.1:{a.port}',flush=True)
    HTTPServer(('127.0.0.1',a.port),Handler).serve_forever()


if __name__=='__main__':main()
