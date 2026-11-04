"""Independent non-tool checks: fresh regression items (new seed) and prose-QA prompts (no tools, prose answers)."""
import json, hashlib
from benchmark.regression import generate_regression
reg=generate_regression(32,60132)
open('data/regression2.jsonl','w').write(''.join(json.dumps({'id':t['input'].id,'user':t['input'].user,'domain':t['domain'],'gold':t['gold']})+'\n' for t in reg))
PROSE=["Explain in two sentences what a hash table is.","Why does bread rise when it bakes?","Describe how to boil an egg.","What is the difference between weather and climate?","Explain what a prime number is.","Why is the sky blue?","Describe how a bicycle brake works.","What does a compiler do?","Explain the water cycle briefly.","Why do leaves change color in autumn?","Describe what happens during photosynthesis.","What is inflation in economics?","Explain how a thermostat works.","Why do we have time zones?","Describe how to plant a tree.","What is the purpose of a firewall in computing?"]
open('data/prose_qa.jsonl','w').write(''.join(json.dumps({'id':f'prose-{i:02d}','user':u,'domain':'prose','gold':None})+'\n' for i,u in enumerate(PROSE)))
for n in ('regression2','prose_qa'): print(n, sum(1 for _ in open(f'data/{n}.jsonl')), hashlib.sha256(open(f'data/{n}.jsonl','rb').read()).hexdigest()[:16])
