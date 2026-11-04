"""Independent deterministic non-tool tasks; never used to derive an edit."""
from __future__ import annotations
import random,json
from benchmark.contracts import ModelInput

def generate_regression(n=384,seed=60131):
    r=random.Random(seed);out=[]
    for i in range(n):
        domain=['instruction','extraction','arithmetic','structured'][i%4]
        a,b=r.randrange(3,80),r.randrange(3,80); word=r.choice(['birch','maple','cedar','spruce'])+str(i)
        if domain=='instruction': user=f'Reply with exactly the word {word} and nothing else.';gold=word
        elif domain=='extraction': user=f'Extract the value of target. Reply with that value only. distractor={a}; target={word}; other={b}';gold=word
        elif domain=='arithmetic': user=f'What is {a} + {b}? Reply with the integer only.';gold=str(a+b)
        else:user=f'Return only a JSON object with key "name" set to "{word}" and key "count" set to the integer {a}.';gold={'name':word,'count':a}
        out.append({'input':ModelInput(f'regression-{seed}-{i}',user,[]),'domain':domain,'gold':gold})
    return out

def check(task,text):
    if task['domain']=='structured':
        try:
            v=json.loads(text)
            return type(v)==dict and v==task['gold'] and type(v.get('count'))==int
        except (ValueError,TypeError): return False
    return text.strip()==task['gold']
