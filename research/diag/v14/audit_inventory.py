"""Inventory of every function/method in src/team048_solution/team048_solution (ast) with line coverage from the
pytest runs (coverage json, Rust and TEAM048_ACCEL=0 combined) and the test files that name it.
Usage: python audit_inventory.py <cov_rust.json> <cov_numpy.json> > table.md"""
import ast,json,os,sys,glob,re
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..');PKG=os.path.join(R,'src/team048_solution/team048_solution')
cov=[json.load(open(f))['files'] for f in sys.argv[1:3]]
tests={os.path.basename(f):open(f).read() for f in glob.glob(os.path.join(R,'src/team048_solution/test/*.py'))}
def executed(fn):
    ex=set();mi=set()
    for c in cov:
        k=[k for k in c if k.endswith('team048_solution/'+fn)]
        if k:ex|=set(c[k[0]]['executed_lines']);mi|=set(c[k[0]]['missing_lines'])
    return ex,mi-ex
rows=[];tot_l=tot_e=0
for f in sorted(glob.glob(PKG+'/*.py')):
    fn=os.path.basename(f);t=ast.parse(open(f).read());ex,mi=executed(fn)
    def visit(node,prefix):
        for n in ast.iter_child_nodes(node):
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
                body=set(range(n.body[0].lineno,n.end_lineno+1))
                inner=set()
                for m in ast.walk(n):
                    if m is not n and isinstance(m,(ast.FunctionDef,ast.ClassDef)):inner|=set(range(m.lineno,m.end_lineno+1))
                stm=(ex|mi)&(body-inner) if False else (ex|mi)&body
                e=len(stm&ex);L=len(stm)
                name=prefix+n.name
                tn=sorted(k for k,v in tests.items() if re.search(r'\b'+re.escape(n.name)+r'\b',v))
                rows.append((fn,n.lineno,name,L,e,tn))
                visit(n,name+'.')
            elif isinstance(n,ast.ClassDef):visit(n,prefix+n.name+'.')
    visit(t,'')
print('| Файл:строка | Функция | Строк (исполн.) | Покрыто | Тесты, где упомянута |\n|---|---|---:|---:|---|')
for fn,l,name,L,e,tn in rows:
    pct='—' if L==0 else f'{100*e/L:.0f}%'
    print(f'| `{fn}:{l}` | `{name}` | {L} | {pct} | {", ".join(t.replace(".py","") for t in tn[:4])}{" …" if len(tn)>4 else ""} |')
n=len(rows);cov0=sum(1 for r in rows if r[3] and r[4]==0);full=sum(1 for r in rows if r[3] and r[4]==r[3])
print(f'\nФункций и методов: {n}; полностью покрыты: {full}; ни одной исполненной строки: {cov0}; без упоминания в тестах: {sum(1 for r in rows if not r[5])}',file=sys.stderr)
for r in rows:
    if r[3] and r[4]==0:print('UNCOVERED',r[0],r[1],r[2],file=sys.stderr)
