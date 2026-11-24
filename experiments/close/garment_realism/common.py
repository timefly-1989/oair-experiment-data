import hashlib,json,os
from pathlib import Path

ROOT=Path(os.environ.get('OAIR_RESEARCH_ROOT', Path.cwd()))
STUDY=ROOT/'realism_v6_20261104'

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2));os.replace(temporary,path)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def records(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def source_hash():
    root=Path(__file__).resolve().parents[1];h=hashlib.sha256()
    for package in ['garment_realism','garment_study','garment_lab','garment_expansion']:
        for p in sorted((root/package).glob('*.py')):h.update(p.relative_to(root).as_posix().encode());h.update(p.read_bytes())
    return h.hexdigest()
