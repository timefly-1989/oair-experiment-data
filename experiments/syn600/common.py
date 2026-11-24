from pathlib import Path
import datetime,hashlib,json,os,subprocess,time
from zoneinfo import ZoneInfo
ROOT=Path(os.environ.get('OAIR_SYN600_RUN_ROOT', Path(__file__).resolve().parent))
BASE=ROOT.parent
SOURCE=BASE/'realism_v6_20261104/source_versions/1de5e0bf7f05fb372077a0e11ac73c86c95b2164cadbf951b0351c03bd4cad4d'
PY=BASE/'env/bin/python'
ENV=dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(SOURCE),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYOPENGL_PLATFORM='egl',EGL_DEVICE_ID='1')
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 return h.hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));t.replace(p)
def now():return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def wait_gpu(index,minimum=2000):
 while True:
  r=subprocess.run(['nvidia-smi','-i',str(index),'--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
  if int(r.stdout.strip())>=minimum:return
  time.sleep(15)
def execute(args,name):
 log=ROOT/'logs'/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('a') as f:
  r=subprocess.run([str(PY)]+[str(x) for x in args],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(name+' failed: '+str(r.returncode)+'; '+str(log))
