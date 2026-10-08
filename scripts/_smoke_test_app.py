import sys, json
sys.path.insert(0, '.')
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location('annotation_app', 'app/annotation_app.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

ROOT = Path('.')
app = mod.create_app(
    ROOT / 'data/annotations/annotation_queue.csv',
    mode='seed',
    seed_path=ROOT / 'data/annotations/active_learning_seed.csv',
    batch_path=ROOT / 'data/annotations/next_active_learning_batch.csv',
    db_path=ROOT / 'data/annotations/annotations.csv',
)
c = app.test_client()

r = c.get('/api/progress')
prog = json.loads(r.data)
print('Seed mode progress:', prog)
assert prog['total'] == 100, 'Expected 100 seed clauses'

r = c.get('/annotate/0')
assert r.status_code == 200
assert b'Party Identification' in r.data
assert b'INITIAL SEED ANNOTATION' in r.data
print('Seed mode annotate/0: OK')

app2 = mod.create_app(
    ROOT / 'data/annotations/annotation_queue.csv', mode='all',
    seed_path=ROOT / 'data/annotations/active_learning_seed.csv',
    db_path=ROOT / 'data/annotations/annotations.csv',
)
c2 = app2.test_client()
r2 = c2.get('/api/progress')
prog2 = json.loads(r2.data)
print('All mode progress:', prog2)
assert prog2['total'] == 699
print('All smoke tests PASSED')
