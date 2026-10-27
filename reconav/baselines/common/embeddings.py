"""Lang2LTL-2 embedding cache and API client with a shared spending budget."""
import fcntl
import os
import json
import math
from pathlib import Path
import time
import uuid
import zipfile
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from reconav.baselines.common.protocol import digest, read, write
from reconav.baselines.common.client import BudgetExceeded


class Embeddings:
    model = 'text-embedding-3-large'
    price_per_million = .13

    def __init__(self, cache, ledger, key_file, cap=.05):
        if not math.isfinite(cap) or cap < 0:
            raise ValueError('embedding budget must be finite and nonnegative')
        self.cache, self.ledger = Path(cache), Path(ledger)
        self.key_file = Path(key_file) if key_file else None
        self.cap = cap
        self.calls = []
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        from reconav.paths import lang2ltl_embeddings
        bundle = lang2ltl_embeddings()
        self.bundle = zipfile.ZipFile(bundle) if bundle.is_file() else None
        self.bundled_keys = set(self.bundle.namelist()) if self.bundle else set()

    @staticmethod
    def serialize(value):
        import json
        return json.dumps(value).replace('\n', ' ')

    def many(self, values):
        texts = [self.serialize(v) for v in values]
        keys = [digest([self.model,t]) for t in texts]
        missing = {k:t for k,t in zip(keys,texts)
                   if not (self.cache/(k+'.json')).is_file() and k+'.json' not in self.bundled_keys}
        for offset in range(0,len(missing),64):
            batch = list(missing.items())[offset:offset+64]
            # UTF-8 byte bound dominates token counts. No context clipping.
            if any(len(t.encode()) > 8192 for _,t in batch):
                raise BudgetExceeded('embedding input exceeds conservative token admission')
            bound = sum(len(t.encode()) for _,t in batch)
            reserved = bound*self.price_per_million/1e6
            if self.cap == 0:
                raise BudgetExceeded('embedding dollar cap reached; no request sent')
            key = self.key_file.read_text().strip() if self.key_file else (os.environ.get('RECONAV_EMBEDDING_API_KEY') or os.environ.get('OPENAI_API_KEY', '')).strip()
            if not key or '\n' in key:
                raise RuntimeError('Missing embedding API key: set RECONAV_EMBEDDING_API_KEY or pass --embedding-key-file')
            call_id = uuid.uuid4().hex
            with self.ledger.with_suffix('.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                book = read(self.ledger) if self.ledger.exists() else {'model':self.model,'calls':[]}
                if book['model'] != self.model:
                    raise RuntimeError('embedding ledger model mismatch')
                spent = sum(c.get('charged_usd',c['reserved_usd']) for c in book['calls'])
                if spent+reserved > self.cap:
                    raise BudgetExceeded('embedding dollar cap reached; no request sent')
                book['calls'].append({'id':call_id,'reserved_usd':reserved,'state':'pending',
                                      'text_sha256': [k for k,_ in batch]})
                write(self.ledger,book)
            payload = {'model':self.model,'input':[t for _,t in batch],'encoding_format':'float'}
            request = Request('https://api.openai.com/v1/embeddings', data=json.dumps(payload).encode(),
                              headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
            started = time.monotonic()
            try:
                with urlopen(request,timeout=120) as response:
                    body = json.load(response)
            except Exception as error:
                detail=f'HTTP {error.code}' if isinstance(error,HTTPError) else type(error).__name__
                with self.ledger.with_suffix('.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    book=read(self.ledger)
                    row=next(c for c in book['calls'] if c['id']==call_id)
                    row.update(state='transport_failed_unknown_charge',error=detail,
                               seconds=time.monotonic()-started)
                    write(self.ledger,book)
                    self.calls.append(dict(row))
                raise RuntimeError(f'embedding {detail}; reservation retained, no retry') from None
            usage = body['usage']
            charge = usage['prompt_tokens']*self.price_per_million/1e6
            rows = sorted(body['data'],key=lambda r:r['index'])
            if len(rows)!=len(batch) or any(len(r['embedding'])!=3072 for r in rows):
                raise RuntimeError('embedding shape mismatch')
            with self.ledger.with_suffix('.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                book = read(self.ledger)
                row = next(c for c in book['calls'] if c['id']==call_id)
                row.update(state='settled',usage=usage,charged_usd=charge,seconds=time.monotonic()-started)
                write(self.ledger,book)
                self.calls.append(dict(row))
            for (k,t),r in zip(batch,rows):
                write(self.cache/(k+'.json'),{'model':self.model,'serialized_text':t,'vector':r['embedding']})
        vectors = []
        for key, text in zip(keys, texts):
            path = self.cache / (key+'.json')
            row = read(path) if path.is_file() else json.loads(self.bundle.read(key+'.json'))
            if row['model'] != self.model or row['serialized_text'] != text or len(row['vector']) != 3072:
                raise ValueError('embedding cache identity or dimensions do not match')
            vectors.append(row['vector'])
        return vectors

    def __call__(self,value):
        return self.many([value])[0]
