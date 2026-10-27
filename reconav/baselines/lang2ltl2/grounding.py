"""Released REG/SPG with scene-only ASG text/landmarks as its map input."""
from reconav import paths as release_paths
import ast
import copy
import json
import os
from pathlib import Path
import pickle
from itertools import product
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from reconav.baselines.common.protocol import file_digest, write
from reconav.baselines.common.upstream import definitions
from reconav.baselines.common.formal_language import PredictionFailure

def load_file(path):
    path = Path(path)
    return pickle.loads(path.read_bytes()) if path.suffix == '.pkl' else json.loads(path.read_text())

def save_file(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.pkl':
        path.write_bytes(pickle.dumps(value))
    else:
        write(path, value)

def ground(frontend, records, policy, output, embeddings, cache):
    root = release_paths.upstream_path('Lang2LTL-2')
    namespace = {'os': os, 'Path': Path, 'np': np, 'cosine_similarity': cosine_similarity, 'get_embed': embeddings, 'load_from_file': load_file, 'save_to_file': save_file, 'tqdm': tqdm}
    reg = definitions(root / 'reg.py', ['REG', 'reg'], namespace)
    descriptions = {k: dict(r['description'], name=k) for k, r in records.items()}
    vectors = embeddings.many(list(descriptions.values()))
    reg['embed_texts'] = lambda *_: dict(zip(descriptions, vectors))
    reg['load_from_file'] = lambda *_: {}
    query_cache = Path(cache) / 'queries.pkl'
    reg['REG'].__init__.__globals__['load_from_file'] = load_file
    map_dir = Path(cache) / policy['scene_id']
    save_file({}, map_dir / 'obj_locs.json')
    save_file(descriptions, map_dir / 'landmarks.json')
    parsed = copy.deepcopy(frontend['original_srer'])
    reg['reg'](str(map_dir), str(map_dir / 'landmarks.json'), [parsed], 10, 'image', str(query_cache))
    spg_namespace = dict(namespace, product=product, MAX_RANGE=2.0, DIST_TO_ANCHOR=2.0, plt=plt)
    tree = ast.parse((root / 'spg.py').read_text())
    assignment = next((n for n in tree.body if isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'KNOWN_RELATIONS' for t in n.targets))))
    spg_namespace['KNOWN_RELATIONS'] = ast.literal_eval(assignment.value)
    spg = definitions(root / 'spg.py', None, spg_namespace)
    locations = {k: {'x': r['position'][0], 'y': r['position'][2]} for k, r in records.items()}
    locations['robot'] = {'x': policy['start_position'][0], 'y': policy['start_position'][2]}
    result = spg['spg'](locations, parsed, 10, str(Path(cache) / 'known_rel_embeds.json'), max_range=2.0)
    parsed['grounded_sps'] = result
    bindings = {}
    for symbol, sre in parsed['lifted_symbol_map'].items():
        choices = result.get(sre, [])
        if not choices or choices[0] is None:
            write(output / 'grounding.json', parsed)
            raise PredictionFailure('original_SPG_no_grounding: ' + sre)
        first = choices[0]
        bindings[symbol] = first.get('target', first)
    parsed['sym2ground'] = bindings
    parsed['source_sha256'] = {str(root / n): file_digest(root / n) for n in ('reg.py', 'spg.py')}
    parsed['protocol'] = {'topk': 10, 'spg_max_range_m': 2.0, 'input': 'all located scene ASG landmarks', 'missing_geometry': 'unlocated entities cannot be landmarks; recorded separately', 'original_enumerate_query_serialization_preserved': True, 'scene_embeddings': 'original text-embedding-3-large/3072', 'selected_grounding': 'first original SPG candidate, no oracle reranking'}
    write(output / 'grounding.json', parsed)
    return bindings
