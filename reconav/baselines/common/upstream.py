"""Load upstream algorithm definitions and select their runtime imports."""
import ast
import importlib.util


def verify_recorded_source(root, directory):
    """Require the pinned dependency and its declared paper-specific source patch."""
    import hashlib
    import json
    from pathlib import Path
    import subprocess
    manifest = json.loads((Path(__file__).parents[1] / 'upstream.json').read_text())
    entry = next(x for x in manifest if x['directory'] == directory)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if head != entry['commit']:
        raise RuntimeError(f'{directory} is not at the recorded revision')
    for name, expected in entry.get('patched_files', {}).items():
        if hashlib.sha256((Path(root) / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'{directory}: apply the recorded patch {entry["patch"]} before running')


def definitions(path, names=None, namespace=None):
    namespace = {} if namespace is None else namespace
    selected = [n for n in ast.parse(path.read_text()).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and (names is None or n.name in names)]
    if names is not None and {n.name for n in selected} != set(names):
        raise RuntimeError('upstream definitions changed: ' + str(path))
    namespace['__file__'] = str(path)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result
