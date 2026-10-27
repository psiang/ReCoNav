"""Authoritative TSV vocabularies plus the explicit benchmark semantic policy."""
from __future__ import annotations
from reconav import paths as release_paths
import csv
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
SEMANTIC_POLICY_PATH = Path(__file__).with_name('semantic_policy.json')
DEFAULT_OBJECT_VOCABULARY_PATH = Path(str(release_paths.data_root() / 'benchmark/vocabularies/hm3dsem_category_mappings.tsv'))

@dataclass(frozen=True)
class ObjectCategory:
    raw: str
    canonical: str
    mpcat40: str
    mpcat40_all: frozenset[str] = frozenset()

@lru_cache(maxsize=1)
def load_semantic_policy() -> dict[str, Any]:
    value = json.loads(SEMANTIC_POLICY_PATH.read_text(encoding='utf-8'))
    if value.get('policy_name') != 'benchmark_step1_manual_semantic_policy':
        raise ValueError(f'unexpected semantic policy: {SEMANTIC_POLICY_PATH}')
    return value


@lru_cache(maxsize=1)
def _surface_head_aliases() -> dict[str, str]:
    return {_normalize(key): _normalize(value) for key, value in load_semantic_policy()['surface_head_aliases'].items()}

def object_head_alias(head: str) -> str:
    normalized = _normalize(head)
    return _surface_head_aliases().get(normalized, normalized)


def load_object_categories(path: Path) -> dict[str, ObjectCategory]:
    with Path(path).open('r', encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    if not rows or set(rows[0]) != {'raw_category', 'category', 'mpcat40'}:
        raise ValueError(f'unexpected object vocabulary schema: {path}')
    normalized_rows = [(_normalize(row['raw_category']), _normalize(row['category']), _normalize(row['mpcat40'])) for row in rows]
    canonical_mpcats: dict[str, set[str]] = {}
    for _raw, canonical, mpcat in normalized_rows:
        canonical_mpcats.setdefault(canonical, set()).add(mpcat)
    result: dict[str, ObjectCategory] = {}
    for raw, canonical, mpcat in normalized_rows:
        record = ObjectCategory(raw=raw, canonical=canonical, mpcat40=mpcat, mpcat40_all=frozenset(canonical_mpcats[canonical]))
        if raw in result:
            prior = result[raw]
            if prior.canonical != canonical or prior.mpcat40 != mpcat:
                raise ValueError(f'conflicting normalized object category {raw!r}: {prior} != {record}')
            continue
        result[raw] = record
    return result

@lru_cache(maxsize=1)
def default_object_categories() -> dict[str, ObjectCategory]:
    configured = os.environ.get('HM3DSEM_OBJECT_VOCABULARY')
    if configured:
        path = Path(configured).expanduser().resolve()
    else:
        path = release_paths.data_root() / 'benchmark/vocabularies/hm3dsem_category_mappings.tsv'
    return load_object_categories(path)


def canonical_mpcat40_sets(categories: dict[str, ObjectCategory]) -> dict[str, frozenset[str]]:
    result: dict[str, set[str]] = {}
    for record in categories.values():
        result.setdefault(record.canonical, set()).update(record.mpcat40_all or {record.mpcat40})
    return {key: frozenset(values) for key, values in result.items()}


def object_confusion_groups(categories: dict[str, ObjectCategory]) -> dict[str, frozenset[str]]:
    """Compute the appendix's directional, non-recursive denotations exactly."""
    policy = load_semantic_policy()
    canonical_mpcats = canonical_mpcat40_sets(categories)
    canonicals = frozenset(canonical_mpcats)
    buckets: dict[str, set[str]] = {canonical: {canonical} for canonical in canonicals}
    for raw_head, rule in policy['generic_head_rules'].items():
        head = _normalize(raw_head)
        if head not in canonicals:
            continue
        selected_mpcats = {_normalize(value) for value in rule.get('mpcat40', ())}
        denotation = {canonical for canonical, mpcats in canonical_mpcats.items() if mpcats & selected_mpcats}
        denotation.update((_normalize(value) for value in rule.get('add', ()) if _normalize(value) in canonicals))
        denotation.difference_update((_normalize(value) for value in rule.get('remove', ())))
        denotation.add(head)
        buckets[head] = denotation
    synonym_groups = [frozenset((_normalize(value) for value in group if _normalize(value) in canonicals)) for group in policy['synonym_groups']]
    synonym_groups = [group for group in synonym_groups if group]
    for group in synonym_groups:
        combined: set[str] = set()
        for spelling in group:
            combined.update(buckets[spelling])
        for spelling in group:
            buckets[spelling] = set(combined)
    spelling_group = {spelling: group for group in synonym_groups for spelling in group}
    result: dict[str, frozenset[str]] = {}
    for head, denotation in buckets.items():
        expanded: set[str] = set()
        for label in denotation:
            expanded.update(spelling_group.get(label, frozenset({label})))
        expanded.add(head)
        result[head] = frozenset(expanded)
    for alias, target in policy['surface_head_aliases'].items():
        normalized_alias = _normalize(alias)
        normalized_target = _normalize(target)
        if normalized_target not in result:
            raise ValueError(f'semantic policy alias {normalized_alias!r} targets absent canonical {normalized_target!r}')
        result[normalized_alias] = result[normalized_target]
    return result
NON_ROOM_REGION_LABELS = frozenset({'hallway', 'staircase', 'garage', 'balcony', 'porch'})
ORDINARY_PORTAL_LABELS = frozenset({'door', 'opening', 'untyped'})

def region_confusion_groups(labels: frozenset[str]) -> dict[str, frozenset[str]]:
    result = {label: frozenset({label}) for label in labels}
    for head, denotation in load_semantic_policy()['region_generic_rules'].items():
        normalized_head = _normalize(head)
        if normalized_head in result:
            result[normalized_head] = frozenset((_normalize(value) for value in denotation if _normalize(value) in labels))
    result['room'] = labels - NON_ROOM_REGION_LABELS
    return result


def _normalize(value: str) -> str:
    return ' '.join(str(value).strip().lower().replace('_', ' ').split())
