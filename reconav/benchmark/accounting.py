"""Metric aggregation for newly generated experiment runs."""

def mean(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return dict(mean=sum(values) / len(values) if values else None, n=len(values))

def aggregate(rows, total, complete):
    done = [r for r in rows if complete(r)]
    return dict(planned=total, observed=len(rows), completed=len(done), completion_rate=len(done) / total, means={key: mean(done, key) for key in ('full_is', 'hard_ns', 'prefer', 'seconds', 'total_tokens')})

def scoring_complete(row):
    return row['execution_status'] == 'completed' and row.get('scoring_completed') and (not row.get('resource_reason'))

def fixed_quality(rows, population, preference_tasks, complete=scoring_complete):
    """Fixed, predeclared denominator; never drop terminal failures or pending IDs."""
    population = set(population)
    selected = [r for r in rows if r['task_id'] in population]
    assert len(selected) == len({r['task_id'] for r in selected}), 'duplicate task'
    if {r['task_id'] for r in selected} != population or any((r['execution_status'] not in {'completed', 'timeout', 'resource_exhausted', 'failed', 'unsupported_input'} for r in selected)):
        return dict(pending=True, quality_denominator=len(population))
    sums = dict(full_is=0.0, hard_ns=0.0, prefer=0.0)
    for row in selected:
        if not complete(row):
            continue
        for key in ('full_is', 'hard_ns'):
            assert row.get(key) is not None, (row['task_id'], key)
            sums[key] += row[key]
        if row['task_id'] in preference_tasks:
            assert row.get('prefer') is not None, (row['task_id'], 'prefer')
            sums['prefer'] += row['prefer']
    n = len(population)
    pn = len(population & set(preference_tasks))
    return dict(pending=False, quality_denominator=n, quality_task_ids=sorted(population), quality_policy='fixed_supported_terminal_noncompletion_zero', preference_policy='fixed_preference_applicable_terminal_noncompletion_zero', prefer_denominator=pn, quality={k: dict(mean=v / (pn if k == 'prefer' else n) if (pn if k == 'prefer' else n) else None, n=pn if k == 'prefer' else n, numerator=v) for k, v in sums.items()})
