"""Per-episode execution budgets and process supervision.

The clock covers task initialization, perception, language and motion.
Scoring, batch preflight and resident API-server startup are outside the clock.
Uncapped execution retains method-specific action/search bounds and transport timeouts."""
import math
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from reconav.baselines.common.protocol import digest, file_digest, read, write
TERMINAL = {'completed', 'unsupported_input', 'timeout', 'resource_exhausted'}
RESOURCE_TERMINATIONS = {'token_budget_exhausted', 'search_budget_exhausted'}

def diagnostic_read(path, default):
    try:
        return read(path)
    except (OSError, ValueError):
        return default

def add_arguments(parser):
    parser.add_argument('--split', choices=['all', 'hard44'], default='all', help='Task population; timing is selected separately with --time-protocol.')
    parser.add_argument('--time-protocol', type=Path)
    parser.add_argument('--time-limit-seconds', type=float, help='Override capped-task wall budget; 0 means uncapped for all selected tasks.')
    parser.add_argument('--partition', choices=['all', 'capped', 'hard'], default='all')
    parser.add_argument('--population', choices=['all', 'common'], default='all')

class TimePolicy:

    def __init__(self, args, tasks):
        self.doc = read(args.time_protocol) if args.time_protocol else None
        self.limit = args.time_limit_seconds
        if self.limit is not None and (not math.isfinite(self.limit) or self.limit < 0):
            raise ValueError('wall budget must be finite and nonnegative')
        if not self.doc and (args.partition != 'all' or args.population != 'all'):
            raise ValueError('partition/population selection requires --time-protocol')
        self.ids = set(self.doc['uncapped_task_ids']) if self.doc else set()
        if self.doc:
            if len(self.ids) != self.doc['subset_size'] or not self.ids <= set(self.doc['common_task_ids']):
                raise ValueError('invalid uncapped subset')
            for path in tasks:
                if self.doc['task_sha256'].get(path.stem) != file_digest(path):
                    raise ValueError('time protocol task identity changed: ' + path.stem)
            if self.limit is None:
                self.limit = self.doc['capped_seconds']
        self.tasks = [p for p in tasks if (args.partition == 'all' or (p.stem in self.ids) == (args.partition == 'hard')) and (args.population == 'all' or p.stem in self.doc['common_task_ids'])]
        if not self.tasks:
            raise ValueError('empty time-protocol task selection')

    @property
    def enabled(self):
        return self.doc is not None or self.limit is not None

    def settings(self, task_id):
        if not self.enabled:
            return {}
        hard = task_id in self.ids
        return {'time_protocol': {'version': 'wall-budget-conditional-metrics-v2-compute-only', 'manifest_sha256': digest(self.doc) if self.doc else None, 'partition': 'hard' if hard else 'capped', 'limit_seconds': 0 if hard else self.limit, 'clock': 'task_initialization_through_policy_end_excluding_scoring', 'resource_exhaustion_is_incomplete': True, 'resource_exclusion_policy': 'time_token_solver_only', 'native_steps_and_feedback_repair_limits_scored': True}}

    def deadline(self, task_id):
        settings = self.settings(task_id)
        return Deadline(settings['time_protocol']['limit_seconds']) if settings else None

class BudgetTimeout(TimeoutError):
    pass

class ResourceExhausted(RuntimeError):
    pass

class Deadline:

    def __init__(self, seconds):
        self.seconds = seconds
        self.started = time.monotonic()
        self.finished = None

    def remaining(self):
        if not self.seconds or self.finished is not None:
            return None
        return self.seconds - (time.monotonic() - self.started)

    def environment(self, env, cell):
        return dict(env, EXTERNAL_POLICY_DONE=str(Path(cell) / 'policy_done.json'), EXTERNAL_POLICY_STARTED=str(self.started))

    def observe(self, cell):
        marker = Path(cell) / 'policy_done.json'
        if marker.exists():
            doc = read(marker)
            if doc.get('started_monotonic') == self.started:
                ended = doc['finished_monotonic']
                if self.seconds and ended - self.started > self.seconds:
                    raise BudgetTimeout('policy finished after wall deadline')
                self.finished = ended

    def elapsed(self):
        return (self.finished or time.monotonic()) - self.started

def mark_policy_complete():
    """Called by policy workers immediately before the ground-truth evaluator."""
    path = os.environ.get('EXTERNAL_POLICY_DONE')
    if path:
        write(path, {'started_monotonic': float(os.environ['EXTERNAL_POLICY_STARTED']), 'finished_monotonic': time.monotonic()})

def terminate_group(proc, grace=2.0):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()

def wait_process(proc, timeout, budget=None, cell=None, fatal_log=None):
    if budget is None and fatal_log is None:
        try:
            return proc.wait(timeout=timeout if timeout and timeout > 0 else None)
        except BaseException:
            terminate_group(proc)
            raise
    scoring_started = None
    started = time.monotonic()
    log_offset = 0
    log_tail = ''
    try:
        while True:
            if fatal_log is not None:
                with Path(fatal_log).open(errors='replace') as log:
                    log.seek(log_offset)
                    new_text = log.read()
                    log_offset = log.tell()
                text = log_tail + new_text
                if 'Exception in thread ' in text:
                    raise RuntimeError(f'native environment thread crashed; see {fatal_log}')
                log_tail = text[-128:]
            if budget is not None:
                budget.observe(cell)
            elif timeout and timeout > 0 and time.monotonic() - started > timeout:
                raise subprocess.TimeoutExpired(proc.args, timeout)
            if budget is not None and budget.finished is not None:
                if scoring_started is None:
                    scoring_started = time.monotonic()
                if time.monotonic() - scoring_started > 900:
                    raise RuntimeError('post-policy cleanup/scoring watchdog; not a policy timeout')
            left = budget.remaining() if budget is not None else None
            if left is not None and left <= 0:
                raise BudgetTimeout('episode wall budget exhausted')
            try:
                code = proc.wait(timeout=min(0.1, left) if left is not None else 0.1)
            except subprocess.TimeoutExpired:
                continue
            if budget is not None:
                budget.observe(cell)
            left = budget.remaining() if budget is not None else None
            if left is not None and left <= 0:
                raise BudgetTimeout('episode wall budget exhausted')
            return code
    except BaseException:
        terminate_group(proc)
        raise

def resource_reason(status, cell=None):
    termination = status.get('termination')
    if termination in RESOURCE_TERMINATIONS:
        return termination
    if status.get('execution_status') == 'resource_exhausted':
        return status.get('resource_reason', 'resource_exhausted')
    error = str(status.get('error', ''))
    if 'embedding dollar cap reached' in error or 'budget admission denied:' in error:
        return 'dollar_budget_exhausted'
    if any((s in error for s in ('episode token budget:', 'input admission limit;', 'embedding input exceeds conservative token admission'))):
        return 'token_budget_exhausted'
    if 'local transport safety limit reached' in error:
        return 'request_count_limit'
    if cell is not None:
        cell = Path(cell)
        from reconav.baselines.common.context_retry import eligible_context_failure
        if eligible_context_failure(cell):
            return 'context_length_exceeded'
        for filename in ('worker_status.json', 'trajectory.json'):
            if (cell / filename).exists():
                record = diagnostic_read(cell / filename, {})
                reason = resource_reason({**status, **record})
                if reason:
                    return reason
        for filename in ('calls.json', 'llm_api_provenance.json'):
            if (cell / filename).exists():
                records = diagnostic_read(cell / filename, [])
                calls = records if isinstance(records, list) else records.get('calls', [])
                if calls and calls[-1].get('finish_reason') in {'length', 'max_tokens'}:
                    return 'output_token_limit'
                if isinstance(records, dict) and records.get('admission_denials'):
                    return 'token_budget_exhausted'
        admission = cell / 'last_token_admission.json'
        if admission.exists() and diagnostic_read(admission, {}).get('admitted') is False:
            return 'token_budget_exhausted'
    return None

def internal_resource_reason(record):
    """Use final failure plus positive budget evidence, never any failed query.

    A recovered intermediate timeout is not an unfinished final prediction.
    UNSAT/search-space exhaustion is a normal algorithm conclusion, not a cap.
    Feedback/repair iteration exhaustion is scored as a completed outcome.
    """
    failure = record.get('failure') or {}
    text = json.dumps(failure, ensure_ascii=False).lower()
    reason = failure.get('reason', '')
    if reason == 'UNKNOWN' and any((s in text for s in ('budget', 'cancel', 'timeout', 'time limit'))):
        return 'solver_budget_exhausted'
    if 'SOLVER_UNKNOWN' in str(failure):
        if any((s in text for s in ('querybudgetexhausted', 'solver budget exhausted', 'solver time limit'))):
            return 'solver_budget_exhausted'
        for query in record.get('solver_queries', []):
            detail = str(query.get('status_reason', '')).lower()
            if any((s in detail for s in ('budget', 'cancel', 'timeout', 'time limit'))):
                return 'solver_budget_exhausted'
    if failure and str(reason).upper() in {'SOLVER_BUDGET_EXHAUSTED', 'QUERY_BUDGET_EXHAUSTED', 'TIME_BUDGET_EXHAUSTED', 'TOKEN_BUDGET_EXHAUSTED', 'TOKEN_LIMIT', 'CONTEXT_BUDGET_EXCEEDED', 'INPUT_BUDGET_EXCEEDED'}:
        return 'internal_resource_exhausted'
    return None

def incomplete(status, cell, kind, reason):
    status.update(execution_status=kind, scoring_completed=False, metrics_eligible=False, resource_reason=reason)
    path = Path(cell) / 'metrics.json'
    if path.exists():
        raw = diagnostic_read(path, None)
        if raw is not None:
            write(Path(cell) / 'metrics_unbudgeted_diagnostic.json', raw)
    write(path, {'full_is': None, 'hard_ns': None, 'prefer': None, 'scoring_completed': False, 'excluded_reason': reason})

def record_plan(output, entries):
    path = Path(output) / 'evaluation_plan.json'
    old = read(path)['entries'] if path.exists() else []
    key = lambda r: (r['method'], r.get('condition', 'native'), r['task_id'])
    merged = {key(r): r for r in old}
    for row in entries:
        if key(row) in merged and merged[key(row)] != row:
            raise RuntimeError('evaluation protocol changed; use a new output directory')
        merged[key(row)] = row
    write(path, {'entries': sorted(merged.values(), key=key)})

def reporting(output, rows):
    """Summarize fixed populations and completion-conditioned quality."""
    plan_path = Path(output) / 'evaluation_plan.json'
    entries = read(plan_path)['entries'] if plan_path.exists() else rows
    groups = []
    keys = sorted({(r['method'], r.get('condition', 'native')) for r in entries})
    for method, condition in keys:
        for partition in ('all', 'capped', 'hard'):

            def selected(r):
                return r['method'] == method and r.get('condition', 'native') == condition and (partition == 'all' or r.get('time_protocol', {}).get('partition') == partition)
            planned = [r for r in entries if selected(r)]
            if not planned:
                continue
            planned_ids = {r['task_id'] for r in planned}
            actual = [r for r in rows if selected(r) and r['task_id'] in planned_ids]
            eligible = [r for r in planned if r.get('supported', r.get('execution_status') != 'unsupported_input')]
            scored = [r for r in actual if r.get('execution_status') == 'completed' and r.get('scoring_completed', True) and (not resource_reason(r))]
            count = len(eligible)
            timeouts = sum((r.get('execution_status') == 'timeout' for r in actual))
            resource = sum((r.get('execution_status') == 'resource_exhausted' or (r.get('execution_status') == 'completed' and bool(resource_reason(r))) for r in actual))
            quality = {}
            for field in ('full_is', 'hard_ns', 'prefer'):
                values = [r[field] for r in scored if r.get(field) is not None]
                quality[field] = {'mean': sum(values) / len(values) if values else None, 'n': len(values)}
            groups.append({'method': method, 'condition': condition, 'partition': partition, 'planned': len(planned), 'supported': count, 'unsupported': len(planned) - count, 'observed': len(actual), 'pending': len(planned) - len(actual), 'completed': len(scored), 'completion_rate': len(scored) / count if count else None, 'timeouts': timeouts, 'timeout_rate': timeouts / count if count else None, 'resource_exhausted': resource, 'resource_exhaustion_rate': resource / count if count else None, 'failed': sum((r.get('execution_status') == 'failed' for r in actual)), 'running': sum((r.get('execution_status') == 'running' for r in actual)), 'quality_completed_only': quality})
    return {'groups': groups, 'definition': 'Completion means normal policy termination and completed scoring. Rates use the fixed supported population, including pending tasks; conditional quality uses completed scored tasks, with Prefer restricted to preference-applicable tasks. All combines the selected execution budgets.'}
