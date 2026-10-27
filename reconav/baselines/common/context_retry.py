"""Measure context-capacity failures for protocol retry classification."""
import json
from pathlib import Path
from urllib.request import Request, urlopen

from reconav.baselines.common.protocol import digest, read, write


def confirm_context_rejection(output, endpoint, payload, http_status, error, call_id):
    """Only HTTP context rejections with measured overflow qualify for TP2."""
    if http_status != 400 or not any(s in error.lower() for s in
            ('context length', 'context_length', 'max_model_len', 'maximum model length')):
        return None
    try:
        body = {k: payload[k] for k in ('model', 'messages', 'chat_template_kwargs') if k in payload}
        body['add_generation_prompt'] = True
        request = Request(endpoint.removesuffix('/v1') + '/tokenize',
                          data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=20) as response:
            counted = json.load(response)
        with urlopen(endpoint + '/models', timeout=10) as response:
            models = json.load(response)['data']
        model = next(r for r in models if r['id'] == payload['model'])
        count, limit, reserve = counted['count'], model['max_model_len'], payload['max_tokens']
        if not all(isinstance(n, int) and n > 0 for n in (count, limit, reserve)):
            return None
        if count + reserve <= limit:
            return None
        evidence = {'reason': 'context_length_exceeded', 'http_status': http_status,
                    'provider_error': error, 'call_id': call_id, 'request_sha256': digest(payload),
                    'input_tokens': count, 'output_reservation': reserve,
                    'requested_context_tokens': count + reserve, 'server_context_tokens': limit,
                    'endpoint': endpoint, 'server_identity': model,
                    'token_count_source': 'same server /tokenize, no generation',
                    'tp2_retry_eligible': True}
        write(Path(output)/'context_capacity_failure.json', evidence)
        return evidence
    except Exception:
        # Failed diagnosis is NOT evidence of a token overflow.
        return None


def eligible_context_failure(cell):
    path = Path(cell)/'context_capacity_failure.json'
    if not path.exists():
        return None
    try:
        evidence = read(path)
        count, reserve, limit = (evidence[k] for k in
            ('input_tokens', 'output_reservation', 'server_context_tokens'))
        if (evidence.get('reason') == 'context_length_exceeded' and
                evidence.get('http_status') == 400 and evidence.get('tp2_retry_eligible') is True and
                all(isinstance(n, int) and n > 0 for n in (count, reserve, limit)) and
                count + reserve > limit and evidence.get('request_sha256') and evidence.get('call_id')):
            return evidence
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None
