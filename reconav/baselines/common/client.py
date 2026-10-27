"""Qwen service requests and input-token budgeting."""
from __future__ import annotations
import base64
import json
from pathlib import Path
import time
import uuid
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.parse import urlparse
from reconav.baselines.common.protocol import file_digest, write


class BudgetExceeded(RuntimeError):
    pass


def local_model_identity(base_url,model='qwen3.8-27b'):
    parsed=urlparse(base_url)
    if parsed.scheme!='http' or parsed.hostname not in {'127.0.0.1','localhost'} or parsed.username:
        raise ValueError('Use a localhost Qwen endpoint; forward remote ports with SSH')
    with urlopen(base_url.rstrip('/')+'/models',timeout=10) as response:
        models=json.load(response)['data']
    row=next((r for r in models if r['id']==model),None)
    if row is None:raise RuntimeError('configured local Qwen model is unavailable')
    identity={k:row.get(k) for k in ('id','root','max_model_len')}
    config=Path(row.get('root') or '')/'config.json'
    if config.is_file():identity['weights_config_sha256']=file_digest(config)
    return identity


def stop_text(text, stop):
    """Preserve the original ReAct observation boundary on every transport."""
    positions = [text.find(s) for s in (stop or []) if s and s in text]
    return text[:min(positions)] if positions else text


class LocalQwenClient:
    """Client for the local Qwen service."""

    def __init__(self, config, output):
        self.config, self.output = config, Path(output)
        self.calls = []
        endpoint = config['base_url'].rstrip('/')
        parsed = urlparse(endpoint)
        if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost'} or parsed.username:
            raise ValueError('local backend requires an unauthenticated loopback HTTP endpoint')
        self.endpoint = endpoint

    def admit_tokens(self, payload):
        """Count the full multimodal request locally before spending inference."""
        limit = self.config.get('max_episode_tokens', 0)
        if not limit:
            return None
        if any(c.get('provider_called') and 'usage' not in c for c in self.calls):
            raise BudgetExceeded('previous local call has unknown token usage; refusing more calls')
        body = {k: payload[k] for k in ('model', 'messages', 'chat_template_kwargs')}
        body['add_generation_prompt'] = True
        endpoint = self.endpoint.removesuffix('/v1') + '/tokenize'
        request = Request(endpoint, data=json.dumps(body).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=60) as response:
            counted = json.load(response)
        count = counted.get('count')
        if not isinstance(count, int) or count < 1:
            raise BudgetExceeded('local tokenizer did not return a positive count')
        spent = sum(c.get('usage', {}).get('total_tokens', 0) for c in self.calls)
        admission = {'input_tokens': count, 'output_reservation': payload['max_tokens'],
                     'used_tokens': spent, 'limit': limit,
                     'admitted': spent + count + payload['max_tokens'] <= limit,
                     'source': 'local multimodal /tokenize; no model generation'}
        write(self.output/'last_token_admission.json', admission)
        if not admission['admitted']:
            raise BudgetExceeded(f'episode token budget: {spent}+{count}+{payload["max_tokens"]}>{limit}')
        return admission

    def generate(self, system, prompt, images=(), json_mode=False, stage='decision', stop=None):
        if len(self.calls) >= self.config.get('max_local_calls', 80):
            raise RuntimeError('local transport safety limit reached; no silent truncation')
        content = [{'type': 'text', 'text': prompt}]
        for label, path in images:
            content.extend([{'type': 'text', 'text': label}, {'type': 'image_url', 'image_url': {
                'url': 'data:image/jpeg;base64,' + base64.b64encode(Path(path).read_bytes()).decode()}}])
        payload = {'model': self.config['model'], 'messages': [
            {'role': 'system', 'content': system}, {'role': 'user', 'content': content}],
            'max_tokens': self.config['max_output_tokens'], 'temperature': 0,
            'chat_template_kwargs': {'enable_thinking': False}}
        if stop:
            payload['stop'] = list(stop)
        if json_mode:
            payload['response_format'] = {'type': 'json_object'}
            if stage == 'decision' and self.config.get('method') == 'mapgpt':
                # Exactly the three fields requested by upstream
                # make_r2r_json_prompts. Syntax only: no Action enum, target,
                # route or constraint solution is inserted by the adapter.
                schema = {'type': 'object', 'additionalProperties': False,
                          'properties': {k: {'type': 'string'} for k in
                                         ('Thought', 'New Planning', 'Action')},
                          'required': ['Thought', 'New Planning', 'Action']}
                payload['response_format'] = {'type': 'json_schema', 'json_schema': {
                    'name': 'mapgpt_native_response', 'strict': True, 'schema': schema}}
            if stage == 'perception':
                # This is our scene-observation serialization, not navigation
                # reasoning. Constrain only the documented frontend schema.
                number = {'type': 'number', 'minimum': 0, 'maximum': 1}
                obj = {'type': 'object', 'additionalProperties': False,
                       'properties': {'label': {'type': 'string'}, 'u': number, 'v': number},
                       'required': ['label', 'u', 'v']}
                view = {'type': 'object', 'additionalProperties': False,
                        'properties': {'description': {'type': 'string'},
                                       'objects': {'type': 'array', 'items': obj}},
                        'required': ['description', 'objects']}
                schema = {'type': 'object', 'additionalProperties': False, 'required': ['views'],
                          'properties': {'views': {'type': 'array', 'minItems': 8, 'maxItems': 8, 'items': view}}}
                payload['response_format'] = {'type': 'json_schema', 'json_schema': {
                    'name': 'navigation_observations', 'strict': True, 'schema': schema}}
        admission = self.admit_tokens(payload)
        record = {'id': uuid.uuid4().hex, 'stage': stage, 'system': system, 'prompt': prompt,
                  'images': [{'label': label, 'path': str(path)} for label, path in images],
                  'model': self.config['model'], 'reasoning_effort': 'none', 'provider': 'local-qwen',
                  'provider_called': True, 'transport_attempts': 1, 'charged_usd_upper': 0,
                  'endpoint': self.endpoint, 'server_identity': self.config.get('server_identity')}
        if admission:
            record['token_admission'] = admission
        if json_mode:
            record['response_format'] = payload['response_format']
        if stop:
            record['stop'] = list(stop)
        self.calls.append(record)
        write(self.output/'calls.json', self.calls)
        started = time.monotonic()
        try:
            request = Request(self.endpoint + '/chat/completions', data=json.dumps(payload).encode(),
                              headers={'Content-Type': 'application/json'})
            with urlopen(request, timeout=self.config.get('api_timeout', 300)) as response:
                data = json.load(response)
            usage = data['usage']
            record['usage'] = {'input_tokens': usage['prompt_tokens'], 'output_tokens': usage['completion_tokens'],
                               'total_tokens': usage['total_tokens'],
                               'output_tokens_details': usage.get('completion_tokens_details') or {}}
            choice = data['choices'][0]
            record.update(response=choice['message'].get('content'), response_model=data['model'],
                          response_id=data['id'], finish_reason=choice['finish_reason'])
            if choice['finish_reason'] != 'stop' or not isinstance(record['response'], str):
                raise RuntimeError('local model output incomplete: ' + str(choice['finish_reason']))
            record['raw_response'] = record['response']
            record['response'] = stop_text(record['response'], stop)
            record['status'] = 'completed'
            return record['response']
        except HTTPError as error:
            # Local error bodies contain no credentials and help distinguish image/context limits.
            record['error'] = f'HTTP {error.code}: ' + error.read().decode(errors='replace')[:1500]
            from reconav.baselines.common.context_retry import confirm_context_rejection
            evidence = confirm_context_rejection(self.output, self.endpoint, payload,
                error.code, record['error'], record['id'])
            if evidence:
                record['context_capacity_failure'] = evidence
            raise RuntimeError(record['error']) from None
        except Exception as error:
            record['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            record['seconds'] = time.monotonic() - started
            write(self.output/'calls.json', self.calls)


    def captions(self, images):
        prompt = ('Describe these eight views of the same indoor position, in image order. '
                  'Report visible objects, room appearance, doors and passages; do not infer a goal or route. '
                  'Return JSON {"views": [{"description": "concise description", "objects": '
                  '[{"label": "object category", "u": 0.5, "v": 0.5}]}]}. Include exactly eight views. '
                  'u,v are the normalized image coordinates of a visible point near each object center, '
                  'between 0 and 1; omit objects that cannot be localized in the image. '
                  'Each image is a different clockwise direction. Do not estimate distances; measured depth is used separately.')
        text = self.generate('You describe visual observations for a navigation agent.', prompt,
                             list(zip([f'Direction {i}' for i in range(8)], images)), True, 'perception')
        rows = json.loads(text)['views']
        if not isinstance(rows, list) or len(rows) != 8 or not all(
                isinstance(x, dict) and isinstance(x.get('description'), str) and isinstance(x.get('objects'), list)
                for x in rows):
            raise ValueError('invalid perception output; no synthetic observations substituted')
        return rows
