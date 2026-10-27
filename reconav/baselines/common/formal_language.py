"""Formal-baseline language frontends."""
from reconav import paths as release_paths
import ast
import logging
import os
from pathlib import Path
import re
import string
import time
from reconav.baselines.common.client import LocalQwenClient
from reconav.baselines.common.protocol import file_digest, read, write
from reconav.baselines.common.upstream import definitions, module

class PredictionFailure(Exception):
    """A completed first prediction that the original method cannot execute."""

class Frontend:

    def __init__(self, method, policy, output, base_url, cache=None, cache_cell=None):
        self.method, self.policy, self.output = (method, policy, Path(output))
        self.cache = Path(cache) / method / policy['scene_id'] / policy['benchmark_task_id'] if cache else None
        if cache_cell:
            self.cache = Path(cache_cell)
        self.client = LocalQwenClient({'base_url': base_url, 'model': 'qwen3.8-27b', 'max_output_tokens': 1500 if method == 'lang2ltl2' else 2000, 'max_local_calls': 1, 'api_timeout': 900}, output)
        self.sources = {}

    def source(self, path):
        path = Path(path)
        self.sources[str(path)] = file_digest(path)
        return path

    def generate(self, system, prompt, stage):
        if self.cache and (self.cache / 'calls.json').exists():
            calls = read(self.cache / 'calls.json')
            if not calls or calls[-1].get('status') != 'completed':
                return self.client.generate(system, prompt, stage=stage)
            if (self.cache / 'result.json').exists():
                result = read(self.cache / 'result.json')
            else:
                raw = read(self.cache / 'frontend_raw.json')
                result = {'policy_input': read(self.cache / 'policy.json'), 'source_sha256': raw['sources'], 'max_output_tokens': self.client.config['max_output_tokens']}
            if len(calls) != 1 or result['policy_input'] != self.policy or any((file_digest(Path(p)) != h for p, h in result['source_sha256'].items())):
                raise RuntimeError('formal-language cache input/source mismatch')
            call = calls[0]
            if call['system'] != system or call['prompt'] != prompt or call.get('images') != [] or (call['model'] != 'qwen3.8-27b') or (call.get('reasoning_effort') != 'none') or (call.get('status') != 'completed') or (result.get('max_output_tokens', call.get('token_admission', {}).get('output_reservation')) != self.client.config['max_output_tokens']):
                raise RuntimeError('formal-language cache does not match exact first request')
            self.client.calls.append(dict(call, provider_called=False, cache_source=str(self.cache), actual_seconds=0))
            write(self.output / 'calls.json', self.client.calls)
            return call['response']
        return self.client.generate(system, prompt, stage=stage)

    def run(self):
        if self.method == 'lang2ltl2':
            prompt = self.source(release_paths.resource_path('lang2ltl2_srer_prompt.txt'))
            raw = self.generate(prompt.read_text(), 'Extract the referring expressions to predicates map, lifted command, and symbol map for the following command:\n\nCommand:' + self.policy['instruction'], 'original_srer')
            source = self.source(release_paths.upstream_path('Lang2LTL-2/srer.py'))
            ns = definitions(source, ['parse_llm_output'], {'logging': logging, 'PROPS': ['a', 'b', 'c', 'd', 'h', 'j', 'k'], 'eval': ast.literal_eval})
            write(self.output / 'frontend_raw.json', {'response': raw, 'sources': self.sources})
            try:
                parsed = ns['parse_llm_output'](self.policy['instruction'], raw)
                if not parsed['sre_to_preds'] or not isinstance(parsed['lifted_utt'], str):
                    raise ValueError('missing referring expressions')
            except (KeyError, ValueError, TypeError, IndexError, AttributeError, SyntaxError) as error:
                raise PredictionFailure('invalid_original_SRER_output: ' + str(error)) from error
            native = module(self.source(release_paths.upstream_path('Lang2LTL-2/lt_s2s_sup_tcd.py')), 'original_tcd')
            import torch
            torch.set_num_threads(4)
            started = time.monotonic()
            t5 = native.Seq2Seq((release_paths.model_root() / 'lang2ltl2/checkpoint-best'), 't5-base')
            query = parsed['lifted_utt'].translate(str.maketrans('', '', string.punctuation))
            formula = t5.type_constrained_decode([query])[0]
            prefix = 'translate English to Linear Temporal Logic: '
            local_tokens = {'input_tokens': len(t5.tokenizer(prefix + query)['input_ids']), 'output_tokens': len(t5.tokenizer(formula)['input_ids']), 'output_token_definition': 'retokenized formula, not internal autoregressive decode trace', 'seconds_including_load': time.monotonic() - started}
            result = {'raw_response': raw, 'original_srer': parsed, 'lifted_formula': formula, 't5_usage': local_tokens, 'source_sha256': self.sources}
        else:
            source = self.source(release_paths.upstream_path('robotlimp/limp/language/llm4tl.py'))

            def generate(api, system, user, *args, **kwargs):
                return (self.generate(system, user, 'original_single_stage'), [])
            ns = definitions(source, ['query_llm', 'parse_spatial_lifted_ltl'], {'os': os, 're': re, 'string': string, 'generate_response_from_llm': generate})
            raw, _ = ns['query_llm']('openai', self.policy['instruction'], [], strategy='single_stage', enable_prints=False)
            write(self.output / 'frontend_raw.json', {'response': raw, 'sources': self.sources})
            try:
                bindings, formula = ns['parse_spatial_lifted_ltl'](raw)
                if not bindings:
                    raise ValueError('no original spatial predicates')
            except (ValueError, IndexError, KeyError, TypeError, AttributeError, SyntaxError) as error:
                raise PredictionFailure('invalid_original_LIMP_output: ' + str(error)) from error
            result = {'raw_response': raw, 'spatial_predicates': bindings, 'lifted_formula': formula, 'source_sha256': self.sources, 'human_verification': False}
        write(self.output / 'frontend.json', result)
        return result
