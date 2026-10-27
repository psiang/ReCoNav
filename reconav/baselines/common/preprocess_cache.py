"""Reuse model outputs for identical preprocessing requests."""
from pathlib import Path
from reconav.baselines.common.protocol import read, write, ROOT


def reuse_gcvln(source, destination, policy):
    from reconav.baselines.gcvln.preprocess import load_official_prompt, compose_request
    source, destination = Path(source), Path(destination)
    if not (source/'preprocess_metadata.json').is_file():
        return False
    metadata = read(source/'preprocess_metadata.json')
    prompt, fingerprint = load_official_prompt(ROOT/'reconav/baselines/gcvln/appendix_g_prompt.txt')
    request = compose_request(prompt, policy['instruction'])
    call = metadata['call']
    if (metadata['benchmark_task_id'] != policy['benchmark_task_id'] or
        metadata['input_fields'] != ['instruction'] or
        metadata['source'] != 'official_appendix_g_live_api' or
        metadata['official_prompt_sha256'] != fingerprint or
        metadata['requested_model'] != 'qwen3.8-27b' or
        metadata['reasoning_effort'] != 'none' or
        metadata['status'] not in {'valid_first_response','invalid_first_response'} or
        metadata['semantic_retry_count'] != 0 or
        call.get('system') != '' or call.get('prompt') != request or
        call.get('images') != [] or call.get('status') != 'completed'):
        raise RuntimeError('GC-VLN cached first prediction does not match this exact request')
    # Copy only language artifacts, never trajectories, policy maps or scores.
    for name in ('llm_request.txt','llm_raw_response.txt','paper_dag.json','runtime_dag.json'):
        path = source/name
        if path.exists():
            (destination/name).write_bytes(path.read_bytes())
    metadata['cache_source'] = str(source.resolve())
    write(destination/'preprocess_metadata.json', metadata)
    cached_call = dict(call, provider_called=False, cache_source=str(source.resolve()),
                       actual_seconds=0, actual_usage={'input_tokens':0,'output_tokens':0,'total_tokens':0})
    write(destination/'calls.json', [cached_call])
    return True


def reuse_canav(source,destination,policy,upstream):
    from reconav.baselines.canav.generate_reply import official_prompt
    source,destination=Path(source),Path(destination)
    if not (source/'llm_reply_metadata.json').exists():return False
    meta=read(source/'llm_reply_metadata.json')
    prompt,fingerprint=official_prompt(upstream)
    request=prompt+f'"""{policy["instruction"]}"""'
    call=meta.get('call',{})
    if (meta.get('benchmark_task_id')!=policy['benchmark_task_id'] or
        meta.get('input_fields')!=['instruction'] or meta.get('source')!='official_prompt_live_api' or
        meta.get('model')!='qwen3.8-27b' or meta.get('reasoning_effort')!='none' or
        meta.get('prompt_sha256')!=fingerprint or meta.get('semantic_retry_count')!=0 or
        call.get('prompt')!=request or call.get('system')!='' or call.get('images')!=[] or
        call.get('status')!='completed' or meta.get('status') not in {'completed','invalid_first_response'}):
        raise RuntimeError('CA-Nav cache does not match the exact original request')
    for name in ('llm_prompt.txt','llm_request.txt','llm_raw_response.txt','llm_reply.json'):
        if (source/name).exists():(destination/name).write_bytes((source/name).read_bytes())
    write(destination/'llm_reply_metadata.json',dict(meta,cache_source=str(source)))
    write(destination/'calls.json',[dict(call,provider_called=False,cache_source=str(source),actual_seconds=0)])
    return True
