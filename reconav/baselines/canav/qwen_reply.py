"""Original CA-Nav instruction prompt, local Qwen only, one recorded attempt."""
import argparse
from pathlib import Path
from reconav.baselines.canav.export_task import CA_NAV_EPISODE_ID, load_policy_input
from reconav.baselines.canav.generate_reply import official_prompt, parse_json_reply, write_reply
from reconav.baselines.common.client import LocalQwenClient
from reconav.baselines.common.protocol import write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:8009/v1')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    policy = load_policy_input(args.task)
    prompt, sha = official_prompt(args.upstream)
    request = prompt + f'"""{policy.instruction}"""'
    (args.output/'llm_prompt.txt').write_text(prompt)
    (args.output/'llm_request.txt').write_text(request)
    client = LocalQwenClient({'base_url': args.base_url, 'model': 'qwen3.8-27b',
        'max_output_tokens': 4096, 'max_local_calls': 1, 'api_timeout': 900}, args.output)
    raw = client.generate('', request, stage='original_canav_instruction')
    (args.output/'llm_raw_response.txt').write_text(raw)
    metadata = {'benchmark_task_id': policy.benchmark_task_id, 'ca_nav_episode_id': CA_NAV_EPISODE_ID,
        'input_fields': ['instruction'], 'source': 'official_prompt_live_api', 'variant': 'qwen-local',
        'model': 'qwen3.8-27b', 'requested_model': 'qwen3.8-27b', 'reasoning_effort': 'none',
        'prompt_sha256': sha, 'temperature': 0, 'call': client.calls[-1], 'semantic_retry_count': 0}
    try:
        reply = parse_json_reply(raw)
    except (ValueError, TypeError, KeyError) as error:
        metadata.update(status='invalid_first_response', error=str(error))
        write(args.output/'llm_reply_metadata.json', metadata)
        return 2
    write_reply(reply, args.output/'llm_reply.json')
    metadata.update(status='completed', parsed_result=reply)
    write(args.output/'llm_reply_metadata.json', metadata)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
