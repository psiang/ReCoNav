"""MapGPT policy adapter for native and ASG scene interfaces."""
from __future__ import annotations
import ast
import inspect
from pathlib import Path
import sys
import textwrap
import types
from reconav.baselines.common.context import global_context

def mapgpt_rollout_with_image_limit(rollout, limit):
    """Set the image limit on upstream model requests."""
    if not isinstance(limit, int) or limit < 1:
        raise ValueError('positive MapGPT image limit required')
    if limit == 20:
        return rollout
    tree = ast.parse(textwrap.dedent(inspect.getsource(rollout)))
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.Compare)
               and ast.dump(n.left) == ast.dump(ast.parse('len(image_list)', mode='eval').body)
               and len(n.ops) == 1 and isinstance(n.ops[0], ast.Gt)
               and len(n.comparators) == 1 and isinstance(n.comparators[0], ast.Constant)
               and n.comparators[0].value == 20]
    if len(matches) != 1:
        raise RuntimeError('upstream MapGPT image guard changed; refusing an ambiguous patch')
    matches[0].comparators[0].value = limit
    namespace = {}
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(rollout), 'exec'),
         rollout.__globals__, namespace)
    return namespace[rollout.__name__]

def mapgpt(world, client, config):
    sys.path.insert(0, str(Path(config['upstream'])))
    api = types.ModuleType('GPT.api')

    def infer(system, prompt, images, *args, **kwargs):
        content = [(f'Image {i}', p) for i, p in enumerate(images) if p is not None]
        response = client.generate(system, prompt + global_context(world), content, True)
        # The official parser silently falls back to A on bad output. Preserve
        # the raw response; do not repair it with evaluator/task information.
        return response, client.calls[-1].get('usage', {}).get('total_tokens', 0)

    api.gpt_infer = infer
    sys.modules['GPT.api'] = api
    from vln.gpt_agent import GPTNavAgent

    class Agent(GPTNavAgent):
        def make_equiv_action(self, actions, obs, traj=None):
            for i, action in enumerate(actions):
                if action != -1:
                    vp = obs[i]['candidate'][action]['viewpointId']
                    world.move(vp)
                    traj[i]['path'].append([vp])

    image_limit = config.get('mapgpt_image_limit', 20)
    Agent.rollout = mapgpt_rollout_with_image_limit(GPTNavAgent.rollout, image_limit)

    args = types.SimpleNamespace(batch_size=1, stop_after=3, max_action_len=config['max_steps'],
                                 max_tokens=config['max_output_tokens'], response_format='json',
                                 llm='gpt-4o-2024-05-13')
    # This selector retains the official JSON policy branch; actual API model
    # identity is explicitly recorded by Client and the run configuration.
    agent = Agent(args, world)
    image_audit = []
    original_prompt = agent.prompt_manager.make_action_prompt
    def audited_prompt(*args, **kwargs):
        result = original_prompt(*args, **kwargs)
        images = agent.prompt_manager.node_imgs[0]
        image_audit.append({'step': len(image_audit), 'slots': len(images),
                            'images': sum(p is not None for p in images)})
        return result
    agent.prompt_manager.make_action_prompt = audited_prompt
    result = agent.rollout()[0]
    result['adapter_image_audit'] = {'limit': image_limit, 'steps': image_audit}
    actions = list(result['a_t'].values())
    if actions and actions[-1] == 0:
        reason = 'native_image_limit' if len(agent.prompt_manager.node_imgs[0]) > image_limit else 'policy_stop'
    else:
        reason = 'max_steps'
    return result, reason
