"""NavGPT policy adapter for native and ASG scene interfaces."""
from __future__ import annotations
from pathlib import Path
import sys
import types
from reconav.baselines.common.context import global_context

def navgpt(world, client, config):
    sys.path.insert(0, str(Path(config['navgpt_deps'])))
    sys.path.insert(0, str(Path(config['upstream']) / 'nav_src'))
    import agent as native
    from langchain.llms.base import LLM

    class PolicyLLM(LLM):
        @property
        def _llm_type(self):
            return 'local-qwen-transport'

        def _call(self, prompt, stop=None, run_manager=None):
            return client.generate('Follow the navigation agent protocol below.', prompt + global_context(world), stop=stop)

    native.OpenAI = lambda **kwargs: PolicyLLM()
    args = types.SimpleNamespace(llm_model_name='gpt-4', temperature=0.0, use_tool_chain=False,
                                 use_single_action=True, use_history_chain=False, use_navigable=False,
                                 use_relative_angle=True, load_instruction=True, load_action_plan=False,
                                 max_iterations=config['max_steps'])
    agent = native.NavAgent(world, args)
    result = agent.rollout()[0]
    reason = 'max_steps' if result.get('llm_output', '').startswith('Agent stopped due to') else 'policy_stop'
    return result, reason
