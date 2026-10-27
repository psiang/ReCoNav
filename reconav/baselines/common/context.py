"""Shared global-ASG context for native language-agent policies."""
import json

def global_context(world):
    if world.condition != 'asg':
        return ''
    # Connectivity is independently consumed by World, not merely serialized.
    return ('\nGLOBAL SCENE ASG (scene facts, not a route or selected target):\n' +
            json.dumps(world.public_asg, separators=(',', ':')) +
            '\nWorld coordinates: x/z horizontal, y vertical. Current position: ' +
            json.dumps(world.graph.nodes[world.current]['position']) + '\nCandidate coordinates: ' +
            json.dumps({vp: world.graph.nodes[world.raw_keys[vp]]['position'] for vp in world.candidates()}))
