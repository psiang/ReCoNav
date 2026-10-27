"""ASG landmarks and physical graph interfaces for formal baselines."""
import math
import numpy as np
from reconav.baselines.common.protocol import read


def landmarks(world, asg_dir):
    public = world.public_asg
    geometry = read(asg_dir/'geometry.json')
    regions = {r['id']:r for r in public['regions']}
    obj_geometry = {o['index']:o for o in geometry['objects']}
    portal_geometry = {o['index']:o for o in geometry['portals']}
    records, unlocated = {}, []
    for kind, rows in [('object',public['objects']),('region',public['regions']),('portal',public['portals'])]:
        for record in rows:
            if record['id'] < 0:
                continue
            key = f"{kind}_{record['id']}"
            center = record.get('center_xz')
            region = regions.get(record.get('region',record.get('region_a',record['id'])),{})
            floor = region.get('floor_height')
            if floor is None:
                same_level=[r['floor_height'] for r in regions.values() if r['level']==record.get('level')]
                if same_level:floor=float(np.median(same_level))
            if kind=='portal':
                geom = portal_geometry.get(record['id'],{})
                center = geom.get('center')
            else:
                geom = obj_geometry.get(record['id'],{}) if kind=='object' else record
            if center is None or floor is None:
                unlocated.append({'id':key,'label':record['label']})
                continue
            # Exact object center height for original OSG above/below predicates.
            low, high = geom.get('min_height'),geom.get('max_height')
            height = (low+high)/2 if low is not None and high is not None else floor
            x,z = map(float,center)
            records[key] = {'id':key,'kind':kind,'label':record['label'],
                'position':[x,float(height),z],'floor_height':float(floor),
                'level':record.get('level',record.get('level_a')),
                'description':{'category':record['label'],'kind':kind,
                    'room':region.get('label','unassigned'),
                    'floor':record.get('level',record.get('level_a'))}}
    return records,unlocated


def write_trajectory(world, output, method, termination, path):
    from reconav.baselines.common.protocol import write
    if not path or path[0]!='START':
        raise RuntimeError('physical execution must start at the exact benchmark pose')
    positions = [world.graph.nodes[path[0]]['position']]
    moves = []
    for a,b in zip(path,path[1:]):
        if a==b:
            continue
        if not world.graph.has_edge(a,b):
            raise RuntimeError('planned trajectory includes a non-physical G0 edge')
        positions.append(world.graph.nodes[b]['position'])
        moves.append({'from':a,'to':b,'distance_m':world.graph.edges[a,b]['weight']})
    trajectory = {'method':method+'+ASG','task_id':world.policy['benchmark_task_id'],
        'route':{'waypoints':[dict(zip(('x','y','z'),map(float,p))) for p in positions]},
        'executed_actions':len(moves),'recorded_poses':len(positions),
        'path_length_m':sum(math.dist(a,b) for a,b in zip(positions,positions[1:])),
        'termination':termination,'valid_for_scoring':True,'physical_moves':moves,
        'policy_information':['instruction','exact_start_pose','scene_only_ASG','physical_G0'],
        'ground_truth_bindings_supplied':False,
        'physical_backend':getattr(world, 'physical_backend', None)}
    write(output/'trajectory.json',trajectory)
    return trajectory
