"""Expose ASG scene data through GC-VLN's SceneGraph/BEV interface.

The 2D planner receives the current-floor map and entity records from every
floor. The waypoint follower executes routes on the physical navigation graph.
"""
import math
from pathlib import Path
from types import SimpleNamespace
import cv2
import networkx as nx
import numpy as np
import torch
from reconav.baselines.common.protocol import read, write
from reconav.baselines.gcvln.g0_env import world_to_gc, gc_to_world, active_map_floor


class ASGSceneGraph:
    def __init__(self, config, policy, asset_dir, asg_dir, output, pose_provider):
        self.config, self.policy, self.pose_provider = config, policy, pose_provider
        self.size, self.resolution = config.MAP.SIZE, config.MAP.RESOLUTION
        self.device = config.UNI.DEVICE
        self.output = Path(output)
        self.source = read(Path(asset_dir)/policy['scene_id']/'asg_graph.json')
        self.physical = read(Path(asset_dir)/policy['scene_id']/'physical_free_space.json')
        if self.source['graph_id'] != self.physical['graph_id']:
            raise RuntimeError('ASG geometry/graph mismatch')
        self.public = self.source['public_asg']
        self.geometry = read(Path(asg_dir)/'geometry.json')
        self.regions = {r['id']:r for r in self.public['regions']}
        self.segment2d_results = []
        self.layer_cache = {}
        self.connector_layers = {}
        self.map_draft = SimpleNamespace(scene_graph=nx.Graph())
        self.reset()

    def pixels(self, points):
        return np.rint(world_to_gc(points, self.policy)*self.resolution+self.size//2).astype(np.int32)

    def reset(self, dag=None, objects_sum=None):
        self.dag = dag
        self.map_draft.scene_graph = nx.Graph()
        objects = [dict(o, kind='object') for o in self.public['objects'] if o['id'] >= 0]
        objects += [dict(r, kind='region') for r in self.public['regions'] if r['id'] >= 0]
        portal_geoms = {g['index']:g for g in self.geometry['portals']}
        for portal in self.public['portals']:
            geom = portal_geoms.get(portal['id'])
            if geom:
                x,z = geom['center']
                objects.append(dict(portal, kind='portal', center_xz=[x,z],
                    bounds_xz=[x-.1,z-.1,x+.1,z+.1], level=portal['level_a']))
        missing_geometry = []
        for obj in objects:
            if obj.get('geometry_missing') or 'bounds_xz' not in obj or 'center_xz' not in obj:
                missing_geometry.append({'kind':obj['kind'],'id':obj['id'],'label':obj['label']})
                continue  # Keep the semantic record, but never invent its spatial location.
            x1,z1,x2,z2 = obj['bounds_xz']
            pts = self.pixels([[x1,z1],[x1,z2],[x2,z1],[x2,z2]])
            self.map_draft.scene_graph.add_node(self.map_draft.scene_graph.number_of_nodes(),
                asg_entity_key=f"{obj['kind']}:{obj['id']}",
                caption=obj['label'], caption_conf=1.,
                num_detections=self.config.RS.NUM_DETECTIONS_THRESHOLD,
                center=self.pixels(obj['center_xz']), scope=np.stack([pts.max(0),pts.min(0)]),
                level=obj.get('level'), source='provided_scene_ASG',
                detection_count_semantics='authoritative map entry admitted at original confidence gate')
        write(self.output/'asg_bridge_manifest.json', {'condition':'asg',
            'graph_id':self.source['graph_id'], 'source_identity':self.source['source_identity'],
            'entities':self.map_draft.scene_graph.number_of_nodes(), 'all_levels_retained':True,
            'unlocated_entities_not_inserted_in_geometric_graph':missing_geometry,
            'native_map_size':self.size, 'cells_per_meter':self.resolution,
            'coordinate_transform':'inverse start yaw; GC x=-local z, GC y=-local x',
            'geometry_interface':'current-floor ASG BEV -> original waypoint selection -> shared G0 executor',
            'unchanged':['instruction graph','Region_Solver','navigation tree','STOP decision'],
            'changed':['scene graph provided by ASG','FMM/Habitat motion replaced by G0 physical paths'],
            'cross_floor_policy_limit':'Current-floor BEV geometry with all-floor entity records',
            'gold_task_fields':[], 'online_detection_replaced_by_provided_ASG':True})

    def raster(self, height):
        from shapely.geometry import shape
        from shapely import contains_xy
        floor = active_map_floor(height,(v['floor_height'] for v in self.physical['regions'].values()))
        if floor in self.layer_cache:
            return self.layer_cache[floor]
        # Fill each polygon separately so a hole does not erase another region.
        mask = np.zeros((self.size,self.size), dtype=np.uint8)
        connectors = np.zeros_like(mask)
        active = set()
        for rid, record in self.physical['regions'].items():
            if abs(record['floor_height']-floor) > .75:
                continue
            active.add(int(rid))
            geom = record['free_space']
            polygons = [geom['coordinates']] if geom['type']=='Polygon' else geom.get('coordinates', [])
            for rings in polygons:
                if not rings:
                    continue
                part = np.zeros_like(mask)
                cv2.fillPoly(part, [self.pixels(rings[0])], 1)
                for hole in rings[1:]:
                    cv2.fillPoly(part, [self.pixels(hole)], 0)
                # Raster drawing rounds polygon boundaries. Admit only actual
                # G0-free cell centers, so a displayed free waypoint is not
                # rejected by the exact executor due to rounding overshoot.
                rows, cols = np.nonzero(part)
                world = gc_to_world((np.stack([cols,rows],axis=-1)-self.size//2)/self.resolution,self.policy)
                part[rows,cols] = contains_xy(shape(geom).buffer(1e-8),world[:,0],world[:,1])
                mask |= part
        # Actual G0 cross-region edges, not all pairs of adjacent room centers.
        nodes = self.source['nodes']
        for a, edges in self.source['adjacency'].items():
            for b, _ in edges:
                if ((nodes[a]['region'] not in active and nodes[b]['region'] not in active)
                        or nodes[a]['region']==nodes[b]['region'] or b<a):
                    continue
                pa, pb = nodes[a]['position'], nodes[b]['position']
                p = self.pixels([[pa[0],pa[2]],[pb[0],pb[2]]])
                cv2.line(connectors, tuple(p[0]), tuple(p[1]), 1, 1)
                # Cover the actual continuous segment's rounded cells, not
                # only the line joining two independently rounded endpoints.
                samples = np.linspace([pa[0],pa[2]],[pb[0],pb[2]],
                    max(2,math.ceil(math.hypot(pa[0]-pb[0],pa[2]-pb[2])*self.resolution*4)+1))
                pixels = self.pixels(samples)
                pixels = pixels[((pixels>=0)&(pixels<self.size)).all(axis=1)]
                connectors[pixels[:,1],pixels[:,0]] = 1
        mask |= connectors
        self.connector_layers[floor] = connectors
        self.layer_cache[floor] = mask
        return mask

    def is_connector_cell(self, point, height):
        floor = active_map_floor(height,(v['floor_height'] for v in self.physical['regions'].values()))
        self.raster(height)
        row,col = np.rint(point).astype(int)
        return bool(0 <= row < self.size and 0 <= col < self.size and self.connector_layers[floor][row,col])

    def get_scenegraph(self, stage, image_rgb, depth_array, cam_K, pose, agent_coordinate, thin_type):
        from src.scenegraph.utils.map_utils import thining, thining_v2
        from src.solver.math_utils import extract_valid_mask_from_skeleton_bev
        height = float(self.pose_provider()[0][1])
        bev = torch.as_tensor(self.raster(height), dtype=torch.int, device=self.device)
        pixel = np.rint(np.asarray(agent_coordinate)[::-1]*self.resolution+self.size//2).astype(int).tolist()
        if not all(0 <= x < self.size for x in pixel):
            raise RuntimeError('position outside native fixed-size GC-VLN map')
        connected = extract_valid_mask_from_skeleton_bev(pixel, bev)
        zeros = torch.zeros_like(bev)
        thin = (thining(zeros, bev, zeros, connected, pixel) if thin_type==1 else thining_v2(bev, connected))
        self.map_draft.bev_map = bev
        self.map_draft.bev_map_fmm = bev
        self.map_draft.bev_wall = bev
        self.map_draft.bev_thin = thin
        return False  # No fake image detections for the visualization branch.
