"""Execute Lang2LTL-2 formulas on the physical navigation graph.

Terminal poses use infinite stuttering for formula evaluation."""
import heapq
import math
from functools import lru_cache
import numpy as np
from reconav.baselines.common.formal_language import PredictionFailure


class Automaton:
    def __init__(self,formula):
        import spot
        import buddy
        self.spot,self.buddy=spot,buddy
        try:
            self.aut=spot.translate(spot.formula(formula),'deterministic','complete','parity')
        except (SyntaxError,RuntimeError) as error:
            raise PredictionFailure('invalid_original_T5_formula: '+str(error)) from error
        self.ap=[str(p) for p in self.aut.ap()]
        self.variables=[self.aut.register_ap(p) for p in self.aut.ap()]
        self.initial=self.aut.get_init_state_number()

    @lru_cache(maxsize=None)
    def edge(self,state,mask):
        valuation=self.buddy.bddtrue
        for i,var in enumerate(self.variables):
            valuation &= self.buddy.bdd_ithvar(var) if mask&(1<<i) else self.buddy.bdd_nithvar(var)
        matches=[e for e in self.aut.out(state) if e.cond & valuation != self.buddy.bddfalse]
        if len(matches)!=1:
            raise RuntimeError('expected complete deterministic Spot automaton')
        e=matches[0]
        return e.dst,tuple(e.acc.sets())

    @lru_cache(maxsize=None)
    def accepts_stay(self,state,mask):
        seen,marks={},[]
        while state not in seen:
            seen[state]=len(marks)
            state,acc=self.edge(state,mask)
            marks.append(acc)
        cycle=set(x for a in marks[seen[state]:] for x in a)
        return self.aut.acc().accepting(self.spot.mark_t(sorted(cycle)))


def event_masks(world,automaton,bindings,records):
    nodes=list(world.graph)
    positions=np.asarray([world.graph.nodes[n]['position'] for n in nodes])
    masks=np.zeros(len(nodes),dtype=np.int64)
    audit={}
    for index,ap in enumerate(automaton.ap):
        if ap not in bindings:
            raise PredictionFailure('formula_atom_missing_original_grounding: '+ap)
        target=bindings[ap]
        if isinstance(target,dict):
            # Original anchor-relative 2D coordinates have no emitted floor;
            # preserve that limitation, rather than invent a floor constraint.
            point=np.asarray([target['x'],target['y']])
            active=np.linalg.norm(positions[:,[0,2]]-point,axis=1)<=.5
        else:
            record=records[target]
            identifier=int(target.rsplit('_',1)[1])
            if record['kind']=='region':
                active=np.asarray([world.graph.nodes[n].get('region')==identifier for n in nodes])
                from reconav.baselines.common.protocol import read
                start=read(world.cache/'starts'/(world.policy['benchmark_task_id']+'.json'))
                active[nodes.index('START')]=start['region']==identifier
            else:
                # ASG's scene-only physical interaction anchors. No desired
                # task target, contract or evaluator predicate enters here.
                anchors=[d['position'] for _,d in world.graph.nodes(data=True)
                    if d.get('kind')==record['kind']+'_anchor' and d.get('entity_id')==identifier]
                if not anchors:
                    active=np.zeros(len(nodes),dtype=bool)
                else:
                    from scipy.spatial import cKDTree
                    active=cKDTree(anchors).query(positions)[0]<=.10
        masks[active]|=1<<index
        audit[ap]={'grounding':target,'satisfying_nodes':int(active.sum())}
    return dict(zip(nodes,map(int,masks))),audit


def plan(world,formula,bindings,records):
    automaton=Automaton(formula)
    masks,audit=event_masks(world,automaton,bindings,records)
    initial=(world.current,automaton.edge(automaton.initial,masks[world.current])[0])
    queue=[(0.,initial)]; costs={initial:0.}; previous={}
    final=None
    while queue:
        cost,state=heapq.heappop(queue)
        if cost!=costs[state]:
            continue
        node,q=state
        if automaton.accepts_stay(q,masks[node]):
            final=state
            break
        for nxt in world.graph[node]:
            next_q=automaton.edge(q,masks[nxt])[0]
            successor=(nxt,next_q)
            candidate=cost+world.graph.edges[node,nxt]['weight']
            if candidate<costs.get(successor,math.inf):
                costs[successor]=candidate;previous[successor]=state
                heapq.heappush(queue,(candidate,successor))
    path=['START']
    if final is not None:
        path=[final[0]]
        while final in previous:
            final=previous[final];path.append(final[0])
        path.reverse()
    return path,{'termination':'ltl_plan_complete' if len(path)>1 or automaton.accepts_stay(initial[1],masks['START']) else 'ltl_no_feasible_plan',
        'automaton_states':automaton.aut.num_states(),'product_states_explored':len(costs),'atomic_events':audit,
        'executor':'shared G0 shortest accepting physical walk, final-pose infinite stutter',
        'event_semantics':'region membership; object/portal interaction-anchor arrival within 0.10m; no portal crossing/count repair',
        'gold_task_fields':[]}
