"""Prompts for direct prediction, feedback and candidate selection."""
from __future__ import annotations

import json
from reconav.method.semantic.compiler import load_system_prompt

_BINDING_RULES = """
Bind EVERY nominal occurrence in the semantic contract exactly once, including
nominals inside qualifier references and explicit coreference antecedents.
Each binding contains a JSONPath RELATIVE TO the semantic_contract root, ASG kind
(object/region/portal/level), an ASG category label, and actual instance ID list.
The reserved label "any" is valid for every kind and denotes all instances of
that kind. Areas/spaces use region/any; rooms use region/room, which excludes
hallway, staircase, garage, balcony and porch regions. Ordinary portals use
portal/portal (door/opening/untyped connections, excluding stairs). Building
floors/storeys use level/any. These collective labels are valid ASG labels.
Object/floor denotes physical floor surfaces, not building storeys. Preserve
specific categories and named levels when the instruction supplies them.
Preserve qualifiers on references, e.g. the bedroom on the lower floor. A portal
connected to a room uses an IN qualifier with that room as its reference.
Use paths such as $.route.goal.target and $.constraints.avoid[0].target; never
prefix a binding path with $.semantic_contract. A goal or VIA is a committed
identity: choose one ID according to the WHOLE instruction and scene, including
ordering, avoidance, counts and stated preferences. Do not choose randomly, by
lowest ID, or by nearest distance unless the language supplies that criterion.
For an existential (quantifier="exists") reference inside the qualifiers of a
committed GOAL/VIA, return one or more legal witness IDs; enumerating every
matching reference is not required. This also applies to nested references.
The chosen GOAL/VIA must actually satisfy its qualifiers with those exact
witnesses. Do not change an existential reference to "unique" to choose one ID.
AVOID/COUNT, including their preferred forms, denote SETS: return every matching
target ID and complete sets for existential references within those constraints;
one convenient witness must not narrow a constraint's target set.
A singular (quantifier="unique") reference denotes one object, region, or level
as permitted by the relation. Explicit coreferences must keep the same identity.
START is not a nominal occurrence and must never receive a binding.
The ASG view supplies the public directional label-denotation rules as well as
all scene entities; use these rules rather than treating generic labels as exact
string identity. They do not tell you which candidate is the intended target.
Use only supplied scene IDs. If you cannot produce a complete justified binding,
return status="unresolved" and explain; do not fabricate an ID or drop a clause.
The semantic contract remains scene-independent; put IDs ONLY in bindings.
START is the original task position, never a later VIA's position. NEAR uses 2m
within the reference's room. NEAREST/FARTHEST compare candidates across rooms
on the reference's level (or the original START level for a START reference).
These are implicit relation domains; do not add unstated IN qualifiers.
Scope boundaries are named events: open intervals exclude exact boundary entries,
not all earlier encounters with the eventual endpoint object. COUNT counts
DISTINCT entity IDs. AVOID and COUNT<=0 use the same occurrence convention.
One model response only. No solver-based retry, hidden gold or automatic binding
correction will be provided. The solver executes the contract you commit.
"""


def direct_prompt():
    return ("Interpret the instruction AND bind it to the supplied ASG in one structured response. "
            "The schema has status, reason, semantic_contract and bindings. "
            "Apply the semantic specification below inside semantic_contract, but ALSO perform "
            "the scene-binding task in the outer bindings list.\n" + _BINDING_RULES + "\n" + load_system_prompt())


def direct_feedback_prompt():
    return (
        "Revise a complete Direct Contract attempt after mechanical validation or "
        "solver feedback. Return the full status/reason/semantic_contract/bindings "
        "artifact again; do not return a patch. The original instruction is the "
        "semantic authority. Feedback may identify schema/type/denotation/lowering/"
        "solver failures, but it is not permission to delete a hard requirement or "
        "invent scene IDs. Use only the supplied ASG.\n"
        + _BINDING_RULES.replace(
            "One model response only. No solver-based retry, hidden gold or automatic binding\n"
            "correction will be provided. The solver executes the contract you commit.",
            "This response is one bounded feedback revision. No hidden gold or automatic\n"
            "binding correction is provided. The solver executes the revised contract you commit.",
        )
        + "\n"
        + load_system_prompt()
    )


def direct_id_sequence_prompt():
    return (
        "Plan the instruction directly as a concise ordered sequence of typed ASG "
        "entity IDs. Return JSON with exactly two fields: status=\"resolved\" and "
        "a non-empty targets list with at most 32 entries. Do not return a reason, "
        "explanation, analysis, or any field other than status and targets. "
        "Return one minified JSON line with no spaces or newlines, for example "
        "{\"status\":\"resolved\",\"targets\":[{\"kind\":\"object\",\"id\":7}]}. "
        "Targets may be objects, regions, "
        "portals, or levels. Do not repeat an ID unless the instruction genuinely "
        "requires revisiting it. "
        "Include the task-relevant route anchors and only those intermediate scene "
        "entities that change the intended route. Omit mechanically implied rooms "
        "and portals because the shared physical solver supplies connective paths. "
        "The final target must be an object, region, or level, never a portal.\n\n"
        "Use the same fixed task semantics as the structured methods. GOAL is the "
        "final destination; explicitly required intermediate visits and orders are "
        "hard requirements. AVOID forbids encounters with its target within its "
        "stated scope. COUNT constrains the number of DISTINCT target entity IDs "
        "encountered within its scope. Preferences are soft requirements and should "
        "be satisfied when possible without violating hard requirements. Spatial "
        "qualifiers such as IN, NEAR, NEAREST_TO, FARTHEST_FROM, and BETWEEN refine "
        "which scene entity is meant. NEAR uses the benchmark's 2m relation within "
        "the reference's room; "
        "NEAREST_TO and FARTHEST_FROM compare candidates on the reference level; "
        "BETWEEN for a portal means connecting the two referenced regions. Constraint "
        "scopes are open event intervals: exact AFTER/UNTIL boundary occurrences are "
        "excluded. START is the original task start.\n\n"
        "Do not emit a semantic contract or symbolic constraints. Do not try to "
        "prove the complete geometric connectivity of the route and do not enumerate "
        "every physical transition. The downstream WAYPOINT executor is responsible "
        "for geometrically connecting each consecutive pair of IDs that you choose. "
        "It will NOT repair AVOID, COUNT, ordering, preference, or identity mistakes. "
        "Use only IDs present in the supplied ASG and commit to your best complete "
        "typed-ID sequence."
    )


def candidate_identity_selection_prompt():
    return (
        "Choose exactly one scene ID for every ambiguous identity path listed in "
        "candidate_domains. The semantic contract, label grounding, and candidate "
        "domains are fixed; do not edit them. Use the original instruction and "
        "supplied ASG evidence to make the best-supported identity choice. The "
        "payload may additionally contain candidate_evidence from prior solver "
        "queries. When present, treat proved HARD_UNSAT as binding negative evidence, "
        "treat SAT as a witnessed feasible assignment, and treat UNKNOWN/incomplete "
        "evidence as unresolved rather than false. A proved preference cost may be "
        "used only when penalty_proven=true. Solver evidence is computed under the "
        "current semantic contract; the original instruction remains the semantic "
        "authority when judging which identity the user intended. Do not invent IDs, "
        "change semantics, or add/drop candidate paths. Return status, reason, and "
        "one path/grounded_id choice per ambiguous path. Every ID must belong to its "
        "provided candidate domain. If you cannot choose, return status=\"unresolved\"."
    )


def forced_early_binding_prompt():
    """Select an identity before route evidence is available."""
    return candidate_identity_selection_prompt()


def final_fallback_binding_prompt():
    """Select an execution identity using the accumulated candidate evidence."""
    return candidate_identity_selection_prompt()


def prompt_payload(request, scene_view, *, contract=None, occurrences=None):
    value = {"instruction": request.instruction, "original_start": request.robot_position,
             "start_heading": request.heading, "asg": scene_view}
    if contract is not None:
        value["semantic_contract"] = contract.model_dump(mode="json")
        value["nominal_occurrences"] = occurrences
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
