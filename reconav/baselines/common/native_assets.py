"""Restore released, version-matched native geometry into a movable cache."""
from pathlib import Path
import shutil

from reconav.paths import habitat_navigation_root
from reconav.baselines.common.protocol import file_digest, file_identity, read, scene_assets, write


def prepare_native_assets(version, cache=None, scene_ids=None):
    from reconav.baselines.common.robot import CACHE, PROFILE
    cache = Path(cache) if cache is not None else CACHE
    source = habitat_navigation_root() / version
    if not (source / 'manifest.json').is_file():
        raise FileNotFoundError(
            'Habitat navigation resources not found at ' + str(source)
            + '. Download habitat-navigation-v1.0.0.tar.gz from the GitHub release and set '
            + 'RECONAV_HABITAT_NAVIGATION_ROOT; see docs/baselines.md#habitat-navigation-resources.'
        )
    manifest = read(source / 'manifest.json')
    if (manifest.get('format') != 'reconav-native-navigation-v1'
            or manifest.get('robot_profile') != PROFILE
            or manifest.get('habitat_version') != version):
        raise ValueError('native navigation package has a different robot profile or version')
    identities = {}
    selected = set(manifest['scenes']) if scene_ids is None else set(scene_ids)
    unknown = selected - set(manifest['scenes'])
    if unknown:
        raise ValueError('Unknown native scenes: ' + str(sorted(unknown)))
    for scene_id, row in manifest['scenes'].items():
        if scene_id not in selected:
            continue
        scene = scene_assets(scene_id)['scene']
        if not scene.is_file():
            raise FileNotFoundError('Native baselines require official HM3D scene files; set RECONAV_HM3D_ROOT: ' + str(scene))
        mesh = source / scene_id / 'robot.navmesh'
        if file_digest(scene) != row['scene_sha256'] or file_digest(mesh) != row['navmesh_sha256']:
            raise ValueError('native navigation geometry does not match the official scene: ' + scene_id)
        identities[scene_id] = (scene, mesh)
    restored = {}
    for scene_id, (scene, mesh) in identities.items():
        target = cache / version / scene_id / 'robot.navmesh'
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or file_digest(target) != manifest['scenes'][scene_id]['navmesh_sha256']:
            shutil.copyfile(mesh, target)
        record = {key: value for key, value in manifest['scenes'][scene_id].items()
                  if key not in {'scene_sha256', 'navmesh_sha256'}}
        record.update(scene_identity=file_identity(scene), navmesh_identity=file_identity(target))
        write(target.with_name('mesh.json'), record)
        restored[scene_id] = record
    starts = read(source / 'exact_starts.json')
    target_starts = cache / version / 'exact_starts.json'
    previous = read(target_starts) if target_starts.exists() else {}
    rows = {row['task_id']: row for row in previous.get('rows', [])} if previous.get('robot_profile') == PROFILE and previous.get('habitat_version') == version else {}
    for row in starts['rows']:
        if row['scene_id'] not in restored:
            continue
        identity = restored[row['scene_id']]
        row.update(scene_identity=identity['scene_identity'], navmesh_identity=identity['navmesh_identity'])
        rows[row['task_id']] = row
    starts['rows'] = sorted(rows.values(), key=lambda row: row['task_id'])
    for key, value in (('total', len(rows)), ('supported', sum(row['supported'] for row in rows.values()))):
        if key in starts:
            starts[key] = value
    write(target_starts, starts)
    return True
