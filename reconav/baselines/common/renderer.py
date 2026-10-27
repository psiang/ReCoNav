from __future__ import annotations
import math
import multiprocessing
import os
from pathlib import Path
from typing import Any
import numpy as np
TRIMESH_TO_HABITAT = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)

class RGBDRenderer:

    def __init__(self, geometry: Path, *, width: int, height: int, hfov: float) -> None:
        if 'semantic' in geometry.name.lower() or not geometry.is_file():
            raise ValueError(f'expected non-semantic HM3D GLB: {geometry}')
        from reconav.baselines.common.rendering import backend
        os.environ.setdefault('PYOPENGL_PLATFORM', 'egl' if backend() == 'nvidia' else 'osmesa')
        import pyrender
        import trimesh
        source = trimesh.load(geometry, force='scene', process=False)
        self.scene = pyrender.Scene(bg_color=np.array([0, 0, 0, 0]), ambient_light=np.ones(3))
        for node in source.graph.nodes_geometry:
            transform, name = source.graph[node]
            self.scene.add(pyrender.Mesh.from_trimesh(source.geometry[name], smooth=False), pose=TRIMESH_TO_HABITAT @ transform)
        vertical_fov = 2 * math.atan(math.tan(math.radians(hfov) / 2) * height / width)
        self.camera = pyrender.PerspectiveCamera(yfov=vertical_fov, aspectRatio=width / height)
        self.camera_node = self.scene.add(self.camera, pose=np.eye(4))
        self.renderer = pyrender.OffscreenRenderer(width, height)
        from OpenGL import GL
        from reconav.baselines.common.rendering import configuration
        self.metadata = {**configuration(), 'vendor': GL.glGetString(GL.GL_VENDOR).decode(),
                         'renderer': GL.glGetString(GL.GL_RENDERER).decode(),
                         'opengl_version': GL.glGetString(GL.GL_VERSION).decode()}

    def render(self, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import pyrender
        self.scene.set_pose(self.camera_node, pose=pose)
        color, depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
        return (np.asarray(color)[..., :3].astype(np.uint8), np.asarray(depth, dtype=np.float32))

def _renderer_worker(connection: Any, geometry: str, width: int, height: int, hfov: float) -> None:
    """Own the OpenGL context in a process isolated from Habitat's GL loader."""
    try:
        renderer = RGBDRenderer(Path(geometry), width=width, height=height, hfov=hfov)
        connection.send({'ready': True, 'rendering': renderer.metadata})
        while True:
            pose = connection.recv()
            if pose is None:
                break
            connection.send(renderer.render(np.asarray(pose, dtype=np.float64)))
    except BaseException as exc:
        connection.send({'error': f'{type(exc).__name__}: {exc}'})
    finally:
        connection.close()

class RGBDRendererProcess:

    def __init__(self, geometry: Path, *, width: int, height: int, hfov: float) -> None:
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_renderer_worker, args=(child, str(geometry), width, height, hfov), daemon=True)
        self.process.start()
        child.close()
        message = self.connection.recv()
        if not message.get('ready'):
            raise RuntimeError(f'RGB-D renderer process failed: {message}')
        self.metadata = message['rendering']

    def render(self, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.connection.send(pose)
        result = self.connection.recv()
        if isinstance(result, dict) and 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def close(self) -> None:
        if self.process.is_alive():
            self.connection.send(None)
            self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        self.connection.close()
