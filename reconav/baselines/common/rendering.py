"""Select software rendering or NVIDIA EGL for external workers."""
import os


def backend(environment=None):
    value = (os.environ if environment is None else environment).get('RECONAV_RENDER_BACKEND', 'mesa')
    if value not in {'mesa', 'nvidia'}:
        raise ValueError('RECONAV_RENDER_BACKEND must be mesa or nvidia')
    return value


def worker_environment(environment):
    env = dict(environment)
    env['RECONAV_RENDER_BACKEND'] = backend(env)
    if backend(env) == 'mesa':
        env.update(LIBGL_ALWAYS_SOFTWARE='1', EGL_PLATFORM='surfaceless', PYOPENGL_PLATFORM='osmesa')
        env.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES', '/usr/share/glvnd/egl_vendor.d/50_mesa.json')
    else:
        for name in ('LIBGL_ALWAYS_SOFTWARE', 'EGL_PLATFORM', '__GLX_VENDOR_LIBRARY_NAME'):
            env.pop(name, None)
        if 'mesa' in env.get('__EGL_VENDOR_LIBRARY_FILENAMES', '').lower():
            env.pop('__EGL_VENDOR_LIBRARY_FILENAMES')
        env['PYOPENGL_PLATFORM'] = 'egl'
    return env


def simulator_gpu_id():
    return 0 if backend() == 'nvidia' else -1


def configuration(environment=None):
    env = os.environ if environment is None else environment
    keys = ('CUDA_VISIBLE_DEVICES', 'EGL_DEVICE_ID', 'PYOPENGL_PLATFORM',
            'LIBGL_ALWAYS_SOFTWARE', 'EGL_PLATFORM', '__EGL_VENDOR_LIBRARY_FILENAMES')
    return {'backend': backend(env), **{key: env.get(key) for key in keys}}
