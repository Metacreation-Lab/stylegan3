"""Shared build-cache utilities for the custom C++/CUDA ops.

Both the JIT loader (`torch_utils.ops.custom_ops`) and the standalone
precompiler (`scripts/precompile_ops.py`) build into the same on-disk cache.
A cache entry is keyed by (source digest, torch version, CUDA version,
compute capability) -- never by GPU name -- so entries are shared across
GPUs of the same compute capability and can be produced on one machine and
consumed on another. An entry is only trusted once its completion marker
exists; incomplete entries are treated as failed builds and deleted once
they look abandoned. A complete entry is imported directly from disk,
without compiler or CUDA toolkit access.
"""

import glob
import hashlib
import importlib.machinery
import importlib.util
import os
import shutil
import sys
import time
import uuid

import torch
import torch.utils.cpp_extension

_COMPLETE_MARKER = '.build_complete'
_LOCK_STALE_SECONDS = 30 * 60 # a crashed build leaves its lock behind; consider it dead after this
_DIR_GRACE_SECONDS = 60       # freshly created entries may still be racing towards their build lock

#----------------------------------------------------------------------------
# Cache key and directory resolution.

def source_digest(source_files):
    hash_md5 = hashlib.md5()
    for src in sorted(source_files):
        with open(src, 'rb') as f:
            hash_md5.update(f.read())
    return hash_md5.hexdigest()

def current_capability():
    major, minor = torch.cuda.get_device_capability()
    return f'{major}.{minor}'

def plugin_build_dir(module_name, source_files, capability, verbose=False):
    assert torch.version.cuda is not None, 'CUDA-enabled torch build required'
    root = torch.utils.cpp_extension._get_build_directory(module_name, verbose=verbose) # pylint: disable=protected-access
    key = f'{source_digest(source_files)}-torch{torch.__version__}-cuda{torch.version.cuda}-sm{capability.replace(".", "")}'
    return os.path.join(root, key)

#----------------------------------------------------------------------------
# Entry completion state and failed-build cleanup.

def _artifact_path(build_dir, module_name):
    ext = '.pyd' if os.name == 'nt' else '.so'
    return os.path.join(build_dir, module_name + ext)

def is_complete(build_dir, module_name):
    return os.path.isfile(os.path.join(build_dir, _COMPLETE_MARKER)) and os.path.isfile(_artifact_path(build_dir, module_name))

def mark_complete(build_dir, module_name):
    artifact = _artifact_path(build_dir, module_name)
    assert os.path.isfile(artifact), f'build did not produce {artifact}'
    with open(os.path.join(build_dir, _COMPLETE_MARKER), 'w') as f:
        f.write(os.path.basename(artifact) + '\n')

def clean_failed_build(build_dir, module_name):
    """Delete an incomplete cache entry unless a live build seems to own it."""
    if not os.path.isdir(build_dir) or is_complete(build_dir, module_name):
        return
    now = time.time()
    lock = os.path.join(build_dir, 'lock') # FileBaton used by torch.utils.cpp_extension.load()
    if os.path.exists(lock):
        if now - os.path.getmtime(lock) < _LOCK_STALE_SECONDS:
            return # build in progress; torch's FileBaton will serialize us behind it
    elif now - os.path.getmtime(build_dir) < _DIR_GRACE_SECONDS:
        return # entry just created here or by a concurrent process
    shutil.rmtree(build_dir, ignore_errors=True)

#----------------------------------------------------------------------------
# Source staging (atomic, timestamp-stable so ninja rebuilds stay incremental).

def populate_sources(build_dir, source_files):
    if os.path.isdir(build_dir):
        return
    tmpdir = os.path.join(os.path.dirname(build_dir), f'srctmp-{uuid.uuid4().hex}')
    os.makedirs(tmpdir)
    for src in source_files:
        shutil.copyfile(src, os.path.join(tmpdir, os.path.basename(src)))
    try:
        os.replace(tmpdir, build_dir) # atomic
    except OSError:
        # Entry appeared concurrently; discard our copy.
        shutil.rmtree(tmpdir)
        if not os.path.isdir(build_dir):
            raise

#----------------------------------------------------------------------------
# Loading a completed entry (no compiler, no CUDA toolkit, no ninja).

def import_from_cache(module_name, build_dir):
    if module_name in sys.modules:
        return sys.modules[module_name]
    filename = _artifact_path(build_dir, module_name)
    loader = importlib.machinery.ExtensionFileLoader(module_name, filename)
    spec = importlib.util.spec_from_file_location(module_name, filename, loader=loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    loader.exec_module(module)
    return module

#----------------------------------------------------------------------------
# Compiler environment.

def _find_compiler_bindir():
    patterns = [
        'C:/Program Files*/Microsoft Visual Studio/*/Professional/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio/*/Community/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio */vc/bin',
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if len(matches):
            return matches[-1]
    return None

def setup_compiler_env():
    """Make sure the C++ compiler is reachable; raise if it cannot be found."""
    if os.name == 'nt' and os.system('where cl.exe >nul 2>nul') != 0:
        compiler_bindir = _find_compiler_bindir()
        if compiler_bindir is None:
            raise RuntimeError(f'Could not find MSVC/GCC/CLANG installation on this computer. Check _find_compiler_bindir() in "{__file__}".')
        os.environ['PATH'] += ';' + compiler_bindir

def pin_arch_list(capability=None):
    # Match upstream custom_ops: an empty TORCH_CUDA_ARCH_LIST makes nvcc
    # target the current device, and overriding neutralizes container-set
    # values that could break the build or target the wrong archs. The
    # precompile script passes an explicit capability to cross-build.
    os.environ['TORCH_CUDA_ARCH_LIST'] = capability if capability is not None else ''

#----------------------------------------------------------------------------
