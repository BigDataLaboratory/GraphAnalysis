import logging
import os
import sys
import tempfile
import time

import torch
if sys.platform == 'win32':
    import msvcrt
else:
    import fcntl
    
logger = logging.getLogger(__name__)

LOCK_DIR = os.path.join(tempfile.gettempdir(), 'gpu_locks')


def acquire_gpu(retry_interval=10, timeout=3600):
    """
    Waits for a free GPU and acquires it via a file lock.

    :param retry_interval: Seconds between retries.
    :param timeout: Max seconds to wait before raising TimeoutError.
    :return: (gpu_id, lock_file_handle)
    """
    if not torch.cuda.is_available():
        logger.warning("No CUDA device available, falling back to CPU.")
        return None, None

    os.makedirs(LOCK_DIR, exist_ok=True)
    n_gpus = torch.cuda.device_count()
    elapsed = 0

    while elapsed < timeout:
        for gpu_id in range(n_gpus):
            lock_path = os.path.join(LOCK_DIR, f'gpu_{gpu_id}.lock')
            f = open(lock_path, 'w')
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                logger.info(f"GPU {gpu_id} acquired after {elapsed}s.")
                return gpu_id, f
            except BlockingIOError:
                f.close()

        logger.info(f"All GPUs busy, retrying in {retry_interval}s... ({elapsed}s/{timeout}s)")
        time.sleep(retry_interval)
        elapsed += retry_interval

    raise TimeoutError(f"No GPU available after {timeout}s.")


def release_gpu(gpu_id, lock_file):
    """
    Releases the GPU lock.

    :param gpu_id: GPU index to release.
    :param lock_file: File handle returned by acquire_gpu.
    """
    if lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()
        logger.info(f"GPU {gpu_id} released.")