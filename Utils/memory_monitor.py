import os
import psutil
import functools
import logging
from datetime import datetime
import tracemalloc
from typing import Callable, Any


# Set up logging
def setup_memory_logging(log_file: str = "memory_usage.log") -> None:
    """Configure logging for memory tracking"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def get_memory_usage() -> tuple[float, float, float]:
    """Get current memory usage statistics"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    # Convert to GB for readability
    rss_gb = memory_info.rss / (1024 ** 3)  # Resident Set Size
    vms_gb = memory_info.vms / (1024 ** 3)  # Virtual Memory Size
    memory_percent = process.memory_percent()

    return rss_gb, vms_gb, memory_percent


def log_memory(message: str = "") -> None:
    """Log current memory usage with optional message"""
    rss_gb, vms_gb, memory_percent = get_memory_usage()
    logging.info(f"{message} - RSS: {rss_gb:.2f}GB, VMS: {vms_gb:.2f}GB, Memory Usage: {memory_percent:.2f}%")


def memory_tracker(func: Callable) -> Callable:
    """Decorator to track memory usage before and after function execution"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        func_name = func.__name__
        logging.info(f"Starting {func_name}")

        # Start memory tracking
        tracemalloc.start()
        start_time = datetime.now()
        log_memory(f"Memory before {func_name}")

        try:
            # Execute the function
            result = func(*args, **kwargs)

            # Log memory after execution
            log_memory(f"Memory after {func_name}")
            current, peak = tracemalloc.get_traced_memory()
            execution_time = datetime.now() - start_time

            logging.info(f"{func_name} completed in {execution_time}")
            logging.info(f"Memory tracking for {func_name}:")
            logging.info(f"Current memory: {current / (1024 ** 3):.2f}GB")
            logging.info(f"Peak memory: {peak / (1024 ** 3):.2f}GB")

            return result

        finally:
            tracemalloc.stop()

    return wrapper