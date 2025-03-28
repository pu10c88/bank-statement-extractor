#!/usr/bin/env python
"""
Utility script to clean up Streamlit processes and cache
Run this if you're having problems with Streamlit not showing updated data
"""

import os
import sys
import shutil
import subprocess
import psutil
import signal
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_streamlit_cache():
    """Clean the Streamlit cache directory"""
    cache_dir = os.path.expanduser("~/.streamlit/cache")
    if os.path.exists(cache_dir):
        logger.info(f"Removing Streamlit cache directory: {cache_dir}")
        try:
            shutil.rmtree(cache_dir, ignore_errors=True)
            logger.info("Cache directory removed successfully")
        except Exception as e:
            logger.error(f"Error removing cache: {str(e)}")
    else:
        logger.info("No Streamlit cache directory found")

def kill_streamlit_processes():
    """Kill all running Streamlit processes"""
    killed_count = 0
    current_pid = os.getpid()
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            pid = proc.info['pid']
            
            # Skip the current process
            if pid == current_pid:
                continue
                
            cmdline = proc.info['cmdline'] if proc.info['cmdline'] else []
            cmdline_str = ' '.join(cmdline)
            
            # Look for streamlit processes
            if 'streamlit' in cmdline_str and 'clean_streamlit.py' not in cmdline_str:
                logger.info(f"Killing Streamlit process {pid}: {cmdline_str[:100]}...")
                try:
                    os.kill(pid, signal.SIGTERM)
                    killed_count += 1
                except Exception as e:
                    logger.error(f"Error killing process {pid}: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing {proc.info['pid'] if 'pid' in proc.info else 'unknown'}: {str(e)}")
    
    logger.info(f"Killed {killed_count} Streamlit processes")
    return killed_count

def main():
    """Main function to clean up Streamlit"""
    logger.info("Starting Streamlit cleanup utility")
    
    # Kill all Streamlit processes
    killed_count = kill_streamlit_processes()
    
    # If we killed some processes, wait a moment for them to fully terminate
    if killed_count > 0:
        logger.info("Waiting for processes to terminate...")
        time.sleep(2)
    
    # Clean the Streamlit cache
    clean_streamlit_cache()
    
    logger.info("Streamlit cleanup completed")
    
if __name__ == "__main__":
    main() 