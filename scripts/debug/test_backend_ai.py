#!/usr/bin/env python3
"""
Test script for backend.ai command execution
"""

import os
import subprocess
import sys
import time

def main():
    """
    Test backend.ai command execution
    """
    print("Testing backend.ai command execution")
    
    # Check if backend.ai is in PATH
    which_backend_cmd = ["which", "backend.ai"]
    try:
        which_result = subprocess.run(which_backend_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"backend.ai path: {which_result.stdout.strip()}")
        if which_result.returncode != 0:
            print(f"Error finding backend.ai: {which_result.stderr}")
            return
    except Exception as e:
        print(f"Error checking backend.ai path: {str(e)}")
        return

    # Check if backend.ai is authenticated
    auth_check_cmd = ["backend.ai", "session", "list", "--limit", "1"]
    try:
        auth_result = subprocess.run(auth_check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        print(f"Auth check return code: {auth_result.returncode}")
        if auth_result.returncode != 0:
            print(f"Auth check error: {auth_result.stderr}")
            print("backend.ai might not be authenticated. Try running 'backend.ai login' manually first.")
            return
        else:
            print("backend.ai appears to be authenticated")
    except Exception as e:
        print(f"Error checking backend.ai authentication: {str(e)}")
        return

    # Simple test command
    test_cmd = ["backend.ai", "session", "list", "--limit", "5"]
    print(f"Running test command: {' '.join(test_cmd)}")
    
    # Method 1: subprocess.run with shell=False
    try:
        print("\nMethod 1: subprocess.run with shell=False")
        start_time = time.time()
        result = subprocess.run(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        end_time = time.time()
        print(f"Command completed in {end_time - start_time:.2f} seconds with return code: {result.returncode}")
        print(f"Output: {result.stdout}")
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"Error running command: {str(e)}")

    # Method 2: subprocess.run with shell=True
    try:
        print("\nMethod 2: subprocess.run with shell=True")
        shell_cmd = ' '.join(test_cmd)
        start_time = time.time()
        result = subprocess.run(shell_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        end_time = time.time()
        print(f"Command completed in {end_time - start_time:.2f} seconds with return code: {result.returncode}")
        print(f"Output: {result.stdout}")
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"Error running command: {str(e)}")

    # Method 3: os.system
    try:
        print("\nMethod 3: os.system")
        shell_cmd = ' '.join(test_cmd)
        start_time = time.time()
        return_code = os.system(shell_cmd)
        end_time = time.time()
        print(f"Command completed in {end_time - start_time:.2f} seconds with return code: {return_code}")
    except Exception as e:
        print(f"Error running command: {str(e)}")

    print("\nTest completed")

if __name__ == "__main__":
    main()
