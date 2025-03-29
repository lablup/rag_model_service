# RAG Model Service Changes

## GitHub Cloning Optimization

### Date: 2025-03-28

### Description
Modified the GitHub cloning functionality to only clone the specific directory (e.g., docs) when a user provides a GitHub URL that points to a specific path, rather than cloning the entire repository. This makes the RAG service creation process more efficient.

### Files Changed

1. **utils/github_utils.py**
   - Enhanced `clone_github_repo` function to use SVN export for specific directories
   - Added fallback to shallow git clone if SVN is not available
   - Optimized with `--depth=1` and `--single-branch` options for faster, more efficient cloning

2. **interfaces/portal/github.py**
   - Updated `clone_github_repo` function to provide better user feedback
   - Added specific messaging when sparse cloning a directory

3. **interfaces/cli_app/github_cli.py**
   - Updated to use the improved GitHub cloning functionality
   - Added better feedback for sparse cloning
   - Added a dedicated async function for repository cloning

### Testing
- Created test scripts to verify the functionality:
  - `scripts/debug/test_sparse_checkout.py`: Tests the sparse checkout functionality
  - `scripts/debug/test_github_cli.sh`: Tests the CLI tool with sparse checkout

### Benefits
- Faster RAG service creation when users provide URLs to specific documentation directories
- Reduced disk space usage by only cloning necessary files
- Improved user experience with better feedback during the cloning process
- Consistent behavior across all interfaces (portal, CLI)
