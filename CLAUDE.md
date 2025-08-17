# Cypher AI Assistant Project

## Project Overview
Cypher is an AI assistant project with multiple capabilities. The current focus is on the `orpheus/` folder, which contains the Text-to-Speech (TTS) implementation that needs to be integrated into the main workflow.

## Tech Stack
- **Primary Language**: Python 3
- **Secondary**: Small amounts of HTML, JavaScript, and Bash
- **Architecture**: Single repository with modular structure
- **Deployment**: Server environment (no IDE access)

## Project Structure
- **Main codebase**: Core AI assistant functionality
- **`orpheus/`**: TTS implementation (current focus area)
  - Contains TTS-related code that will eventually integrate with main workflow
  - Currently has excessive files and needs cleanup/organization
- **Mixed structure**: May appear messy but follows understandable patterns

## Development Environment
- **Runtime**: `python3` command for server execution
- **Environment**: Server deployment (no local IDE)
- **Team**: Solo developer

## Code Standards
- **Style**: Industrial, professional, maintainable, and modular
- **Structure**: Prefer clean, organized, modular code
- **Documentation**: Clear and professional

## Primary Objectives for Orpheus Folder

### 1. Deep Understanding Required
- **MUST** analyze the entire codebase structure first
- **MUST** understand the relationship between main codebase and orpheus folder
- **MUST** understand why Orpheus TTS is needed in the broader Cypher ecosystem
- **MUST** map out current file dependencies and usage patterns

### 2. Cleanup and Organization Goals
- **Current Problem**: Too many unused/unnecessary files in orpheus folder
- **Target Solution**: Consolidate into a few well-organized `.sh` scripts
- **Deployment Goal**: Simple execution via `chmod +x script.sh && bash script.sh`
- **Alternative**: Propose even more professional solutions than basic shell scripts

### 3. Analysis and Planning Workflow
- **Step 1**: Comprehensive codebase analysis and understanding
- **Step 2**: Map current orpheus folder structure and identify unused files
- **Step 3**: Propose cleanup and organization strategy
- **Step 4**: **WAIT FOR EXPLICIT APPROVAL** before any code changes

## Critical Rules - NO CODE WITHOUT APPROVAL

### MANDATORY APPROVAL PROCESS
1. **NEVER** delete, modify, or create files without explicit approval
2. **ALWAYS** present a detailed plan before suggesting any changes
3. **MUST** explain the reasoning behind proposed solutions
4. **REQUIRED** to wait for user confirmation before proceeding

### Analysis-Only Phase
- Explore and understand all code
- Read all files in orpheus folder and main codebase
- Identify patterns, dependencies, and unused components
- Map the relationship between orpheus and main project
- Propose solutions and optimizations

### Execution Phase (Only After Approval)
- Implement approved changes only
- Follow the exact plan that was approved
- Create clean, professional, modular solutions

## Commands
- **Run Server**: `python3 [main_server_file]`
- **Make Executable**: `chmod +x script.sh`
- **Run Script**: `bash script.sh`

## Orpheus-Specific Instructions

### Understanding Phase
1. Read and analyze all files in the orpheus folder
2. Understand the TTS implementation architecture
3. Identify which files are actually used vs unused
4. Map dependencies between files
5. Understand how orpheus should integrate with main Cypher workflow

### Cleanup Proposal Phase
1. Identify unused/redundant files
2. Group related functionality
3. Propose script-based organization (or better alternatives)
4. Design clean deployment workflow
5. Present comprehensive reorganization plan

### Professional Solutions to Consider
- Shell scripts for different TTS operations
- Python modules with proper imports
- Configuration-based setup
- Docker containers (if applicable)
- Virtual environment setup scripts
- Automated dependency management

## Success Criteria
- Clean, organized orpheus folder with minimal necessary files
- Professional deployment process
- Clear understanding of TTS integration path
- Maintainable and modular code structure
- Simple server deployment workflow

## Constraints
- **File Operations**: Must get approval before any file changes
- **Server Environment**: Solutions must work without IDE
- **Simplicity**: Prefer simple, reliable solutions over complex ones
- **Modularity**: Maintain clean separation of concerns

## Current Focus Areas
1. **orpheus/** folder cleanup and organization
2. Understanding TTS integration requirements
3. Streamlining server deployment process
4. Reducing file complexity while maintaining functionality

---

**REMEMBER**: This is an analysis and planning phase. Understand everything first, propose solutions second, execute only after explicit approval.