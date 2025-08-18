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

### 2. Critical Audio Output Investigation
- **Primary Issue**: TTS system limited to maximum 24 seconds of audio output
- **Extended Issue**: When increasing token output for longer text:
  - Server produces maximum 40 seconds of audio
  - Audio starts in the middle of text instead of from the beginning
  - Complete text is not being processed properly

### 3. Senior Engineering Analysis Required
- **ULTRATHINK** like a professional senior software engineer
- **Deep dive** into the server architecture and TTS processing pipeline
- **Identify bottlenecks** in audio generation and text processing
- **Analyze** token handling, memory management, and audio streaming
- **Investigate** text chunking, audio concatenation, and buffer management
- **Examine** server limitations, timeout issues, and resource constraints

### 4. Analysis and Planning Workflow
- **Step 1**: Comprehensive codebase analysis and understanding
- **Step 2**: Deep investigation of TTS audio length limitations
- **Step 3**: Identify root causes of 24-second limit and mid-text starting issues
- **Step 4**: Engineer professional solutions for extended audio generation
- **Step 5**: **WAIT FOR EXPLICIT APPROVAL** before any code changes

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
3. Map the complete audio generation pipeline
4. Identify server-side processing limitations
5. Understand how orpheus integrates with main Cypher workflow

### Critical Investigation Phase - Audio Length Limitations
1. **Investigate 24-second limit**:
   - Analyze audio generation code for hard limits
   - Check buffer sizes, memory allocation, and streaming limits
   - Examine token processing and text chunking mechanisms
   - Identify timeout configurations and resource constraints

2. **Diagnose 40-second + mid-text issue**:
   - Trace text processing pipeline from input to audio output
   - Analyze text segmentation and audio concatenation logic
   - Check for buffer overflow, memory leaks, or processing interruptions
   - Examine async processing, threading, and queue management

3. **Senior Engineering Analysis**:
   - **Performance bottlenecks**: CPU, memory, I/O constraints
   - **Architecture review**: Streaming vs batch processing
   - **Scalability issues**: Token limits, API constraints, server capacity
   - **Error handling**: Graceful degradation and recovery mechanisms
   - **Resource management**: Memory cleanup, garbage collection
   - **Threading/Async**: Concurrent processing optimization

### Professional Solution Design Phase
1. Engineer solutions for extended audio generation (>24 seconds)
2. Design robust text-to-audio pipeline for complete text processing
3. Propose server architecture improvements
4. Design professional deployment and scaling strategies
5. Present comprehensive technical improvement plan

### Professional Engineering Solutions to Consider
- **Streaming Architecture**: Real-time audio streaming with chunked processing
- **Queue-based Processing**: Asynchronous text processing with audio concatenation
- **Memory Optimization**: Efficient buffer management and resource cleanup
- **Text Segmentation**: Intelligent text chunking with context preservation
- **Error Recovery**: Robust handling of processing failures and timeouts
- **Load Balancing**: Distributed processing for longer texts
- **Caching Mechanisms**: Audio segment caching for improved performance

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


## orpheus
1. The orpheus TTS repo can be found in the code base also read it throughfully. 
---

**REMEMBER**: This is an analysis and planning phase. Understand everything first, propose solutions second, execute only after explicit approval.