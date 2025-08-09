# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Cypher Chat Assistant** - Conversational AI assistant being refactored from monolithic ChatManager to modern pipeline pattern.

## Current Structure
- `interfaces/chat.py` - Main conversation manager (NEEDS REFACTORING TO PIPELINE)
- `core/nlp.py` - LLM API interface
- `core/memory.py` - Conversation history management  
- `services/utilities.py` - Service dispatch system
- `interfaces/speech.py` - TTS/STT handling
- `config.py` - Configuration settings
- `main.py` - Entry point

## Refactoring Goal
Transform ChatManager in `interfaces/chat.py` into pipeline pattern:
**BasicContextHandler → VisualContextHandler → LLMHandler → ResponseFormatter → TTSHandler**

## TTS Architecture Revolution - Critical Decision Point

### Session Summary (ULTRATHINK Analysis Completed)
We conducted deep analysis of TTS solutions for human-like conversational voice with <150ms perceived latency.

### Key Discoveries

#### DIA Analysis Results:
- **NO STREAMING**: Uses DAC codec with delay patterns requiring complete sequence
- **Voice Cloning**: Uses audio prompts, not embeddings (good for consistency)
- **Emotions**: Text-based only "(laughs)" - no parameter control
- **Deployment**: Requires vast.ai GPU, 1-2s generation time
- **Modification Difficulty**: DAC streaming modifications = 2-3 weeks complex work

#### FishAudio (OpenAudio S1) - GAME CHANGER:
- **NATIVE STREAMING**: Already implemented, works out-of-box!
- **#1 on TTS-Arena2**: Beats ElevenLabs, Google, OpenAI
- **FREE & OPEN SOURCE**: Apache license code, CC-BY-NC-SA weights
- **Two Models**: S1-mini (0.5B local) + S1 (4B remote)
- **40+ Emotions Built-in**: (angry), (excited), (whispering) etc.
- **Voice Cloning**: 10-30 second sample
- **Performance**: 0.008 WER, real-time factor 1:7 on RTX 4090

### Architecture Decision - FishAudio Wins:
1. **Streaming eliminates RTT chunking issues** (DIA's main problem)
2. **Emotion markers work natively** - LLM generates tags, Fish speaks them
3. **Dual model setup perfect**: S1-mini local + S1 remote on vast.ai
4. **No codec modifications needed** - saves weeks of work

### Implementation Plan:
1. **Test FishAudio S1-mini locally** - Verify streaming works
2. **Record voice samples** - 10-30 seconds for cloning
3. **LLM emotion tag integration** - Modify LLM prompts to generate emotion markers
4. **BatonNet unnecessary** - Streaming eliminates complex handoff needs
5. **Deploy S1 on vast.ai** - For premium quality after local preview

### Current Status:
- Both DIA and FishAudio repos cloned
- FishAudio S1-mini weights need verification (check HuggingFace availability)
- **CONCERN**: Full S1 model weights availability unclear - need to verify

### Next Session Priority:
1. **Verify FishAudio model weight availability** (both S1-mini and S1)
2. **Test FishAudio streaming with S1-mini**
3. **Implement emotion tag generation in LLM handler**
4. **Test voice cloning with personal samples**
5. **Compare actual quality vs DIA**

### Dependencies Policy
- Use ONLY standard library (asyncio, logging, typing, dataclasses, abc)
- Preserve existing project modules (config, memory, nlp, utilities)
- NO external pipeline libraries
- Maintain all existing functionality
- Backward compatibility required

## Code Standards
- Preserve error handling and logging patterns
- Use type hints throughout  
- Follow existing async/await patterns
- Keep interfaces/ directory structure