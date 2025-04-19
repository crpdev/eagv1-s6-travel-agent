# AI Travel Advisor Agent

An intelligent travel advisor agent that provides personalized travel recommendations based on user preferences. The agent is built with a functional programming architecture consisting of four main modules: Perception, Memory, Decision-Making, and Action.

## Features

- Functional programming architecture with immutable state
- Personalized travel recommendations based on user preferences
- Contextual understanding of user queries using Google's Gemini AI
- Persistent memory with immutable state management
- Structured decision-making process
- Clear reasoning and confidence scoring
- State persistence across sessions

## Architecture

The agent consists of four main modules, all implemented using functional programming principles:

1. **Perception (LLM)**: Pure functions for user input processing and natural language understanding using Google's Gemini Pro model
2. **Memory**: Immutable state management for conversation history, user preferences, and recommendations
3. **Decision-Making**: Pure functions for analyzing input and determining appropriate actions
4. **Action**: Pure functions for executing decisions and generating formatted responses

## Setup

1. Clone the repository

2. Install uv (if not already installed):
   ```bash
   pip install uv
   ```

3. Create and activate a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate  # On Unix/macOS
   .venv\Scripts\activate     # On Windows
   ```

4. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

5. Set up your Google API key:
   - Option 1: Set environment variable:
     ```bash
     export GOOGLE_API_KEY='your-api-key'     # Unix/macOS
     set GOOGLE_API_KEY='your-api-key'        # Windows
     ```
   - Option 2: Pass directly when initializing the agent

## Usage

Run the agent:
```bash
python main.py
```

The agent will:
1. Ask for your travel preferences
2. Store these preferences in an immutable state
3. Accept your travel-related queries
4. Provide recommendations based on your preferences and context

Example interactions:
```
> Suggest a weekend getaway
> What are some budget-friendly destinations in Europe?
> Find me activities in Barcelona
```

## State Management

The agent uses immutable state management through:
- `NamedTuple` and `dataclass` for type-safe state containers
- Pure functions for state transformations
- Persistent data structures for efficient immutability
- Automatic state persistence to `agent_state.json`

## Requirements

- Python 3.7+
- Google API key for Gemini Pro
- uv package manager
- Dependencies listed in pyproject.toml

## Development

The codebase follows functional programming principles:
- Pure functions
- Immutable state
- Type hints
- No side effects (except for I/O operations)
- Composition over inheritance 