import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from perception import configure_gemini, get_user_preferences, process_input
from memory import (
    MemoryState, create_memory_state, store_preferences,
    add_to_history, get_recent_context, save_to_file,
    load_from_file
)
from decision_making import analyze_input, DecisionType
from action import execute, log_action
from logger_config import setup_logging, get_logger

# Load environment variables from .env file
load_dotenv()

# Set up logging
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", None)
)

logger = get_logger(__name__)

@dataclass
class AgentState:
    """Container for agent state."""
    memory: MemoryState
    api_key: str
    awaiting_clarification: bool = False
    last_decision: Optional[Dict] = None

def create_agent(api_key: Optional[str] = None) -> AgentState:
    """Create a new agent state."""
    logger.info("Creating new agent instance")
    
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("No API key provided")
        raise ValueError(
            "Google API key must be provided either directly or through environment variable GOOGLE_API_KEY.\n"
            "You can set it in your .env file or export it as an environment variable."
        )
    
    try:
        configure_gemini(api_key)
        agent_state = AgentState(create_memory_state(), api_key)
        logger.info("Agent created successfully")
        return agent_state
    except Exception as e:
        logger.error("Failed to create agent", extra_data={'error': str(e)})
        raise

def initialize_preferences(state: AgentState) -> AgentState:
    """Get initial user preferences and store them."""
    logger.info("Initializing user preferences")
    try:
        preferences = get_user_preferences()
        new_state = AgentState(
            store_preferences(state.memory, preferences),
            state.api_key
        )
        logger.info("Preferences initialized successfully", 
                   extra_data={'num_preferences': len(preferences)})
        return new_state
    except Exception as e:
        logger.error("Failed to initialize preferences", 
                    extra_data={'error': str(e)})
        raise

def format_response_for_display(response: str) -> str:
    """Format the response for better console display."""
    # Split response into sections
    sections = response.split('\n\n')
    formatted_sections = []
    
    for section in sections:
        if section.strip():
            # Handle the destination section
            if section.startswith('Destination:'):
                formatted_sections.append(section)
            # Handle the activities section
            elif section.startswith('Suggested Activities:'):
                activities = ['Suggested Activities:']
                for line in section.split('\n')[1:]:  # Skip the header
                    if line.strip():
                        # Clean up markdown and formatting
                        line = line.replace('*', '').replace('**', '')
                        if line.startswith('- '):
                            line = '  ' + line  # Add indentation
                        activities.append(line)
                formatted_sections.append('\n'.join(activities))
            # Handle other sections
            else:
                formatted_sections.append(section)
    
    # Join sections with double newlines for better readability
    return '\n\n' + '\n\n'.join(formatted_sections) + '\n'

def process_query(state: AgentState, user_input: str) -> Tuple[AgentState, str]:
    """Process a user query and return updated state and response."""
    logger.info("Processing user query", 
                extra_data={
                    'input_length': len(user_input),
                    'awaiting_clarification': state.awaiting_clarification
                })
    
    try:
        # Get context and preferences
        context = get_recent_context(state.memory)
        preferences = state.memory.user_preferences
        
        logger.debug("Retrieved context and preferences", 
                    extra_data={
                        'context_size': len(context),
                        'num_preferences': len(preferences)
                    })
        
        # Process input through perception
        processed_input = process_input(
            user_input,
            context,
            preferences,
            is_clarification_response=state.awaiting_clarification
        )
        
        if not processed_input["success"]:
            raise Exception(processed_input["error"])
        
        # Make decision
        decision = analyze_input(
            processed_input,
            context,
            preferences
        )
        
        # Execute action
        action_result = execute(decision)
        
        # Store interaction in memory
        new_memory = add_to_history(
            state.memory,
            {
                "user_input": user_input,
                "processed_input": processed_input,
                "decision": decision,
                "action_result": action_result
            }
        )
        
        # Update state based on decision type
        # Only keep clarification state if we're not already in one
        # This prevents loops of clarification requests
        new_awaiting_clarification = (
            decision.decision_type == DecisionType.CLARIFICATION
            and not state.awaiting_clarification
        )
        
        new_state = AgentState(
            new_memory,
            state.api_key,
            awaiting_clarification=new_awaiting_clarification,
            last_decision=decision
        )
        
        logger.info("Query processed successfully", 
                   extra_data={
                       'decision_type': str(decision.decision_type),
                       'confidence': decision.confidence,
                       'awaiting_clarification': new_state.awaiting_clarification
                   })
        
        # Format the response for display
        formatted_response = format_response_for_display(action_result["response"])
        return new_state, formatted_response
        
    except Exception as e:
        logger.error("Failed to process query", 
                    extra_data={
                        'error': str(e),
                        'error_type': type(e).__name__
                    })
        raise

def save_agent_state(state: AgentState, filename: Optional[str] = None) -> None:
    """Save the agent's memory state to a file."""
    filename = filename or os.getenv("MEMORY_FILE", "agent_state.json")
    logger.info("Saving agent state", extra_data={'filename': filename})
    
    try:
        save_to_file(state.memory, filename)
        logger.debug("Agent state saved successfully")
    except Exception as e:
        logger.error("Failed to save agent state", 
                    extra_data={'error': str(e)})
        raise

def load_agent_state(api_key: str, filename: Optional[str] = None) -> Optional[AgentState]:
    """Load the agent's memory state from a file."""
    filename = filename or os.getenv("MEMORY_FILE", "agent_state.json")
    logger.info("Loading agent state", extra_data={'filename': filename})
    
    try:
        memory = load_from_file(filename)
        if memory:
            logger.debug("Agent state loaded successfully")
            return AgentState(memory, api_key)
        else:
            logger.warning("No existing agent state found")
            return None
    except Exception as e:
        logger.error("Failed to load agent state", 
                    extra_data={'error': str(e)})
        return None

def main():
    """Main function to demonstrate the agent's usage."""
    logger.info("Starting Travel Advisor application")
    
    try:
        print("\n=== Welcome to your AI Travel Advisor! ===")
        print("Initializing the agent...")
        
        # Initialize the agent
        agent_state = create_agent()
        
        print("\nFirst, I'd like to get to know your travel preferences better.\n")
        
        # Get initial preferences
        agent_state = initialize_preferences(agent_state)
        
        print("\nGreat! I now have a better understanding of your preferences.")
        print("You can ask me for travel recommendations or advice at any time.")
        print("Type 'quit' to exit.\n")
        
        # Main interaction loop
        while True:
            user_input = input("> ").strip()
            
            if not user_input:
                print("\nPlease enter a query or type 'quit' to exit.\n")
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                logger.info("User requested to quit")
                print("\nThank you for using the AI Travel Advisor. Safe travels!")
                save_agent_state(agent_state)  # Save state before exiting
                break
            
            agent_state, response = process_query(agent_state, user_input)
            print(response)
            
    except Exception as e:
        logger.critical("Application error", 
                       extra_data={
                           'error': str(e),
                           'error_type': type(e).__name__
                       })
        print(f"\nAn error occurred: {str(e)}")
        print("\nPlease make sure you have:")
        print("1. Set up your Google API key in the .env file or as an environment variable")
        print("2. Installed all required dependencies using 'pip install -r requirements.txt'")
        print("3. Activated your virtual environment")
    finally:
        logger.info("Shutting down Travel Advisor application")

if __name__ == "__main__":
    main() 