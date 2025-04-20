import os
import json
from datetime import datetime
from typing import Dict, Optional
from dotenv import load_dotenv
from perception import configure_gemini, get_user_preferences, process_input
from logger_config import setup_logging, get_logger
from models import (
    AgentState, MemoryState, Interaction, Decision,
    DecisionType, ProcessedInput, UserPreferences
)

# Initialize logging
setup_logging()
logger = get_logger(__name__)

def create_agent() -> AgentState:
    """Create and initialize a new agent state."""
    logger.info("Creating new agent state")
    try:
        # Force reload environment variables
        load_dotenv(override=True)
        
        # Try multiple ways to get the API key
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            # Try getting it directly from environment
            api_key = os.environ.get('GOOGLE_API_KEY')
        
        logger.debug("API Key status", extra_data={
            'api_key_exists': bool(api_key),
            'api_key_length': len(api_key) if api_key else 0,
            'env_vars': {k: bool(v) for k, v in os.environ.items() if 'API' in k}
        })
        
        if not api_key:
            logger.error("Google API key not found in environment variables")
            raise ValueError(
                "Google API key not found in environment variables. "
                "Please ensure GOOGLE_API_KEY is set in your .env file "
                "and the file is in the correct directory."
            )
        
        # Configure Gemini with explicit key
        configure_gemini(api_key)
        
        # Get initial user preferences
        preferences = get_user_preferences()
        
        # Create initial memory state
        memory = MemoryState(
            user_preferences=preferences,
            interaction_history=[],
            last_interaction_time=datetime.now()
        )
        
        # Create and return agent state
        agent_state = AgentState(
            memory=memory,
            api_key=api_key,
            awaiting_clarification=False,
            last_decision=None
        )
        
        logger.info("Agent state created successfully")
        return agent_state
    except Exception as e:
        logger.error("Failed to create agent state", extra_data={'error': str(e)})
        raise

def save_agent_state(agent_state: AgentState, filename: str = "agent_state.json") -> None:
    """Save the agent's state to a file."""
    logger.info("Saving agent state", extra_data={'filename': filename})
    try:
        # Convert to dict and save
        state_dict = agent_state.model_dump()
        with open(filename, 'w') as f:
            json.dump(state_dict, f, default=str)
        logger.debug("Agent state saved successfully")
    except Exception as e:
        logger.error("Failed to save agent state", 
                    extra_data={'error': str(e), 'filename': filename})
        raise

def load_agent_state(filename: str = "agent_state.json") -> Optional[AgentState]:
    """Load the agent's state from a file."""
    logger.info("Loading agent state", extra_data={'filename': filename})
    try:
        if not os.path.exists(filename):
            logger.warning("Agent state file not found", extra_data={'filename': filename})
            return None
        
        with open(filename, 'r') as f:
            state_dict = json.load(f)
        
        # Convert datetime strings back to datetime objects
        if 'memory' in state_dict:
            if 'last_interaction_time' in state_dict['memory']:
                state_dict['memory']['last_interaction_time'] = \
                    datetime.fromisoformat(state_dict['memory']['last_interaction_time'])
        
        # Create AgentState from dict
        agent_state = AgentState.model_validate(state_dict)
        logger.debug("Agent state loaded successfully")
        return agent_state
    except Exception as e:
        logger.error("Failed to load agent state", 
                    extra_data={'error': str(e), 'filename': filename})
        return None

def process_user_input(agent_state: AgentState, user_input: str) -> None:
    """Process user input and update agent state."""
    logger.info("Processing user input", 
                extra_data={'input_length': len(user_input)})
    
    try:
        # Process the input
        processed_result = process_input(
            user_input=user_input,
            context={i: interaction.model_dump() for i, interaction in 
                    enumerate(agent_state.memory.interaction_history)},
            preferences=agent_state.memory.user_preferences,
            is_clarification_response=agent_state.awaiting_clarification
        )
        
        # Create a decision based on the processed input
        decision = Decision(
            decision_type=DecisionType.RECOMMENDATION 
                if processed_result.success 
                else DecisionType.ERROR,
            action={'response': processed_result.processed_input or ''} if processed_result.success 
                  else {'error': processed_result.error or 'Unknown error'},
            confidence=0.9 if processed_result.success else 0.0,
            reasoning="Successfully processed user input" if processed_result.success 
                     else f"Error: {processed_result.error or 'Unknown error'}"
        )
        
        # Create an interaction record
        interaction = Interaction(
            user_input=user_input,
            processed_input=ProcessedInput(
                processed_input=processed_result.processed_input or '',
                success=processed_result.success,
                error=processed_result.error,
                is_clarification_response=processed_result.is_clarification_response
            ),
            decision=decision,
            action_result={'response': processed_result.processed_input or ''} if processed_result.success 
                         else {'error': processed_result.error or 'Unknown error'},
            timestamp=datetime.now()
        )
        
        # Update agent state
        agent_state.memory.interaction_history.append(interaction)
        agent_state.memory.last_interaction_time = datetime.now()
        agent_state.last_decision = decision
        agent_state.awaiting_clarification = "clarification" in (
            processed_result.processed_input or "").lower()
        
        # Save updated state
        save_agent_state(agent_state)
        
        # Print the response
        if processed_result.success:
            print("\nResponse:", processed_result.processed_input)
        else:
            print("\nError:", processed_result.error or "An unknown error occurred")
            
        logger.info("User input processed successfully")
    except Exception as e:
        logger.error("Failed to process user input", 
                    extra_data={'error': str(e)})
        print(f"\nAn error occurred: {str(e)}")

def main():
    """Main function to run the travel advisor agent."""
    logger.info("Starting travel advisor agent")
    try:
        # Load or create agent state
        agent_state = load_agent_state()
        if agent_state is None:
            agent_state = create_agent()
        
        print("\nWelcome to your AI Travel Advisor!")
        print("Type 'quit' to exit.")
        
        while True:
            user_input = input("\nWhat would you like to know about travel? > ")
            if user_input.lower() == 'quit':
                break
            
            process_user_input(agent_state, user_input)
        
        print("\nThank you for using AI Travel Advisor!")
        logger.info("Travel advisor agent terminated normally")
    except Exception as e:
        logger.error("Travel advisor agent terminated with error", 
                    extra_data={'error': str(e)})
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main() 