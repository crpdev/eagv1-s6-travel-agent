from typing import Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field, validator
from datetime import datetime

class PreferenceQuestion(str, Enum):
    DESTINATION_TYPE = "What type of destinations do you prefer (urban/nature/beach/etc.)?"
    BUDGET = "What's your preferred travel budget (budget/moderate/luxury)?"
    ACTIVITIES = "What activities interest you most while traveling?"
    DIETARY = "Any dietary restrictions or preferences?"
    LOCATION_TYPE = "Do you prefer popular tourist spots or off-the-beaten-path locations?"

class BudgetLevel(str, Enum):
    BUDGET = "budget"
    MODERATE = "moderate"
    LUXURY = "luxury"

class DecisionType(str, Enum):
    RECOMMENDATION = "recommendation"
    CLARIFICATION = "clarification"
    ERROR = "error"
    REFINEMENT = "refinement"

class UserPreferences(BaseModel):
    destination_type: str = Field(..., description="User's preferred type of destinations")
    budget: BudgetLevel = Field(..., description="User's travel budget level")
    activities: str = Field(..., description="Preferred activities while traveling")
    dietary_restrictions: Optional[str] = Field(None, description="Any dietary restrictions")
    location_preference: str = Field(..., description="Preference for tourist spots vs off-beaten-path")

    @validator('budget', pre=True)
    def validate_budget(cls, v):
        if isinstance(v, str):
            v = v.lower()
            if v in [b.value for b in BudgetLevel]:
                return v
            raise ValueError(f"Invalid budget level. Must be one of: {[b.value for b in BudgetLevel]}")
        return v

class Activity(BaseModel):
    name: str = Field(..., description="Name of the activity")
    description: str = Field(..., description="Detailed description of the activity")
    duration: Optional[str] = Field(None, description="Expected duration of the activity")
    considerations: Optional[str] = Field(None, description="Special considerations for the activity")

class TravelRecommendation(BaseModel):
    destination: str = Field(..., description="Name of the recommended destination")
    activities: List[Activity] = Field(..., min_items=1, max_items=5, description="List of recommended activities")
    budget_alignment: bool = Field(..., description="Whether recommendation aligns with budget")
    budget_explanation: str = Field(..., description="Explanation of budget alignment")
    reasoning: str = Field(..., description="Reasoning behind the recommendation")
    timestamp: datetime = Field(default_factory=datetime.now)

class ProcessedInput(BaseModel):
    processed_input: str = Field(..., description="Processed text from the model")
    success: bool = Field(..., description="Whether processing was successful")
    error: Optional[str] = Field(None, description="Error message if processing failed")
    is_clarification_response: bool = Field(..., description="Whether this is a clarification response")

class Decision(BaseModel):
    decision_type: DecisionType
    action: Dict = Field(..., description="Action details based on decision type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the decision")
    reasoning: str = Field(..., description="Reasoning behind the decision")

class Interaction(BaseModel):
    user_input: str = Field(..., description="Original user input")
    processed_input: ProcessedInput = Field(..., description="Processed input details")
    decision: Decision = Field(..., description="Decision made based on input")
    action_result: Dict = Field(..., description="Result of executing the decision")
    timestamp: datetime = Field(default_factory=datetime.now)

class MemoryState(BaseModel):
    user_preferences: UserPreferences = Field(..., description="User's stored preferences")
    interaction_history: List[Interaction] = Field(default_factory=list, description="History of interactions")
    last_interaction_time: datetime = Field(default_factory=datetime.now)

class AgentState(BaseModel):
    memory: MemoryState = Field(..., description="Agent's memory state")
    api_key: str = Field(..., description="API key for the model")
    awaiting_clarification: bool = Field(default=False, description="Whether waiting for clarification")
    last_decision: Optional[Decision] = Field(None, description="Last decision made by the agent")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 