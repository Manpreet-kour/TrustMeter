import streamlit as st
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

# Configuration
DATA_DIR = "./data"
CONVERSATIONS_FILE = os.path.join(DATA_DIR, "conversations.ndjson")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="HILT Conversation Recorder",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data models
class ConversationTurn:
    def __init__(self, user_prompt: str, ai_reply: str, timestamp: str, 
                 trust_rating: Optional[int] = None, is_edited: bool = False, 
                 is_accepted: bool = False):
        self.user_prompt = user_prompt
        self.ai_reply = ai_reply
        self.original_ai_reply = ai_reply
        self.timestamp = timestamp
        self.trust_rating = trust_rating
        self.is_edited = is_edited
        self.is_accepted = is_accepted

class Conversation:
    def __init__(self, conversation_id: str, created_at: str):
        self.conversation_id = conversation_id
        self.created_at = created_at
        self.turns: List[ConversationTurn] = []
        self.title = "New Conversation"
    
    def add_turn(self, turn: ConversationTurn):
        self.turns.append(turn)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "created_at": self.created_at,
            "title": self.title,
            "turns": [
                {
                    "user_prompt": turn.user_prompt,
                    "ai_reply": turn.ai_reply,
                    "original_ai_reply": turn.original_ai_reply,
                    "timestamp": turn.timestamp,
                    "trust_rating": turn.trust_rating,
                    "is_edited": turn.is_edited,
                    "is_accepted": turn.is_accepted
                }
                for turn in self.turns
            ]
        }

# Placeholder LLM function
def generate_ai_reply(user_prompt: str) -> str:
    """
    Placeholder LLM function. Replace this with your actual LLM integration.
    
    Example OpenAI integration:
    
    from openai import OpenAI
    import os
    
    def generate_ai_reply(user_prompt: str) -> str:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        return response.choices[0].message.content.strip()
    """
    # Simple placeholder responses based on keywords
    prompt_lower = user_prompt.lower()
    
    if "hello" in prompt_lower or "hi" in prompt_lower:
        return "Hello! How can I help you today?"
    elif "weather" in prompt_lower:
        return "I don't have access to real-time weather data, but I'd be happy to help you find weather information or discuss weather-related topics!"
    elif "help" in prompt_lower:
        return "I'm here to help! What would you like to know or discuss?"
    elif "thank" in prompt_lower:
        return "You're welcome! Is there anything else I can help you with?"
    else:
        return f"I understand you said: '{user_prompt}'. This is a placeholder response. In a real implementation, this would be replaced with an actual LLM call."

# Data persistence functions
def save_conversation(conversation: Conversation):
    """Save a conversation to the NDJSON file"""
    try:
        with open(CONVERSATIONS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(conversation.to_dict(), ensure_ascii=False) + "\n")
    except Exception as e:
        st.error(f"Error saving conversation: {e}")

def load_conversations() -> List[Dict[str, Any]]:
    """Load all conversations from the NDJSON file"""
    conversations = []
    if os.path.exists(CONVERSATIONS_FILE):
        try:
            with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        conversations.append(json.loads(line.strip()))
        except Exception as e:
            st.error(f"Error loading conversations: {e}")
    return conversations

def update_conversation_in_file(conversation: Conversation):
    """Update a specific conversation in the NDJSON file"""
    conversations = load_conversations()
    # Find and update the conversation
    for i, conv in enumerate(conversations):
        if conv["conversation_id"] == conversation.conversation_id:
            conversations[i] = conversation.to_dict()
            break
    
    # Rewrite the entire file
    try:
        with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    except Exception as e:
        st.error(f"Error updating conversation: {e}")

# Initialize session state
if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

def main():
    st.title("💬 HILT Conversation Recorder")
    st.markdown("Record, edit, and manage multi-turn AI conversations with trust ratings")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Choose a page:", ["New Conversation", "Admin View"])
        
        if page == "New Conversation":
            st.markdown("---")
            st.markdown("### Current Session")
            if st.session_state.current_conversation:
                st.info(f"Active: {st.session_state.current_conversation.title}")
                if st.button("Start New Conversation"):
                    st.session_state.current_conversation = None
                    st.rerun()
            else:
                st.info("No active conversation")
    
    if page == "New Conversation":
        conversation_page()
    elif page == "Admin View":
        admin_page()

def conversation_page():
    """Main conversation interface"""
    
    # Initialize or get current conversation
    if st.session_state.current_conversation is None:
        conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state.current_conversation = Conversation(
            conversation_id=conversation_id,
            created_at=datetime.now().isoformat()
        )
    
    conversation = st.session_state.current_conversation
    
    # Display conversation title
    col1, col2 = st.columns([3, 1])
    with col1:
        conversation.title = st.text_input("Conversation Title", value=conversation.title)
    with col2:
        if st.button("💾 Save Conversation"):
            save_conversation(conversation)
            st.success("Conversation saved!")
    
    st.markdown("---")
    
    # Display conversation history
    if conversation.turns:
        st.subheader("Conversation History")
        for i, turn in enumerate(conversation.turns):
            with st.expander(f"Turn {i+1} - {turn.timestamp}"):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("**User:**")
                    st.text_area("", value=turn.user_prompt, height=100, key=f"user_{i}", disabled=True)
                
                with col2:
                    st.markdown("**AI Reply:**")
                    edited_reply = st.text_area("", value=turn.ai_reply, height=100, key=f"ai_{i}")
                    
                    # Check if reply was edited
                    if edited_reply != turn.original_ai_reply:
                        turn.ai_reply = edited_reply
                        turn.is_edited = True
                        st.warning("⚠️ Reply has been edited")
                
                # Trust rating and acceptance controls
                col3, col4, col5 = st.columns([1, 1, 1])
                with col3:
                    trust_rating = st.selectbox(
                        "Trust Rating", 
                        options=[None, 1, 2, 3, 4, 5],
                        index=0 if turn.trust_rating is None else turn.trust_rating,
                        key=f"trust_{i}"
                    )
                    turn.trust_rating = trust_rating
                
                with col4:
                    is_accepted = st.checkbox("Accept", value=turn.is_accepted, key=f"accept_{i}")
                    turn.is_accepted = is_accepted
                
                with col5:
                    if st.button("Update", key=f"update_{i}"):
                        # Update the conversation in the file
                        update_conversation_in_file(conversation)
                        st.success("Turn updated!")
                        st.rerun()
    
    # New message input
    st.markdown("---")
    st.subheader("New Message")
    
    # Use a form to handle the input properly
    with st.form("new_message_form", clear_on_submit=True):
        user_input = st.text_area("Your message:", height=100, key="message_input")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submitted = st.form_submit_button("Send", type="primary")
        
        if submitted:
            if user_input.strip():
                # Generate AI reply
                with st.spinner("Generating AI response..."):
                    ai_reply = generate_ai_reply(user_input)
                
                # Create new turn
                new_turn = ConversationTurn(
                    user_prompt=user_input.strip(),
                    ai_reply=ai_reply,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                
                # Add to conversation
                conversation.add_turn(new_turn)
                st.rerun()
            else:
                st.warning("Please enter a message")

def admin_page():
    """Admin interface for browsing and managing conversations"""
    st.subheader("📊 Admin View - Conversation Management")
    
    # Load all conversations
    conversations = load_conversations()
    
    if not conversations:
        st.info("No conversations found. Start a new conversation to see it here.")
        return
    
    st.metric("Total Conversations", len(conversations))
    
    # Filter and search options
    col1, col2 = st.columns([2, 1])
    with col1:
        search_term = st.text_input("Search conversations", placeholder="Search by title or content...")
    with col2:
        sort_by = st.selectbox("Sort by", ["Date (Newest)", "Date (Oldest)", "Title"])
    
    # Filter conversations
    filtered_conversations = conversations
    if search_term:
        filtered_conversations = [
            conv for conv in conversations
            if search_term.lower() in conv.get("title", "").lower() or
            any(search_term.lower() in turn.get("user_prompt", "").lower() or
                search_term.lower() in turn.get("ai_reply", "").lower()
                for turn in conv.get("turns", []))
        ]
    
    # Sort conversations
    if sort_by == "Date (Newest)":
        filtered_conversations.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    elif sort_by == "Date (Oldest)":
        filtered_conversations.sort(key=lambda x: x.get("created_at", ""))
    elif sort_by == "Title":
        filtered_conversations.sort(key=lambda x: x.get("title", "").lower())
    
    # Display conversations
    for i, conv in enumerate(filtered_conversations):
        with st.expander(f"{conv.get('title', 'Untitled')} - {conv.get('created_at', 'Unknown date')}"):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**ID:** {conv.get('conversation_id', 'Unknown')}")
                st.write(f"**Turns:** {len(conv.get('turns', []))}")
                
                # Show trust ratings summary
                trust_ratings = [turn.get('trust_rating') for turn in conv.get('turns', []) if turn.get('trust_rating')]
                if trust_ratings:
                    avg_trust = sum(trust_ratings) / len(trust_ratings)
                    st.write(f"**Avg Trust Rating:** {avg_trust:.1f}/5")
                
                # Show edited turns count
                edited_turns = sum(1 for turn in conv.get('turns', []) if turn.get('is_edited', False))
                if edited_turns > 0:
                    st.write(f"**Edited Turns:** {edited_turns}")
            
            with col2:
                if st.button("View Details", key=f"view_{i}"):
                    st.session_state.selected_conversation = conv
                    st.rerun()
            
            with col3:
                if st.button("Export JSON", key=f"export_{i}"):
                    # Create download link
                    json_str = json.dumps(conv, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="Download",
                        data=json_str,
                        file_name=f"conversation_{conv.get('conversation_id', 'unknown')}.json",
                        mime="application/json"
                    )
    
    # Detailed view of selected conversation
    if "selected_conversation" in st.session_state:
        st.markdown("---")
        st.subheader("Conversation Details")
        
        conv = st.session_state.selected_conversation
        st.write(f"**Title:** {conv.get('title', 'Untitled')}")
        st.write(f"**Created:** {conv.get('created_at', 'Unknown')}")
        st.write(f"**ID:** {conv.get('conversation_id', 'Unknown')}")
        
        # Display all turns
        for j, turn in enumerate(conv.get('turns', [])):
            st.markdown(f"### Turn {j+1}")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**User:**")
                st.text_area("", value=turn.get('user_prompt', ''), height=100, disabled=True, key=f"admin_user_{j}")
            
            with col2:
                st.markdown("**AI Reply:**")
                st.text_area("", value=turn.get('ai_reply', ''), height=100, disabled=True, key=f"admin_ai_{j}")
            
            # Metadata
            col3, col4, col5 = st.columns([1, 1, 1])
            with col3:
                st.write(f"**Trust Rating:** {turn.get('trust_rating', 'Not rated')}")
            with col4:
                st.write(f"**Edited:** {'Yes' if turn.get('is_edited', False) else 'No'}")
            with col5:
                st.write(f"**Accepted:** {'Yes' if turn.get('is_accepted', False) else 'No'}")
            
            st.write(f"**Timestamp:** {turn.get('timestamp', 'Unknown')}")
            st.markdown("---")
        
        if st.button("Close Details"):
            del st.session_state.selected_conversation
            st.rerun()
    
    # Export all conversations
    st.markdown("---")
    st.subheader("Bulk Export")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Export All as JSON"):
            json_str = json.dumps(conversations, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download All Conversations",
                data=json_str,
                file_name=f"all_conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("Export as CSV"):
            # Convert to CSV format
            csv_data = []
            for conv in conversations:
                for turn in conv.get('turns', []):
                    csv_data.append({
                        'conversation_id': conv.get('conversation_id'),
                        'title': conv.get('title'),
                        'created_at': conv.get('created_at'),
                        'turn_timestamp': turn.get('timestamp'),
                        'user_prompt': turn.get('user_prompt'),
                        'ai_reply': turn.get('ai_reply'),
                        'trust_rating': turn.get('trust_rating'),
                        'is_edited': turn.get('is_edited', False),
                        'is_accepted': turn.get('is_accepted', False)
                    })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_str = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_str,
                    file_name=f"conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()


