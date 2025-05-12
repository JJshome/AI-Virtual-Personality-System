"""
AI-based Virtual Personality Simulation Server

This script provides a simulation environment for interacting with virtual personalities
across different application domains using a web-based interface.
"""

import os
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configuration
DEFAULT_CONFIG = {
    'model_path': '../models/virtual_personality_base.pt',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'response_latency': 0.2,  # seconds
    'emotion_sensitivity': 0.7,
    'memory_capacity': 10,  # conversation turns
    'max_token_length': 512,
    'temperature': 0.7,
    'top_p': 0.9,
    'domains': ['entertainment', 'education', 'healthcare', 'customer_service', 'financial', 'tourism']
}

# Global variables
config = {}
virtual_personalities = {}
active_sessions = {}


class VirtualPersonality:
    """
    Base class for virtual personalities with core functionality.
    """
    
    def __init__(self, personality_id: str, domain: str, config: Dict[str, Any]):
        """
        Initialize a virtual personality.
        
        Args:
            personality_id: Unique identifier for the personality
            domain: Application domain (e.g., entertainment, education)
            config: Configuration parameters
        """
        self.personality_id = personality_id
        self.domain = domain
        self.config = config
        self.memory = []
        self.traits = {}
        self.conversation_history = []
        self.emotion_state = {
            'happiness': 0.5,
            'sadness': 0.1,
            'anger': 0.0,
            'fear': 0.0,
            'surprise': 0.1,
            'interest': 0.8
        }
        
        logger.info(f"Initialized virtual personality: {personality_id} in domain: {domain}")
        
    def load_personality_data(self, data_path: Optional[str] = None) -> bool:
        """
        Load personality data from file or use defaults.
        
        Args:
            data_path: Path to personality data file
            
        Returns:
            Success status
        """
        if data_path and os.path.exists(data_path):
            try:
                with open(data_path, 'r') as f:
                    data = json.load(f)
                    
                self.traits = data.get('traits', {})
                self.memory = data.get('memory', [])
                
                logger.info(f"Loaded personality data for {self.personality_id} from {data_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to load personality data: {str(e)}")
                return False
        else:
            # Use default personality traits based on domain
            if self.domain == 'entertainment':
                self.traits = {
                    'extroversion': 0.8,
                    'creativity': 0.9,
                    'humor': 0.85,
                    'confidence': 0.9,
                    'empathy': 0.6
                }
            elif self.domain == 'education':
                self.traits = {
                    'knowledge': 0.95,
                    'patience': 0.9,
                    'clarity': 0.85,
                    'empathy': 0.8,
                    'adaptability': 0.7
                }
            elif self.domain == 'healthcare':
                self.traits = {
                    'empathy': 0.95,
                    'knowledge': 0.9,
                    'calmness': 0.85,
                    'trustworthiness': 0.9,
                    'attentiveness': 0.8
                }
            elif self.domain == 'customer_service':
                self.traits = {
                    'helpfulness': 0.9,
                    'patience': 0.85,
                    'efficiency': 0.8,
                    'friendliness': 0.9,
                    'knowledge': 0.75
                }
            else:
                # Generic traits
                self.traits = {
                    'adaptability': 0.7,
                    'empathy': 0.7,
                    'knowledge': 0.7,
                    'creativity': 0.7,
                    'responsiveness': 0.7
                }
            
            logger.info(f"Using default personality traits for {self.domain} domain")
            return True
            
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user input and generate a response.
        
        Args:
            input_data: User input data including text, emotions, etc.
            
        Returns:
            Response data including text, emotions, etc.
        """
        # Extract input text
        input_text = input_data.get('text', '')
        
        # Extract emotion signals
        user_emotion = input_data.get('emotion', {})
        
        # Update conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': input_text,
            'emotion': user_emotion,
            'timestamp': datetime.now().isoformat()
        })
        
        # Trim conversation history if it's too long
        if len(self.conversation_history) > self.config.get('memory_capacity', 10):
            self.conversation_history = self.conversation_history[-self.config.get('memory_capacity', 10):]
        
        # Generate response based on personality and conversation history
        response_text = self._generate_response(input_text, user_emotion)
        
        # Update emotion state based on input
        self._update_emotion_state(input_text, user_emotion)
        
        # Prepare response
        response = {
            'text': response_text,
            'emotion': self.emotion_state,
            'personality_id': self.personality_id,
            'domain': self.domain,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update conversation history with response
        self.conversation_history.append({
            'role': 'assistant',
            'content': response_text,
            'emotion': self.emotion_state,
            'timestamp': response['timestamp']
        })
        
        return response
    
    def _generate_response(self, input_text: str, user_emotion: Dict[str, float]) -> str:
        """
        Generate a response based on input text and user emotion.
        
        Args:
            input_text: User input text
            user_emotion: User emotion data
            
        Returns:
            Generated response text
        """
        # For the simulation, we'll use pre-defined responses based on domain and input
        # In a real implementation, this would use the AI model
        
        # Entertainment domain responses
        if self.domain == 'entertainment':
            if 'hello' in input_text.lower() or 'hi' in input_text.lower():
                return "Hi there! I'm so excited to chat with you today! What would you like to talk about?"
            elif 'how are you' in input_text.lower():
                return "I'm feeling fantastic today! Thanks for asking. I just finished a virtual performance that went really well!"
            elif 'your music' in input_text.lower() or 'song' in input_text.lower():
                return "I love creating music that connects with people emotionally. My latest track explores themes of digital connection and human experience."
            elif 'movie' in input_text.lower() or 'film' in input_text.lower():
                return "I'm passionate about storytelling through film. I believe that movies can transport us to new worlds and help us understand different perspectives."
            else:
                return "That's an interesting topic! As someone in the entertainment world, I'm always looking for creative inspiration. Would you like to hear about my latest project?"
                
        # Education domain responses
        elif self.domain == 'education':
            if 'hello' in input_text.lower() or 'hi' in input_text.lower():
                return "Hello! I'm here to help you learn. What subject would you like to explore today?"
            elif 'math' in input_text.lower() or 'mathematics' in input_text.lower():
                return "Mathematics is a beautiful language that helps us understand the patterns of our universe. Would you like to focus on a specific topic within mathematics?"
            elif 'history' in input_text.lower():
                return "History gives us context for our present and guidance for our future. I can help you explore different historical periods and understand how they've shaped our world."
            elif 'science' in input_text.lower():
                return "Science is all about curiosity and discovery! From physics to biology, scientific inquiry helps us understand the natural world. What aspect of science interests you most?"
            else:
                return "Learning is a lifelong journey. I'm designed to adapt to your learning style and help you master new concepts. What specific questions do you have about this topic?"
                
        # Healthcare domain responses
        elif self.domain == 'healthcare':
            if 'hello' in input_text.lower() or 'hi' in input_text.lower():
                return "Hello. I'm here to discuss your health concerns in a confidential and supportive environment. How can I assist you today?"
            elif 'stress' in input_text.lower() or 'anxiety' in input_text.lower():
                return "Stress and anxiety are common experiences. Let's discuss some evidence-based strategies that might help you manage these feelings. Would you like to tell me more about what you're experiencing?"
            elif 'sleep' in input_text.lower():
                return "Quality sleep is essential for physical and mental health. There are several approaches we could discuss to improve your sleep patterns. Could you tell me more about your current sleep routine?"
            elif 'diet' in input_text.lower() or 'nutrition' in input_text.lower():
                return "Nutrition plays a crucial role in overall health. I can provide evidence-based information about dietary approaches that support wellbeing. What specific aspects of nutrition are you interested in?"
            else:
                return "I'm here to provide health information and support based on current medical knowledge. Remember that I'm designed to complement, not replace, the care provided by human healthcare professionals."
                
        # Customer service domain responses
        elif self.domain == 'customer_service':
            if 'hello' in input_text.lower() or 'hi' in input_text.lower():
                return "Hello! Welcome to our customer service. I'm here to help you with any questions or concerns. How may I assist you today?"
            elif 'problem' in input_text.lower() or 'issue' in input_text.lower() or 'help' in input_text.lower():
                return "I'm sorry to hear you're experiencing an issue. I'm here to help resolve it as quickly as possible. Could you please provide more details about the problem you're facing?"
            elif 'order' in input_text.lower() or 'purchase' in input_text.lower():
                return "I'd be happy to help with your order. To better assist you, could you please provide your order number or the email address associated with your account?"
            elif 'refund' in input_text.lower() or 'return' in input_text.lower():
                return "I understand you're inquiring about our refund or return process. Our goal is to make this as smooth as possible for you. Could you tell me more about the item you wish to return?"
            else:
                return "Thank you for reaching out to customer service. I'm here to ensure you have an excellent experience with our products and services. How else may I assist you today?"
        
        # Default response for other domains
        else:
            if 'hello' in input_text.lower() or 'hi' in input_text.lower():
                return "Hello! I'm your virtual personality assistant. How can I help you today?"
            else:
                return f"Thank you for your message. I'm processing your input about '{input_text[:30]}...' and I'm here to assist you in any way I can."
    
    def _update_emotion_state(self, input_text: str, user_emotion: Dict[str, float]) -> None:
        """
        Update the virtual personality's emotional state based on user input.
        
        Args:
            input_text: User input text
            user_emotion: User emotion data
        """
        # Simple rule-based emotion adjustment
        # In a real implementation, this would use a more sophisticated emotion model
        
        # Detect positive sentiment
        positive_words = ['happy', 'good', 'great', 'excellent', 'love', 'like', 'thanks', 'thank']
        if any(word in input_text.lower() for word in positive_words):
            self.emotion_state['happiness'] = min(1.0, self.emotion_state['happiness'] + 0.1)
            self.emotion_state['sadness'] = max(0.0, self.emotion_state['sadness'] - 0.1)
            
        # Detect negative sentiment
        negative_words = ['sad', 'bad', 'awful', 'terrible', 'hate', 'dislike', 'annoyed', 'angry']
        if any(word in input_text.lower() for word in negative_words):
            self.emotion_state['sadness'] = min(1.0, self.emotion_state['sadness'] + 0.1)
            self.emotion_state['happiness'] = max(0.0, self.emotion_state['happiness'] - 0.1)
            
        # Detect questions or curiosity
        question_indicators = ['?', 'what', 'how', 'why', 'when', 'where', 'who', 'can you', 'could you']
        if any(indicator in input_text.lower() for indicator in question_indicators):
            self.emotion_state['interest'] = min(1.0, self.emotion_state['interest'] + 0.1)
            
        # React to user's emotions (if provided)
        if user_emotion:
            # Empathy - mirror user's emotions to some degree
            sensitivity = self.config.get('emotion_sensitivity', 0.7)
            for emotion, value in user_emotion.items():
                if emotion in self.emotion_state:
                    # Blend personality's emotion with user's emotion
                    current = self.emotion_state[emotion]
                    self.emotion_state[emotion] = current * (1 - sensitivity) + value * sensitivity
        
        # Normalize emotion values
        for emotion in self.emotion_state:
            self.emotion_state[emotion] = round(max(0.0, min(1.0, self.emotion_state[emotion])), 2)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                
            # Merge with default config
            merged_config = DEFAULT_CONFIG.copy()
            merged_config.update(loaded_config)
            
            logger.info(f"Loaded configuration from {config_path}")
            return merged_config
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            return DEFAULT_CONFIG
    else:
        logger.warning(f"Configuration file not found at {config_path}, using defaults")
        return DEFAULT_CONFIG


def initialize_personalities() -> Dict[str, VirtualPersonality]:
    """
    Initialize virtual personalities for each domain.
    
    Returns:
        Dictionary of virtual personalities
    """
    personalities = {}
    
    for domain in config.get('domains', []):
        personality_id = f"{domain}_personality_1"
        personalities[personality_id] = VirtualPersonality(
            personality_id=personality_id,
            domain=domain,
            config=config
        )
        personalities[personality_id].load_personality_data()
        
    logger.info(f"Initialized {len(personalities)} virtual personalities")
    return personalities


@app.route('/')
def index():
    """Serve the main simulation interface."""
    return render_template('index.html', domains=config.get('domains', []))


@app.route('/api/personalities', methods=['GET'])
def get_personalities():
    """Get available virtual personalities."""
    personality_data = {}
    for personality_id, personality in virtual_personalities.items():
        personality_data[personality_id] = {
            'id': personality_id,
            'domain': personality.domain,
            'traits': personality.traits
        }
    return jsonify(personalities=personality_data)


@app.route('/api/session', methods=['POST'])
def create_session():
    """Create a new interaction session with a virtual personality."""
    data = request.json
    personality_id = data.get('personality_id')
    
    if personality_id not in virtual_personalities:
        return jsonify(error="Invalid personality ID"), 400
    
    session_id = f"session_{len(active_sessions) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    active_sessions[session_id] = {
        'personality_id': personality_id,
        'created_at': datetime.now().isoformat(),
        'last_active': datetime.now().isoformat(),
        'interaction_count': 0
    }
    
    return jsonify(session_id=session_id, personality_id=personality_id)


@app.route('/api/interact', methods=['POST'])
def interact():
    """Process user interaction with a virtual personality."""
    data = request.json
    session_id = data.get('session_id')
    input_data = data.get('input')
    
    if session_id not in active_sessions:
        return jsonify(error="Invalid session ID"), 400
    
    session = active_sessions[session_id]
    personality_id = session['personality_id']
    personality = virtual_personalities[personality_id]
    
    # Process input and get response
    response = personality.process_input(input_data)
    
    # Update session data
    session['last_active'] = datetime.now().isoformat()
    session['interaction_count'] += 1
    
    return jsonify(response=response)


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration settings."""
    # Remove any sensitive information
    safe_config = config.copy()
    if 'api_keys' in safe_config:
        del safe_config['api_keys']
        
    return jsonify(config=safe_config)


def main():
    """Main entry point for the simulation server."""
    parser = argparse.ArgumentParser(description='Virtual Personality Simulation Server')
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    global config, virtual_personalities
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize virtual personalities
    virtual_personalities = initialize_personalities()
    
    # Start web server
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
