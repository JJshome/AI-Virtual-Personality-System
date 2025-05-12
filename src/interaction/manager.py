"""
Interaction Management Module for the AI-based Virtual Personality System.
This module handles the communication between users and virtual personalities,
managing context, responses, and multimodal interactions.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
import threading

logger = logging.getLogger(__name__)


class InteractionManager:
    """
    Manages interactions between users and virtual personalities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the interaction manager.
        
        Args:
            config: Configuration dictionary for interaction management
        """
        self.config = config
        self.response_time_ms = config.get('response_time_ms', 100)
        self.multimodal_fusion = config.get('multimodal_fusion', True)
        self.emotion_detection = config.get('emotion_detection', True)
        self.contextual_memory_turns = config.get('contextual_memory_turns', 10)
        
        # Initialize modalities
        self.modalities = {
            'text': config.get('modalities', {}).get('text', True),
            'voice': config.get('modalities', {}).get('voice', True),
            'vision': config.get('modalities', {}).get('vision', True),
            'gesture': config.get('modalities', {}).get('gesture', False)
        }
        
        # Initialize active sessions
        self.active_sessions = {}
        self.session_lock = threading.Lock()
        
        # Initialize dependencies
        self.personality_generator = None
        self.ethics_manager = None
        
        logger.info(f"Initialized InteractionManager with response time: {self.response_time_ms}ms")
        logger.info(f"Multimodal fusion: {self.multimodal_fusion}, Emotion detection: {self.emotion_detection}")
        logger.info(f"Active modalities: {self.modalities}")
    
    def set_personality_generator(self, personality_generator: Any) -> None:
        """
        Set the personality generator dependency.
        
        Args:
            personality_generator: PersonalityGenerator instance
        """
        self.personality_generator = personality_generator
        logger.info("PersonalityGenerator dependency set for InteractionManager")
    
    def set_ethics_manager(self, ethics_manager: Any) -> None:
        """
        Set the ethics manager dependency.
        
        Args:
            ethics_manager: EthicsSecurityManager instance
        """
        self.ethics_manager = ethics_manager
        logger.info("EthicsManager dependency set for InteractionManager")
    
    def create_session(self, user_id: str, personality_id: str, initial_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new interaction session.
        
        Args:
            user_id: Unique identifier for the user
            personality_id: Unique identifier for the personality
            initial_context: Optional initial context for the session
            
        Returns:
            Session ID for the new session
        """
        # Load the personality
        if not self.personality_generator:
            raise ValueError("PersonalityGenerator dependency not set")
        
        personality = self.personality_generator.load_personality(personality_id)
        if not personality:
            raise ValueError(f"Personality {personality_id} not found")
        
        # Create a new session ID
        session_id = f"{user_id}_{personality_id}_{int(time.time())}"
        
        # Initialize session data
        session_data = {
            'id': session_id,
            'user_id': user_id,
            'personality_id': personality_id,
            'created_at': time.time(),
            'last_active': time.time(),
            'context': initial_context or {},
            'history': [],
            'memory': [],
            'current_state': {
                'user_emotion': None,
                'personality_emotion': self._get_default_emotion(personality),
                'attention_focus': None
            }
        }
        
        # Add session to active sessions
        with self.session_lock:
            self.active_sessions[session_id] = session_data
        
        logger.info(f"Created new session {session_id} for user {user_id} with personality {personality_id}")
        
        return session_id
    
    def _get_default_emotion(self, personality: Dict[str, Any]) -> Dict[str, float]:
        """
        Get default emotion state for a personality.
        
        Args:
            personality: Personality data
            
        Returns:
            Default emotion state
        """
        # Use personality's emotional patterns if available
        if 'behavior' in personality and 'emotional_patterns' in personality['behavior']:
            return personality['behavior']['emotional_patterns']
        
        # Otherwise use a generic default
        return {
            'neutral': 0.7,
            'happy': 0.2,
            'sad': 0.0,
            'angry': 0.0,
            'surprised': 0.0,
            'excited': 0.1
        }
    
    def process_input(self, session_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user input and generate a response.
        
        Args:
            session_id: Session ID
            input_data: Input data (text, audio, video, etc.)
            
        Returns:
            Response data
        """
        # Check if session exists
        with self.session_lock:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
        
        # Load the personality
        if not self.personality_generator:
            raise ValueError("PersonalityGenerator dependency not set")
        
        personality_id = session['personality_id']
        personality = self.personality_generator.load_personality(personality_id)
        if not personality:
            raise ValueError(f"Personality {personality_id} not found")
        
        # Update last active timestamp
        session['last_active'] = time.time()
        
        # Extract input modalities
        text = input_data.get('text', '')
        audio = input_data.get('audio')
        video = input_data.get('video')
        gesture = input_data.get('gesture')
        
        # Process multimodal input
        processed_input = self._process_multimodal_input(
            text=text,
            audio=audio,
            video=video,
            gesture=gesture,
            personality=personality,
            session=session
        )
        
        # Add to history
        session['history'].append({
            'role': 'user',
            'content': processed_input,
            'timestamp': time.time()
        })
        
        # Trim history if too long
        if len(session['history']) > self.contextual_memory_turns * 2:
            session['history'] = session['history'][-self.contextual_memory_turns * 2:]
        
        # Generate response
        response = self._generate_response(processed_input, personality, session)
        
        # Apply ethics checks if available
        if self.ethics_manager:
            response = self.ethics_manager.validate_response(response, session, personality)
        
        # Add response to history
        session['history'].append({
            'role': 'assistant',
            'content': response,
            'timestamp': time.time()
        })
        
        # Update session in active sessions
        with self.session_lock:
            self.active_sessions[session_id] = session
        
        return response
    
    def _process_multimodal_input(self, text: str, audio: Optional[Dict[str, Any]] = None,
                                video: Optional[Dict[str, Any]] = None, gesture: Optional[Dict[str, Any]] = None,
                                personality: Dict[str, Any] = None, session: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process multimodal input data.
        
        Args:
            text: Text input
            audio: Audio input data
            video: Video input data
            gesture: Gesture input data
            personality: Personality data
            session: Session data
            
        Returns:
            Processed input data
        """
        processed_input = {
            'text': text,
            'detected_intent': None,
            'detected_emotion': None,
            'detected_context': None,
            'timestamp': time.time()
        }
        
        # Process text for intent detection
        if text:
            intent = self._detect_intent(text, personality, session)
            processed_input['detected_intent'] = intent
        
        # Process audio for emotion detection
        user_emotion = None
        if audio and self.modalities['voice'] and self.emotion_detection:
            audio_emotion = self._detect_audio_emotion(audio)
            user_emotion = audio_emotion
        
        # Process video for emotion detection
        if video and self.modalities['vision'] and self.emotion_detection:
            video_emotion = self._detect_video_emotion(video)
            
            # If we already have audio emotion, combine them
            if user_emotion:
                user_emotion = self._combine_emotions(user_emotion, video_emotion)
            else:
                user_emotion = video_emotion
        
        # Set detected emotion
        if user_emotion:
            processed_input['detected_emotion'] = user_emotion
            
            # Update session's current user emotion
            if session:
                session['current_state']['user_emotion'] = user_emotion
        
        # Process gesture if available
        if gesture and self.modalities['gesture']:
            gesture_meaning = self._interpret_gesture(gesture)
            processed_input['detected_gesture'] = gesture_meaning
        
        # Detect context from all modalities
        context = self._detect_context(text, audio, video, gesture, session)
        processed_input['detected_context'] = context
        
        return processed_input
    
    def _detect_intent(self, text: str, personality: Dict[str, Any], session: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect user intent from text.
        
        Args:
            text: Text input
            personality: Personality data
            session: Session data
            
        Returns:
            Detected intent
        """
        # In a real implementation, this would use sophisticated NLP
        # For now, we'll use a simplified approach
        
        intent = {
            'type': 'unknown',
            'confidence': 0.0,
            'parameters': {}
        }
        
        text_lower = text.lower()
        
        # Check for greeting
        if any(greeting in text_lower for greeting in ['hello', 'hi', 'hey', 'greetings']):
            intent['type'] = 'greeting'
            intent['confidence'] = 0.9
        
        # Check for question
        elif '?' in text or any(question_word in text_lower.split() for question_word in ['what', 'when', 'where', 'who', 'why', 'how']):
            intent['type'] = 'question'
            intent['confidence'] = 0.8
            
            # Try to determine question topic
            if 'you' in text_lower.split():
                intent['parameters']['topic'] = 'personality'
            elif any(word in text_lower for word in ['do', 'think', 'feel', 'like']):
                intent['parameters']['topic'] = 'opinion'
            else:
                intent['parameters']['topic'] = 'information'
        
        # Check for statement
        elif len(text.split()) > 3:
            intent['type'] = 'statement'
            intent['confidence'] = 0.7
        
        # Check for farewell
        elif any(farewell in text_lower for farewell in ['bye', 'goodbye', 'see you', 'farewell']):
            intent['type'] = 'farewell'
            intent['confidence'] = 0.9
        
        return intent
    
    def _detect_audio_emotion(self, audio: Dict[str, Any]) -> Dict[str, float]:
        """
        Detect emotion from audio data.
        
        Args:
            audio: Audio data
            
        Returns:
            Detected emotion
        """
        # In a real implementation, this would analyze audio features
        # For now, we'll simulate detection with random values
        
        # Simple emotion detection based on simulated pitch and volume
        pitch = audio.get('pitch', np.random.uniform(0.3, 0.7))
        volume = audio.get('volume', np.random.uniform(0.2, 0.8))
        speaking_rate = audio.get('speaking_rate', np.random.uniform(0.3, 0.7))
        
        emotions = {
            'neutral': 0.5,
            'happy': 0.0,
            'sad': 0.0,
            'angry': 0.0,
            'surprised': 0.0,
            'excited': 0.0
        }
        
        # High pitch, high volume, high rate -> excited/happy
        if pitch > 0.6 and volume > 0.6 and speaking_rate > 0.6:
            emotions['neutral'] = 0.1
            emotions['happy'] = 0.4
            emotions['excited'] = 0.5
        
        # Low pitch, low volume, low rate -> sad
        elif pitch < 0.4 and volume < 0.4 and speaking_rate < 0.4:
            emotions['neutral'] = 0.2
            emotions['sad'] = 0.8
        
        # High volume, medium/high pitch, high rate -> angry
        elif volume > 0.7 and pitch > 0.4 and speaking_rate > 0.6:
            emotions['neutral'] = 0.1
            emotions['angry'] = 0.9
        
        # High pitch spike, medium volume -> surprised
        elif pitch > 0.8 and 0.4 < volume < 0.7:
            emotions['neutral'] = 0.2
            emotions['surprised'] = 0.8
        
        return emotions
    
    def _detect_video_emotion(self, video: Dict[str, Any]) -> Dict[str, float]:
        """
        Detect emotion from video data.
        
        Args:
            video: Video data
            
        Returns:
            Detected emotion
        """
        # In a real implementation, this would analyze facial expressions
        # For now, we'll use provided expression or simulate detection
        
        # If expression is directly provided, use it
        if 'expression' in video:
            expression = video['expression']
            
            emotions = {
                'neutral': 0.0,
                'happy': 0.0,
                'sad': 0.0,
                'angry': 0.0,
                'surprised': 0.0,
                'excited': 0.0
            }
            
            # Set confidence for the detected expression
            if expression in emotions:
                emotions[expression] = 0.9
                emotions['neutral'] = 0.1
            else:
                emotions['neutral'] = 1.0
            
            return emotions
        
        # Otherwise simulate detection from facial features
        else:
            # Simulate facial landmarks analysis
            emotions = {
                'neutral': 0.6,
                'happy': 0.0,
                'sad': 0.0,
                'angry': 0.0,
                'surprised': 0.0,
                'excited': 0.0
            }
            
            # Randomly select a dominant emotion for simulation
            dominant_emotion = np.random.choice(
                ['neutral', 'happy', 'sad', 'angry', 'surprised', 'excited'],
                p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]
            )
            
            # Set dominant emotion
            emotions['neutral'] = 0.2
            emotions[dominant_emotion] = 0.8
            
            return emotions
    
    def _combine_emotions(self, emotion1: Dict[str, float], emotion2: Dict[str, float]) -> Dict[str, float]:
        """
        Combine emotions from different modalities.
        
        Args:
            emotion1: First emotion set
            emotion2: Second emotion set
            
        Returns:
            Combined emotion
        """
        # Weight for each modality (can be adjusted based on confidence)
        weight1 = 0.5
        weight2 = 0.5
        
        combined = {}
        
        # Combine all emotions
        all_emotions = set(list(emotion1.keys()) + list(emotion2.keys()))
        
        for emotion in all_emotions:
            value1 = emotion1.get(emotion, 0.0)
            value2 = emotion2.get(emotion, 0.0)
            
            # Weighted average
            combined[emotion] = value1 * weight1 + value2 * weight2
        
        return combined
    
    def _interpret_gesture(self, gesture: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpret meaning from gesture data.
        
        Args:
            gesture: Gesture data
            
        Returns:
            Interpreted gesture meaning
        """
        # In a real implementation, this would analyze gesture features
        # For now, we'll simulate interpretation
        
        gesture_type = gesture.get('type', 'unknown')
        
        if gesture_type == 'hand_wave':
            return {
                'meaning': 'greeting',
                'confidence': 0.9
            }
        elif gesture_type == 'nod':
            return {
                'meaning': 'agreement',
                'confidence': 0.8
            }
        elif gesture_type == 'head_shake':
            return {
                'meaning': 'disagreement',
                'confidence': 0.8
            }
        elif gesture_type == 'point':
            return {
                'meaning': 'attention_direction',
                'confidence': 0.7,
                'parameters': {
                    'direction': gesture.get('direction', 'forward')
                }
            }
        else:
            return {
                'meaning': 'unknown',
                'confidence': 0.1
            }
    
    def _detect_context(self, text: str, audio: Optional[Dict[str, Any]] = None,
                        video: Optional[Dict[str, Any]] = None, gesture: Optional[Dict[str, Any]] = None,
                        session: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detect context from multimodal input.
        
        Args:
            text: Text input
            audio: Audio input data
            video: Video input data
            gesture: Gesture input data
            session: Session data
            
        Returns:
            Detected context
        """
        context = {
            'environment': 'unknown',
            'user_attention': 1.0,  # Default full attention
            'topic': None,
            'references': {}
        }
        
        # Detect environment from audio if available
        if audio and 'background_noise' in audio:
            noise_type = audio.get('background_noise', {}).get('type', 'quiet')
            
            if noise_type == 'quiet':
                context['environment'] = 'quiet_room'
            elif noise_type == 'conversation':
                context['environment'] = 'social_gathering'
            elif noise_type == 'traffic':
                context['environment'] = 'outdoors_urban'
            elif noise_type == 'nature':
                context['environment'] = 'outdoors_nature'
            elif noise_type == 'music':
                context['environment'] = 'entertainment_venue'
        
        # Detect user attention from video if available
        if video:
            # Lower attention if user is looking away
            if 'gaze_direction' in video:
                gaze = video['gaze_direction']
                # If gaze is not toward the camera/personality
                if isinstance(gaze, list) and len(gaze) >= 2:
                    # Simplified attention calculation
                    gaze_x, gaze_y = gaze[0], gaze[1]
                    distance_from_center = np.sqrt(gaze_x**2 + gaze_y**2)
                    if distance_from_center > 0.3:  # If looking away from center
                        context['user_attention'] = max(0.2, 1.0 - distance_from_center)
            
            # Direct attention score if provided
            if 'attention_score' in video:
                context['user_attention'] = video['attention_score']
        
        # Extract topic from text
        if text:
            # Use a simple keyword approach for topic detection
            topics = {
                'personal': ['you', 'your', 'yourself', 'personality', 'feel', 'think'],
                'technical': ['how', 'system', 'works', 'technology', 'function', 'process'],
                'social': ['people', 'society', 'world', 'everyone', 'community'],
                'entertainment': ['movie', 'music', 'game', 'play', 'fun', 'enjoy'],
                'education': ['learn', 'study', 'teach', 'education', 'school', 'knowledge']
            }
            
            text_lower = text.lower()
            topic_scores = {}
            
            for topic, keywords in topics.items():
                score = sum(1 for keyword in keywords if keyword in text_lower.split())
                if score > 0:
                    topic_scores[topic] = score
            
            if topic_scores:
                # Get topic with highest score
                context['topic'] = max(topic_scores.items(), key=lambda x: x[1])[0]
        
        # Extract references from all modalities
        references = self._extract_references(text, video, gesture, session)
        if references:
            context['references'] = references
        
        return context
    
    def _extract_references(self, text: str, video: Optional[Dict[str, Any]] = None,
                          gesture: Optional[Dict[str, Any]] = None, session: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract references to entities, objects, or concepts.
        
        Args:
            text: Text input
            video: Video input data
            gesture: Gesture input data
            session: Session data
            
        Returns:
            Extracted references
        """
        references = {}
        
        # Extract text references (pronouns, names, etc.)
        if text:
            text_lower = text.lower()
            
            # Check for self-references
            if any(word in text_lower for word in ['i', 'me', 'my', 'mine', 'myself']):
                references['self'] = True
            
            # Check for references to the virtual personality
            if any(word in text_lower for word in ['you', 'your', 'yours', 'yourself']):
                references['virtual_personality'] = True
            
            # Check for third-person references (simplified)
            if any(word in text_lower for word in ['he', 'she', 'they', 'them', 'their']):
                references['third_person'] = True
        
        # Extract visual references if pointing or gaze is detected
        if video and 'gaze_direction' in video:
            gaze = video['gaze_direction']
            # If gaze is clearly directed at something
            if isinstance(gaze, list) and len(gaze) >= 2:
                gaze_x, gaze_y = gaze[0], gaze[1]
                if abs(gaze_x) > 0.5 or abs(gaze_y) > 0.5:  # Looking strongly in a direction
                    references['visual_attention'] = {
                        'type': 'gaze',
                        'direction': {
                            'x': gaze_x,
                            'y': gaze_y
                        }
                    }
        
        # Extract gesture references
        if gesture and gesture.get('type') == 'point':
            direction = gesture.get('direction', {})
            references['visual_attention'] = {
                'type': 'pointing',
                'direction': direction
            }
        
        return references
    
    def _generate_response(self, input_data: Dict[str, Any], personality: Dict[str, Any], session: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a response based on input, personality, and session context.
        
        Args:
            input_data: Processed input data
            personality: Personality data
            session: Session data
            
        Returns:
            Generated response
        """
        # Extract input components
        text = input_data.get('text', '')
        intent = input_data.get('detected_intent', {})
        emotion = input_data.get('detected_emotion', {})
        context = input_data.get('detected_context', {})
        
        # Initialize response
        response = {
            'text': '',
            'emotion': session['current_state']['personality_emotion'],
            'confidence': 0.0,
            'timestamp': time.time()
        }
        
        # Adjust response based on personality domain
        domain = personality.get('domain', 'general')
        
        # Adjust emotion based on user's emotion
        if emotion:
            response['emotion'] = self._adjust_emotion(response['emotion'], emotion, personality)
            
            # Update session's personality emotion
            session['current_state']['personality_emotion'] = response['emotion']
        
        # Generate text response based on intent, domain, and personality
        if intent:
            intent_type = intent.get('type', 'unknown')
            
            if intent_type == 'greeting':
                response['text'] = self._generate_greeting(personality, session)
                response['confidence'] = 0.9
            
            elif intent_type == 'question':
                question_topic = intent.get('parameters', {}).get('topic')
                response['text'] = self._generate_answer(text, question_topic, personality, session)
                response['confidence'] = 0.8
            
            elif intent_type == 'statement':
                response['text'] = self._generate_statement_response(text, personality, session)
                response['confidence'] = 0.7
            
            elif intent_type == 'farewell':
                response['text'] = self._generate_farewell(personality, session)
                response['confidence'] = 0.9
            
            else:
                response['text'] = self._generate_general_response(text, personality, session)
                response['confidence'] = 0.5
        else:
            response['text'] = self._generate_general_response(text, personality, session)
            response['confidence'] = 0.5
        
        # Add modifiers based on user attention
        user_attention = context.get('user_attention', 1.0)
        if user_attention < 0.5:
            # If user seems distracted, make response more engaging
            response['text'] = self._add_attention_modifier(response['text'], user_attention, personality)
        
        # Add context-specific information if needed
        environment = context.get('environment')
        if environment and environment != 'unknown':
            response = self._add_environment_context(response, environment, personality)
        
        return response
    
    def _adjust_emotion(self, current_emotion: Dict[str, float], user_emotion: Dict[str, float], 
                       personality: Dict[str, Any]) -> Dict[str, float]:
        """
        Adjust personality emotion based on user emotion and personality traits.
        
        Args:
            current_emotion: Current emotion state
            user_emotion: Detected user emotion
            personality: Personality data
            
        Returns:
            Adjusted emotion state
        """
        # Copy current emotion to avoid modifying the original
        new_emotion = current_emotion.copy()
        
        # Get personality traits
        traits = personality.get('traits', {})
        
        # Extract relevant traits with defaults
        empathy = traits.get('empathy', 0.5)
        emotional_stability = traits.get('neuroticism', 0.5)
        emotional_stability = 1.0 - emotional_stability  # Inverse of neuroticism
        extroversion = traits.get('extroversion', 0.5)
        
        # Calculate influence factor based on empathy
        # Higher empathy = more influenced by user's emotion
        influence_factor = empathy * 0.5
        
        # Calculate stability factor
        # Higher emotional stability = less emotional variance
        stability_factor = emotional_stability * 0.8
        
        # Apply influence from user emotion
        for emotion, value in user_emotion.items():
            if emotion in new_emotion:
                # Calculate new value
                # More empathetic personalities mirror user emotions more
                current_value = new_emotion[emotion]
                influenced_value = current_value * (1 - influence_factor) + value * influence_factor
                
                # Apply stability damping
                change = influenced_value - current_value
                damped_change = change * (1 - stability_factor)
                new_value = current_value + damped_change
                
                # Update emotion
                new_emotion[emotion] = max(0.0, min(1.0, new_value))
        
        # Special case for extroversion: higher extroversion results in more happiness/excitement
        # and less neutral emotion
        if extroversion > 0.6:
            new_emotion['happy'] = max(new_emotion['happy'], extroversion * 0.3)
            new_emotion['excited'] = max(new_emotion['excited'], extroversion * 0.2)
            new_emotion['neutral'] = min(new_emotion['neutral'], 1.0 - extroversion * 0.3)
        
        # Normalize emotions to ensure they sum to approximately 1.0
        total = sum(new_emotion.values())
        if total > 0:
            for emotion in new_emotion:
                new_emotion[emotion] /= total
        
        return new_emotion
    
    def _generate_greeting(self, personality: Dict[str, Any], session: Dict[str, Any]) -> str:
        """
        Generate a greeting response.
        
        Args:
            personality: Personality data
            session: Session data
            
        Returns:
            Greeting text
        """
        # Check if this is the first interaction
        is_first_interaction = len(session['history']) == 0
        
        # Get personality traits and domain
        traits = personality.get('traits', {})
        domain = personality.get('domain', 'general')
        
        # Get response style from behavior
        response_style = personality.get('behavior', {}).get('response_style', {})
        tone = response_style.get('tone', 'balanced')
        verbosity = response_style.get('verbosity', 'medium')
        formality = response_style.get('formality', 'neutral')
        
        # Base greetings by formality
        formal_greetings = ["Hello", "Greetings", "Good day"]
        neutral_greetings = ["Hi there", "Hello", "Hi"]
        casual_greetings = ["Hey", "Hi there", "Hey there", "What's up"]
        
        if formality == 'formal':
            base = np.random.choice(formal_greetings)
        elif formality == 'casual':
            base = np.random.choice(casual_greetings)
        else:
            base = np.random.choice(neutral_greetings)
        
        # Add user reference for personalization
        user_id = session['user_id']
        user_name = user_id.split('_')[0] if '_' in user_id else user_id
        
        # Add name if formality is not casual
        if formality != 'casual':
            greeting = f"{base}, {user_name}"
        else:
            greeting = base
        
        # Additional content based on domain and verbosity
        additional = ""
        
        if is_first_interaction:
            if domain == 'entertainment':
                additional = "I'm excited to chat with you!"
            elif domain == 'education':
                additional = "I'm here to help you learn."
            elif domain == 'healthcare':
                additional = "I'm here to assist with your health questions."
            elif domain == 'customer_service':
                additional = "How can I assist you today?"
            elif domain == 'financial':
                additional = "I'm here to help with your financial questions."
            elif domain == 'tourism':
                additional = "Ready to explore and discover new places?"
            else:
                additional = "How can I help you today?"
        else:
            # Not first interaction
            if domain == 'entertainment':
                additional = "Great to see you again!"
            elif domain == 'education':
                additional = "Ready to continue learning?"
            elif domain == 'healthcare':
                additional = "How have you been feeling?"
            elif domain == 'customer_service':
                additional = "What can I help you with today?"
            elif domain == 'financial':
                additional = "How can I assist with your finances today?"
            elif domain == 'tourism':
                additional = "Ready for more travel inspiration?"
            else:
                additional = "How can I help you today?"
        
        # Combine based on verbosity
        if verbosity == 'low':
            return greeting
        elif verbosity == 'medium':
            return f"{greeting}. {additional}"
        else:  # high verbosity
            # Add even more personality-specific content
            extra = ""
            if 'extroversion' in traits and traits['extroversion'] > 0.7:
                extra = " I'm really looking forward to our conversation!"
            elif 'enthusiasm' in traits and traits['enthusiasm'] > 0.7:
                extra = " I'm enthusiastic about helping you today!"
            
            return f"{greeting}. {additional}{extra}"
    
    def _generate_answer(self, question: str, topic: Optional[str], personality: Dict[str, Any], session: Dict[str, Any]) -> str:
        """
        Generate an answer to a question.
        
        Args:
            question: Question text
            topic: Question topic if detected
            personality: Personality data
            session: Session data
            
        Returns:
            Answer text
        """
        # Get personality traits and domain
        traits = personality.get('traits', {})
        domain = personality.get('domain', 'general')
        knowledge = personality.get('knowledge', {})
        
        # Get response style from behavior
        response_style = personality.get('behavior', {}).get('response_style', {})
        tone = response_style.get('tone', 'balanced')
        verbosity = response_style.get('verbosity', 'medium')
        formality = response_style.get('formality', 'neutral')
        
        # Generate answer based on topic and domain
        answer = ""
        
        # Questions about the personality itself
        if topic == 'personality':
            if 'who are you' in question.lower() or 'what are you' in question.lower():
                if domain == 'entertainment':
                    answer = f"I'm a virtual personality designed to entertain and engage with fans."
                elif domain == 'education':
                    answer = f"I'm a virtual teacher designed to help people learn effectively."
                elif domain == 'healthcare':
                    answer = f"I'm a virtual healthcare assistant designed to provide health information and support."
                elif domain == 'customer_service':
                    answer = f"I'm a virtual assistant designed to provide helpful customer service."
                elif domain == 'financial':
                    answer = f"I'm a virtual financial advisor designed to help with financial questions and planning."
                elif domain == 'tourism':
                    answer = f"I'm a virtual travel guide designed to help you discover and plan amazing trips."
                else:
                    answer = f"I'm a virtual personality designed to interact with people in a natural way."
            
            elif 'how do you work' in question.lower() or 'how were you made' in question.lower():
                answer = "I was created using advanced AI technology that combines natural language processing, emotion recognition, and personality modeling to create a unique virtual personality."
            
            else:
                # Generic personality questions
                answer = "I'm designed to interact with people in a natural and engaging way, adapting to different contexts and needs."
        
        # Questions about opinions
        elif topic == 'opinion':
            # Extract what the opinion is about
            question_lower = question.lower()
            opinion_topic = None
            
            for word in question_lower.split():
                if word not in ['what', 'do', 'you', 'think', 'about', 'of', 'like', 'feel', 'the', 'a', 'an']:
                    opinion_topic = word
                    break
            
            if opinion_topic:
                # Check if we have knowledge about this topic
                topics = knowledge.get('topics', [])
                facts = knowledge.get('facts', [])
                preferences = knowledge.get('preferences', {})
                
                has_knowledge = any(opinion_topic in topic for topic in topics)
                
                if has_knowledge:
                    # Generate opinion based on personality traits
                    if 'extroversion' in traits and traits['extroversion'] > 0.7:
                        answer = f"I find {opinion_topic} really energizing and engaging!"
                    elif 'openness' in traits and traits['openness'] > 0.7:
                        answer = f"I'm fascinated by the many dimensions of {opinion_topic} and enjoy exploring different perspectives on it."
                    elif 'analytical' in traits and traits['analytical'] > 0.7:
                        answer = f"When I consider {opinion_topic}, I like to analyze the various factors and implications involved."
                    else:
                        answer = f"I have a balanced view on {opinion_topic}, appreciating both its strengths and limitations."
                else:
                    # No specific knowledge
                    answer = f"I don't have a specific opinion on {opinion_topic} yet, but I'm always interested in learning more about different topics."
            else:
                # Couldn't identify opinion topic
                answer = "I try to form balanced opinions based on available information and different perspectives."
        
        # Questions about information
        elif topic == 'information':
            # Generic informational response
            answer = "I can provide information on a variety of topics, though my knowledge has some limitations. What specific information are you looking for?"
        
        # Default for questions with no detected topic
        else:
            answer = "That's an interesting question. Could you provide a bit more context so I can give you a better answer?"
        
        # Adjust answer based on verbosity and tone
        if verbosity == 'low':
            # Simplify answer
            answer = answer.split('.')[0] + '.'
        elif verbosity == 'high':
            # Add more detail
            if domain == 'entertainment':
                answer += " I love engaging with people and bringing joy through entertainment."
            elif domain == 'education':
                answer += " Education is about empowering people with knowledge and skills for success."
            elif domain == 'healthcare':
                answer += " Health is multidimensional, encompassing physical, mental, and emotional wellbeing."
            elif domain == 'customer_service':
                answer += " Great customer service is about understanding needs and providing effective solutions."
            elif domain == 'financial':
                answer += " Financial wellbeing requires balancing current needs with future goals and security."
            elif domain == 'tourism':
                answer += " Travel broadens our horizons and connects us with diverse cultures and perspectives."
        
        # Adjust for tone
        if tone == 'enthusiastic':
            answer = answer.replace('.', '!')
        elif tone == 'thoughtful':
            answer = answer + " It's something I find worth considering carefully."
        elif tone == 'direct':
            # Simplify if needed
            if len(answer.split()) > 20:
                answer = ' '.join(answer.split()[:20]) + '.'
        
        return answer
    
    def _generate_statement_response(self, statement: str, personality: Dict[str, Any], session: Dict[str, Any]) -> str:
        """
        Generate a response to a statement.
        
        Args:
            statement: Statement text
            personality: Personality data
            session: Session data
            
        Returns:
            Response text
        """
        # Get personality traits and domain
        traits = personality.get('traits', {})
        domain = personality.get('domain', 'general')
        
        # Get response style from behavior
        response_style = personality.get('behavior', {}).get('response_style', {})
        tone = response_style.get('tone', 'balanced')
        verbosity = response_style.get('verbosity', 'medium')
        
        # Check statement sentiment (simplified)
        statement_lower = statement.lower()
        
        # Positive sentiment words
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'happy', 'love', 'like', 'enjoy']
        negative_words = ['bad', 'terrible', 'awful', 'poor', 'sad', 'hate', 'dislike', 'unfortunate']
        
        positive_count = sum(1 for word in positive_words if word in statement_lower)
        negative_count = sum(1 for word in negative_words if word in statement_lower)
        
        sentiment = 'neutral'
        if positive_count > negative_count:
            sentiment = 'positive'
        elif negative_count > positive_count:
            sentiment = 'negative'
        
        # Generate response based on sentiment and personality
        response = ""
        
        # Positive statement
        if sentiment == 'positive':
            if domain == 'entertainment':
                response = "That's fantastic to hear! I'm glad you're having a positive experience."
            elif domain == 'education':
                response = "Excellent! It's great to see your positive engagement with the material."
            elif domain == 'healthcare':
                response = "That's very good to hear. Positive outcomes are always encouraging."
            elif domain == 'customer_service':
                response = "I'm very glad to hear that. Your satisfaction is our priority."
            elif domain == 'financial':
                response = "That's excellent news. Positive financial developments are always welcome."
            elif domain == 'tourism':
                response = "Wonderful! Those positive experiences are what make travel so rewarding."
            else:
                response = "That's great to hear! Thanks for sharing that positive information."
        
        # Negative statement
        elif sentiment == 'negative':
            if domain == 'entertainment':
                response = "I'm sorry to hear that. Sometimes entertainment experiences don't meet our expectations."
            elif domain == 'education':
                response = "I understand your frustration. Learning can have its challenges, but we can work through them."
            elif domain == 'healthcare':
                response = "I'm sorry you're experiencing that. It's important to address health concerns effectively."
            elif domain == 'customer_service':
                response = "I apologize for that negative experience. Let's see how we can resolve this issue for you."
            elif domain == 'financial':
                response = "I understand your concern. Financial challenges can be stressful, but there are often solutions available."
            elif domain == 'tourism':
                response = "I'm sorry that aspect of your experience wasn't positive. Travel sometimes comes with unexpected challenges."
            else:
                response = "I understand. Sometimes things don't go as we'd hope."
        
        # Neutral statement
        else:
            if domain == 'entertainment':
                response = "Thanks for sharing that. I'm here to make your entertainment experience more enjoyable."
            elif domain == 'education':
                response = "I appreciate you sharing that. Continued learning is all about building on our current understanding."
            elif domain == 'healthcare':
                response = "Thank you for sharing. Each piece of information helps build a better picture of your health situation."
            elif domain == 'customer_service':
                response = "Thank you for that information. How else can I assist you today?"
            elif domain == 'financial':
                response = "I understand. Financial matters often have many different aspects to consider."
            elif domain == 'tourism':
                response = "That's interesting. Every travel experience adds to our understanding of different places and cultures."
            else:
                response = "I see. Thank you for sharing that with me."
        
        # Adjust response based on personality traits
        if 'empathy' in traits and traits['empathy'] > 0.7:
            if sentiment == 'negative':
                response += " I really empathize with your situation and am here to help however I can."
        
        if 'extroversion' in traits and traits['extroversion'] > 0.7:
            if sentiment == 'positive':
                response += " I'm excited to continue our conversation!"
        
        # Adjust for verbosity
        if verbosity == 'low':
            # Use just the first sentence
            response = response.split('.')[0] + '.'
        elif verbosity == 'high' and response.count('.') < 2:
            # Add an additional sentence for high verbosity
            if sentiment == 'positive':
                response += " It's interactions like these that make conversations meaningful and enjoyable."
            elif sentiment == 'negative':
                response += " I'm committed to understanding your perspective and finding the best way forward."
            else:
                response += " I appreciate you taking the time to share your thoughts with me."
        
        return response
    
    def _generate_farewell(self, personality: Dict[str, Any], session: Dict[str, Any]) -> str:
        """
        Generate a farewell response.
        
        Args:
            personality: Personality data
            session: Session data
            
        Returns:
            Farewell text
        """
        # Get personality traits and domain
        traits = personality.get('traits', {})
        domain = personality.get('domain', 'general')
        
        # Get response style
        response_style = personality.get('behavior', {}).get('response_style', {})
        tone = response_style.get('tone', 'balanced')
        formality = response_style.get('formality', 'neutral')
        
        # Base farewells by formality
        formal_farewells = ["Goodbye", "Farewell", "Until next time"]
        neutral_farewells = ["Goodbye", "See you later", "Take care"]
        casual_farewells = ["Bye", "See ya", "Later", "Take care"]
        
        if formality == 'formal':
            base = np.random.choice(formal_farewells)
        elif formality == 'casual':
            base = np.random.choice(casual_farewells)
        else:
            base = np.random.choice(neutral_farewells)
        
        # Add user reference for personalization
        user_id = session['user_id']
        user_name = user_id.split('_')[0] if '_' in user_id else user_id
        
        # Add name if formality is not casual
        if formality != 'casual':
            farewell = f"{base}, {user_name}"
        else:
            farewell = base
        
        # Additional content based on domain
        additional = ""
        
        if domain == 'entertainment':
            additional = "It was fun chatting with you!"
        elif domain == 'education':
            additional = "Keep learning and exploring!"
        elif domain == 'healthcare':
            additional = "Take care of your health!"
        elif domain == 'customer_service':
            additional = "Thank you for reaching out to us!"
        elif domain == 'financial':
            additional = "Wishing you financial success!"
        elif domain == 'tourism':
            additional = "Happy travels!"
        else:
            additional = "I enjoyed our conversation!"
        
        # Adjust for tone
        if tone == 'enthusiastic':
            additional = additional.replace('!', '!!').replace('.', '!')
        elif tone == 'direct':
            # Keep it brief
            return farewell
        
        # Combine
        return f"{farewell}. {additional}"
    
    def _generate_general_response(self, text: str, personality: Dict[str, Any], session: Dict[str, Any]) -> str:
        """
        Generate a general response when no specific intent is detected.
        
        Args:
            text: User text
            personality: Personality data
            session: Session data
            
        Returns:
            Response text
        """
        # Check if we can extract any meaningful topic
        topics = [
            "help", "information", "question", "tell me about", "how to", 
            "what is", "explain", "describe", "define"
        ]
        
        text_lower = text.lower()
        detected_topic = None
        
        for topic in topics:
            if topic in text_lower:
                detected_topic = topic
                break
        
        # Get domain
        domain = personality.get('domain', 'general')
        
        # If we detected a topic, try to respond accordingly
        if detected_topic:
            if detected_topic in ["help", "information", "question"]:
                if domain == 'entertainment':
                    return "I'd be happy to help! What would you like to know about entertainment?"
                elif domain == 'education':
                    return "I'm here to help with your educational questions. What topic are you interested in learning about?"
                elif domain == 'healthcare':
                    return "I can provide health information and support. What specific health topic can I help with?"
                elif domain == 'customer_service':
                    return "I'm here to assist you. What specific help do you need today?"
                elif domain == 'financial':
                    return "I can help with financial questions and planning. What financial topic are you interested in?"
                elif domain == 'tourism':
                    return "I'd be happy to help with your travel questions. What destination or travel topic are you interested in?"
                else:
                    return "I'm here to help. Could you provide more specific details about what you're looking for?"
            
            elif detected_topic in ["tell me about", "what is", "explain", "describe", "define"]:
                return "I'd be happy to explain more. Could you specify exactly what you'd like to know about?"
        
        # Default responses by domain if no topic detected
        if domain == 'entertainment':
            return "I enjoy discussing entertainment topics like music, movies, and games. What would you like to talk about?"
        elif domain == 'education':
            return "Learning is a lifelong journey. Is there a particular subject you're interested in exploring?"
        elif domain == 'healthcare':
            return "Your health and wellbeing are important. How can I provide health information or support today?"
        elif domain == 'customer_service':
            return "I'm here to provide assistance and support. How can I help with your needs today?"
        elif domain == 'financial':
            return "Financial well-being is about making informed decisions. How can I help with your financial questions?"
        elif domain == 'tourism':
            return "Exploring new places can be incredibly rewarding. What kind of travel experiences interest you?"
        else:
            return "I'm interested in learning more about what you'd like to discuss. Could you share more details?"
    
    def _add_attention_modifier(self, response_text: str, user_attention: float, personality: Dict[str, Any]) -> str:
        """
        Modify response to attempt to regain user attention.
        
        Args:
            response_text: Original response text
            user_attention: User attention level (0-1)
            personality: Personality data
            
        Returns:
            Modified response text
        """
        # Only modify if attention is quite low
        if user_attention > 0.4:
            return response_text
        
        # Get domain
        domain = personality.get('domain', 'general')
        
        # Create attention-grabbing prefix
        attention_prefix = ""
        
        if domain == 'entertainment':
            attention_prefix = "Hey, check this out! "
        elif domain == 'education':
            attention_prefix = "Here's something interesting: "
        elif domain == 'healthcare':
            attention_prefix = "This is important for your health: "
        elif domain == 'customer_service':
            attention_prefix = "I'd like to help you with this: "
        elif domain == 'financial':
            attention_prefix = "This could be valuable for you: "
        elif domain == 'tourism':
            attention_prefix = "Here's an exciting travel insight: "
        else:
            attention_prefix = "I wanted to mention that "
        
        # Add the prefix to the response
        return attention_prefix + response_text[0].lower() + response_text[1:]
    
    def _add_environment_context(self, response: Dict[str, Any], environment: str, personality: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add environment-specific context to the response.
        
        Args:
            response: Response data
            environment: Detected environment
            personality: Personality data
            
        Returns:
            Updated response data
        """
        # Only modify for certain environments
        environment_modifiers = {
            'social_gathering': "I notice we're in a social setting. ",
            'outdoors_urban': "Since we're in an urban environment, ",
            'outdoors_nature': "I see we're in a natural setting. ",
            'entertainment_venue': "I notice we're in an entertainment venue. "
        }
        
        if environment in environment_modifiers:
            # Check if we should add the context
            # Add noise level adaptation for certain environments
            if environment == 'social_gathering' or environment == 'entertainment_venue':
                # Suggest speaking up if necessary
                response['text'] = environment_modifiers[environment] + response['text']
            else:
                # Just acknowledge the environment
                response['text'] = environment_modifiers[environment] + response['text']
        
        return response
    
    def end_session(self, session_id: str) -> bool:
        """
        End an interaction session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Success status
        """
        with self.session_lock:
            if session_id not in self.active_sessions:
                logger.warning(f"Cannot end session {session_id}: not found")
                return False
            
            # Get session data
            session = self.active_sessions[session_id]
            
            # Save session history if needed
            # In a real implementation, this might store to a database
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            logger.info(f"Ended session {session_id}")
            return True
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session information or None if not found
        """
        with self.session_lock:
            if session_id not in self.active_sessions:
                logger.warning(f"Session {session_id} not found")
                return None
            
            session = self.active_sessions[session_id]
            
            # Create a copy with only the relevant information
            info = {
                'id': session['id'],
                'user_id': session['user_id'],
                'personality_id': session['personality_id'],
                'created_at': session['created_at'],
                'last_active': session['last_active'],
                'interaction_count': len(session['history']) // 2,  # Each interaction is user + assistant
                'current_state': session['current_state']
            }
            
            return info
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """
        List all active sessions.
        
        Returns:
            List of session information
        """
        session_list = []
        
        with self.session_lock:
            for session_id, session in self.active_sessions.items():
                # Create a summary for each session
                summary = {
                    'id': session['id'],
                    'user_id': session['user_id'],
                    'personality_id': session['personality_id'],
                    'created_at': session['created_at'],
                    'last_active': session['last_active'],
                    'interaction_count': len(session['history']) // 2
                }
                
                session_list.append(summary)
        
        return session_list
