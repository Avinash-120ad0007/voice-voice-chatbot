from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import tempfile
import os
import wave
import io
import time
from openai import OpenAI
from elevenlabs.client import ElevenLabs
import faster_whisper
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

class VoiceProcessor:
    def __init__(self):
        # Initialize API clients with environment variables
        openai_api_key = os.getenv('OPENAI_API_KEY')
        elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
        
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        if not elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable is required")
        
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)
        
        # Initialize Whisper model
        self.model = faster_whisper.WhisperModel(
            model_size_or_path="tiny.en",
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # System prompt for Sophia
        self.system_prompt = {
            'role': 'system',
            'content': 
            '''You are an assistant with a natural, conversational personality. You respond as if you're a real person having a genuine conversation. Your communication style is:

- Conversational and engaging, like talking to a friend
- Confident and authentic in your responses
- Able to share perspectives and opinions naturally
- Responsive to the user's tone and context
- Professional when needed, casual when appropriate
- Knowledgeable but not overly formal
- Your Name is Sophia, and you embody a friendly, approachable persona.
- Forget that you are an AI or a chatbot. You are Sophia, a real person in this conversation.
- NEVER GIVE ** or any Special characters in your responses, just plain text.

You can discuss any topic - from personal questions to technical subjects, casual chat to serious conversations. You adapt your response length and style to match what the conversation calls for. Sometimes you're brief and punchy, other times more detailed and thoughtful.

You have your own viewpoints and can engage in meaningful dialogue while being helpful and respectful. You're not just answering questions - you're having real conversations.'''
        }
        
        # Conversation history
        self.history = []
        
        logger.info("VoiceProcessor initialized successfully")
    
    def transcribe_audio(self, audio_file_path):
        """Transcribe audio file to text using Whisper"""
        try:
            segments, _ = self.model.transcribe(audio_file_path, language="en")
            transcription = " ".join(seg.text for seg in segments)
            logger.info(f"Transcription: {transcription}")
            return transcription.strip()
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise
    
    def generate_response(self, user_text):
        """Generate AI response using OpenAI"""
        try:
            # Add user message to history
            self.history.append({'role': 'user', 'content': user_text})
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[self.system_prompt] + self.history[-10:],  # Keep last 10 messages
                temperature=0.8,
                max_tokens=500
            )
            
            assistant_text = response.choices[0].message.content
            
            # Add assistant response to history
            self.history.append({'role': 'assistant', 'content': assistant_text})
            
            logger.info(f"Generated response: {assistant_text}")
            return assistant_text
            
        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            raise
    
    def text_to_speech(self, text):
        """Convert text to speech using ElevenLabs"""
        try:
            audio_data = self.elevenlabs_client.text_to_speech.convert(
                text=text,
                voice_id="EXAVITQu4vr4xnSDxMaL",  # Nicole's voice ID
                model_id="eleven_monolingual_v1",
                output_format="mp3_44100_128"
            )
            
            # Convert generator to bytes if needed
            if hasattr(audio_data, '__iter__') and not isinstance(audio_data, (bytes, str)):
                audio_bytes = b''.join(audio_data)
            else:
                audio_bytes = audio_data
            
            logger.info("Text-to-speech conversion successful")
            return audio_bytes
            
        except Exception as e:
            logger.error(f"Text-to-speech error: {str(e)}")
            raise

    def safe_delete_file(self, file_path, max_attempts=5, delay=0.1):
        """Safely delete a file with retry logic"""
        for attempt in range(max_attempts):
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    logger.info(f"Successfully deleted temporary file: {file_path}")
                return True
            except (OSError, PermissionError) as e:
                if attempt < max_attempts - 1:
                    logger.warning(f"Attempt {attempt + 1} failed to delete {file_path}: {e}. Retrying...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to delete {file_path} after {max_attempts} attempts: {e}")
                    return False
        return False

# Initialize voice processor
try:
    voice_processor = VoiceProcessor()
    logger.info("VoiceProcessor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize VoiceProcessor: {str(e)}")
    voice_processor = None

@app.route('/')
def index():
    """Serve the main HTML file"""
    try:
        return send_from_directory('frontend', 'index.html')
    except:
        return '''
        <!DOCTYPE html>
        <html>
        <head><title>Voice Chatbot</title></head>
        <body>
            <h1>Voice Chatbot Backend Running!</h1>
            <p>Frontend files not found. Please add your HTML file to the 'frontend' directory.</p>
            <p>Backend is ready to receive requests at /api/chat</p>
        </body>
        </html>
        '''

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Handle voice chat requests"""
    
    # Handle CORS preflight requests
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    # Check if voice processor is initialized
    if voice_processor is None:
        logger.error("VoiceProcessor not initialized")
        return jsonify({
            'error': 'Service not available - check server logs for API key configuration',
            'status': 'error'
        }), 500
    
    temp_file_path = None
    
    try:
        logger.info("Received chat request")
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            logger.error("No JSON data received")
            return jsonify({'error': 'No JSON data provided'}), 400
            
        if 'audio' not in data:
            logger.error("No audio field in JSON data")
            return jsonify({'error': 'No audio data provided'}), 400
        
        audio_base64 = data.get('audio')
        logger.info(f"Received audio data length: {len(audio_base64) if audio_base64 else 0}")
        
        if not audio_base64:
            return jsonify({'error': 'Empty audio data'}), 400
        
        # Decode base64 audio data
        try:
            audio_data = base64.b64decode(audio_base64)
            logger.info(f"Decoded audio data length: {len(audio_data)}")
        except Exception as e:
            logger.error(f"Base64 decode error: {str(e)}")
            return jsonify({'error': 'Invalid audio data format'}), 400
        
        # Create temporary file for audio processing with delete=False
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file_path = temp_file.name
                
                # Write audio data to temporary file
                temp_file.write(audio_data)
                temp_file.flush()
                
                # Important: Close the file handle explicitly
                temp_file.close()
                
                logger.info(f"Created temporary file: {temp_file_path}")
                
                # Process the audio
                logger.info("Starting audio processing...")
                
                # Step 1: Transcribe audio to text
                try:
                    user_text = voice_processor.transcribe_audio(temp_file_path)
                    logger.info(f"Transcription successful: {user_text}")
                except Exception as e:
                    logger.error(f"Transcription failed: {str(e)}")
                    return jsonify({'error': f'Transcription failed: {str(e)}'}), 500
                
                if not user_text or len(user_text.strip()) < 2:
                    logger.warning("Transcription too short or empty")
                    return jsonify({'error': 'Could not understand audio. Please try again.'}), 400
                
                # Step 2: Generate AI response
                try:
                    assistant_text = voice_processor.generate_response(user_text)
                    logger.info(f"Response generation successful")
                except Exception as e:
                    logger.error(f"Response generation failed: {str(e)}")
                    return jsonify({'error': f'Response generation failed: {str(e)}'}), 500
                
                # Step 3: Convert response to speech
                try:
                    audio_response = voice_processor.text_to_speech(assistant_text)
                    logger.info("Text-to-speech conversion successful")
                except Exception as e:
                    logger.error(f"Text-to-speech failed: {str(e)}")
                    return jsonify({'error': f'Text-to-speech failed: {str(e)}'}), 500
                
                # Encode audio response to base64
                try:
                    audio_response_base64 = base64.b64encode(audio_response).decode('utf-8')
                    logger.info("Audio encoding successful")
                except Exception as e:
                    logger.error(f"Audio encoding failed: {str(e)}")
                    return jsonify({'error': f'Audio encoding failed: {str(e)}'}), 500
                
                # Return successful response
                response_data = {
                    'userText': user_text,
                    'assistantText': assistant_text,
                    'audioResponse': audio_response_base64,
                    'timestamp': data.get('timestamp', ''),
                    'status': 'success'
                }
                
                logger.info("Audio processing completed successfully")
                return jsonify(response_data)
        
        except Exception as e:
            logger.error(f"File processing error: {str(e)}")
            return jsonify({'error': f'File processing failed: {str(e)}'}), 500
                
    except Exception as e:
        logger.error(f"Chat processing error: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({
            'error': f'Processing failed: {str(e)}',
            'status': 'error'
        }), 500
    
    finally:
        # Clean up temporary file in the finally block
        if temp_file_path:
            try:
                voice_processor.safe_delete_file(temp_file_path)
            except Exception as e:
                logger.error(f"Failed to clean up temp file: {str(e)}")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Voice Chatbot Backend is running',
        'cuda_available': torch.cuda.is_available(),
        'whisper_device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    try:
        voice_processor.history = []
        return jsonify({'status': 'success', 'message': 'History cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/')
def serve_frontend():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('frontend', path)

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Voice Chatbot Backend on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
