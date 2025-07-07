from flask import render_template, request, flash, Blueprint, jsonify
from app.bot.utils import create_qa_bot, format_response_for_web
import logging
import traceback

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot_bp = Blueprint('bot_bp', __name__)

# globla bot instance
qa_bot = None

def load_qa_bot():
    """ Load the Q&A bot instance"""
    global qa_bot
    if qa_bot is None:
        try:
            logger.info('Loading Georgian Bot..')
            qa_bot = create_qa_bot()
            logger.info('Bot loaded succesfully')
        except Exception as e:
            logger.error(f'Error loading bot: {e}')
            traceback.print_exc()
            return None
    return qa_bot

"""Main page"""
@bot_bp.route('/')
def index():
    """ Main chatbot page"""
    return render_template('base.html')

@bot_bp.route('/ask', methods=['POST'])
def ask_question():
    """handle question submission via AJAX"""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                'success': False,
                'error': 'კითხვა არ არის მიწოდებული'
            }), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({
                'success': False,
                'error': 'კითხვა არ შეიძლება იყოს ცარიელი'
            }), 400
        
        bot = load_qa_bot()
        if bot is None:
            return jsonify({
                'success': False,
                'error': 'ბოტი მიუწვდომელია. გთხოვთ მოგვიანებით სცადოთ.'
            }), 500
        
        logger.info(f'Processing question: {question}')
        response = bot.get_conversation_response(question)

        formatted_response = format_response_for_web(response)
        logger.info(f"Generated response: {formatted_response['answer'][:100]}...")
        return jsonify(formatted_response)
    except Exception as e:
        logger.error(f'Error processing operation: {e}')
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'შეცდომა მოხდა. გთხოვთ მოგვიანებით სცადოთ.'
        }), 500
    
@bot_bp.route('/health')
def health_check():
    """health check endpoint"""
    bot = load_qa_bot()
    status = "healthy" if bot is not None else "unhealthy"
    return jsonify({
        'status': status,
        'bot_loaded': bot is not None
    })