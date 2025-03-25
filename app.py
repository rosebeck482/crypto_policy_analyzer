# Flask web app for Cryptocurrency Policy Analyzer
# Provides interface for querying Elasticsearch vector store

import os
import logging
import time
import sys
from datetime import datetime
from flask import Flask, render_template, request, jsonify, g
from dotenv import load_dotenv

# Use the correct import path for WebAnalyzer
from policy_analyzer.web_analyzer import WebAnalyzer

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
           template_folder='policy_analyzer/templates',
           static_folder='policy_analyzer/static')
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "default-secret-key")

# Configure Elasticsearch
ES_URL = os.environ.get("ES_URL", "https://my-elasticsearch-project-ad7bed.es.us-east-1.aws.elastic.cloud:443")
ES_INDEX_NAME = os.environ.get("ES_INDEX_NAME", "crypto-policy")

# Get or create the WebAnalyzer instance
def get_analyzer():
    if 'analyzer' not in g:
        logger.info("Initializing WebAnalyzer")
        g.analyzer = WebAnalyzer()
    return g.analyzer

# Log request information and set start time
@app.before_request
def before_request():
    g.start_time = time.time()
    logger.info(f"Request: {request.method} {request.path}")

# Log response information and timing
@app.after_request
def after_request(response):
    if hasattr(g, 'start_time'):
        elapsed = time.time() - g.start_time
        logger.info(f"Response: {response.status_code} - took {elapsed:.2f}s")
    return response

# Render the main page
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint to query the vector store
@app.route('/api/query', methods=['POST'])
def query():
    try:
        # Get JSON data from request
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400
            
        query_text = data['query']
        logger.info(f"Received query: {query_text}")
        
        # Process the query
        analyzer = get_analyzer()
        start_time = time.time()
        results = analyzer.process_query(query_text)
        elapsed = time.time() - start_time
        
        # Add timing information to the response
        results['timing'] = {
            'total_seconds': elapsed,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Query processed in {elapsed:.2f}s")
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health():
    try:
        # Check if Elasticsearch index exists via the analyzer
        analyzer = get_analyzer()
        
        # Check analyzer health
        result = analyzer.health_check()
        
        response_data = {
            "status": "healthy",
            "message": "API is operational",
            "elasticsearch": result.get("status", "unknown"),
            "index": ES_INDEX_NAME
        }
        
        # Get storage details if possible
        try:
            # Check Elasticsearch for stats
            response_data["storage"] = {
                "index_name": ES_INDEX_NAME,
                "url": ES_URL
            }
        except Exception as e:
            logger.warning(f"Could not get Elasticsearch stats: {str(e)}")
            
        return jsonify(response_data)
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    logger.info(f"Starting Flask app on port 5001")
    logger.info(f"Using Elasticsearch index: {ES_INDEX_NAME}")
    logger.info(f"Elasticsearch URL: {ES_URL}")
    app.run(debug=True, port=5001) 