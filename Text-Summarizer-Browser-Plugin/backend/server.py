# Importing necessary Libraries
import time
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from code import load_llm, pre_process_url_data, qa_on_url_summarized_text, pre_process_pdf_data, qa_on_pdf_summarized_text, _validate_external_url
import tempfile
import chromadb

# Initializing the flask app and enabling CORS
app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


@app.route('/select-model', methods=['POST'])
def select_model():
    """
        Model selection function which would further trigger Model compilation function.
    """
    try:
        global current_model
        data = request.get_json()
        model_id = data.get('model_id')
        current_model = load_llm(model_id)
        return jsonify({'message': f'Model {model_id} loaded successfully.'}), 200

    except Exception:
        return jsonify({'message': 'Failed to load model \n'}), 500


def stream_output(process_function, *args):
    """
        Generator function to stream output from a process function.
    """
    try:
        for chunk in process_function(*args):
            if chunk is not None:
                yield f"{chunk}"
    except Exception:
        yield f"Error while streaming output"


@app.route('/process-url', methods=['POST'])
def process_url():
    """
        Fetches URL from the plugin & triggers the URL summarization function.
    """
    try:
        data = request.get_json()
        url = data.get('url')
        if not url:
            return jsonify({'message': 'No URL provided'}), 400
        try:
            _validate_external_url(url)
        except ValueError as e:
            return jsonify({'message': f'Invalid URL: {e}'}), 400
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        return Response(stream_output(pre_process_url_data, [url]), content_type='text/event-stream')

    except Exception:
        return jsonify({'message': f'Error while processing URL'}), 400


@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    """
        Once the PDF's uploaded, the PDF Summarization function's triggered.
    """

    pdf_file = request.files['pdf']
    if pdf_file and pdf_file.content_type == 'application/pdf':
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                pdf_file.save(temp_pdf.name)
                temp_pdf_path = temp_pdf.name
                print(temp_pdf_path)

            chromadb.api.client.SharedSystemClient.clear_system_cache()
            return Response(stream_output(pre_process_pdf_data, temp_pdf_path), content_type='text/event-stream')

        except Exception:
            return jsonify({"message": "Error processing PDF"}), 500

    else:
        return jsonify({"message": "Invalid file type. Please upload a PDF."}), 400


@app.route('/your_query_pdf', methods=['POST'])
def pdf_process_query():
    """
        This function triggers the PDF Question Answering Bot
    """
    try:
        data = request.get_json()
        query = data.get('query')
        if not data:
            return jsonify({'message': 'no query provided'}), 400
        response_message = str(qa_on_pdf_summarized_text(query))
        return jsonify({'message': response_message}), 200
    except Exception:
        return jsonify({'message': 'Error while PDF QA Bot'}), 500


@app.route('/your_query_url', methods=['POST'])
def url_process_query():
    """
        This function triggers the URL question answering Bot
    """
    try:
        data = request.get_json()
        query = data.get('query')
        if not data:
            return jsonify({'message': 'no query provided'}), 400
        response_message = str(qa_on_url_summarized_text(query))
        return jsonify({'message': response_message}), 200
    except Exception:
        return jsonify({'message': 'Error while URL QA Bot'}), 500


if __name__ == '__main__':
    app.run(port=5000)
