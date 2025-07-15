import base64
import json
from flask import Flask, render_template, request, jsonify
from lib.common_function import speech_to_text, text_to_speech, chat_with_ai, generate_cover_letter, generate_cv_improvements, process_rag_query, rag_process_document
from flask_cors import CORS
import os
from pypdf import PdfReader

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/ai-chat', methods=['GET'])
def ai_chat():
    return render_template('ai_chat.html')

@app.route('/resume-builder', methods=['GET'])
def resume_builder():
    return render_template('resume_builder.html')

@app.route('/image-generator', methods=['GET'])
def image_generator():
    return render_template('image_generator.html')

@app.route('/video-generator', methods=['GET'])
def video_generator():
    return render_template('video_generator.html')

@app.route('/audio-generator', methods=['GET'])
def audio_generator():
    return render_template('audio_generator.html')

@app.route('/chess', methods=['GET'])
def chess():
    return render_template('chess.html')

@app.route('/stock-market-predictor', methods=['GET'])
def stock_market_predictor():
    return render_template('stock_market_predictor.html')

@app.route('/resume-builder/generate-results', methods=['POST'])
def generate_results():
    print("here in generate_results")
    resume_file_available = request.form.get('resume')
    position_name = request.form.get('position')
    company = request.form.get('company')
    job_desc = request.form.get('jobDesc')
    cv_content = request.form.get('cvContent')


    resume_content = ""

    if resume_file_available != 'undefined':
        resume_file = request.files['resume']  # Get the uploaded PDF file
        reader = PdfReader(resume_file)
        for i in range(len(reader.pages)):
            resume_content += reader.pages[i].extract_text()
    else:
        resume_content = cv_content    

    cover_letter = generate_cover_letter(company_name=company, 
                                         position_name=position_name, 
                                         job_description=job_desc, 
                                         resume_content=resume_content)
    
    cv_improvements = generate_cv_improvements(position_applied=position_name, 
                                               job_description=job_desc, 
                                               resume_content=resume_content)
    print("ending generate_results")
    
    return jsonify({"coverLetter": cover_letter, "cvImprovements": cv_improvements})
    
    






@app.route('/speech-to-text', methods=['POST'])
def speech_to_text_route():

    print("processing speech-to-text")
    audio_binary = request.data # Get the user's speech from their request
    text = speech_to_text(audio_binary) # Call speech_to_text function to transcribe the speech

    # Return the response back to the user in JSON format
    response = app.response_class(
        response=json.dumps({'text': text}),
        status=200,
        mimetype='application/json'
    )
    print(response)
    print(response.data)
    return response


@app.route('/process-message', methods=['POST'])
def process_prompt_route():
    user_message = request.json['userMessage'] # Get user's message from their request
    print('user_message', user_message)
    voice = request.json['voice'] # Get user's preferred voice from their request
    print('voice', voice)

    # Call chat_with_AI function to process the user's message and get a response back
    response_text = chat_with_ai(user_message)

    # Clean the response to remove any emptylines
    response_text = os.linesep.join([s for s in response_text.splitlines() if s])

    
    file_name = text_to_speech(response_text, voice)
    with open(file_name, "rb") as audio_file:
        response_speech = audio_file.read()
    

    # convert response_speech to base64 string so it can be sent back in the JSON response
    response_speech = base64.b64encode(response_speech).decode('utf-8')

    # Send a JSON response back to the user containing their message's response both in text and speech formats
    response = app.response_class(
        response=json.dumps({"openaiResponseText": response_text, "openaiResponseSpeech": response_speech}),
        status=200,
        mimetype='application/json'
    )
    print(response)
    return response


# Define the route for the RAG page
@app.route('/personal-data-assistant', methods=['GET'])
def personal_data_assistant():
    return render_template('personal_data_assistant.html')  # Render the index.html template

# Define the route for processing messages for
@app.route('/rag-process-message', methods=['POST'])
def rag_process_message():
    user_message = request.json['userMessage']  # Extract the user's message from the request
    print('user_message', user_message)

    bot_response = process_rag_query(user_message)  # Process the user's message 

    # Return the bot's response as JSON
    return jsonify({
        "botResponse": bot_response
    }), 200

# Define the route for processing documents
@app.route('/process-document', methods=['POST'])
def process_document_route():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({
            "botResponse": "It seems like the file was not uploaded correctly, can you try "
                           "again. If the problem persists, try using a different file"
        }), 400

    file = request.files['file']  # Extract the uploaded file from the request

    file_path = file.filename  # Define the path where the file will be saved
    file.save(file_path)  # Save the file

    rag_process_document(file_path)  # Process the document using the worker module

    # Return a success message as JSON
    return jsonify({
        "botResponse": "Thank you for providing your PDF document. I have analyzed it, so now you can ask me any "
                       "questions regarding it!"
    }), 200




if __name__ == "__main__":
    app.run(port=8000, host='0.0.0.0')  
