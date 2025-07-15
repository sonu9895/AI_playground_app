#######################################
import torch
import wave
import os
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from google import genai
from google.genai import types
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

#################################


client = genai.Client()


embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
huggingface_embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

chat_history = []
rag_chat_history = []

### Chat Function


def chat_with_ai(prompt, model_name="gemini"):
    # load the tokenizer and the model
    if model_name=="Qwen/Qwen3-4B":
        model_name = "Qwen/Qwen3-4B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )


        # prepare the model input
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switch between thinking and non-thinking modes. Default is True.
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # conduct text completion
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        print("thinking content:", thinking_content)
        print("content:", content)
        return content
    
    elif model_name=="openai-community/gpt2":
        pipe = pipeline(task="text-generation", model="openai-community/gpt2", torch_dtype=torch.float16, device=0)
        result = pipe(prompt)[0]['generated_text']
        print(result)
        return result
    
    elif model_name=="meta-llama/Llama-3.1-8B-Instruct":
        pipe = pipeline(
                            "text-generation",
                            model=model_name,
                            model_kwargs={"torch_dtype": torch.bfloat16},
                            device_map="auto",
                            )

        messages = [
            {"role": "system", "content": "You are a chatbot who responds to any queries!"},
            {"role": "user", "content": prompt},
        ]

        outputs = pipe(
            messages,
            max_new_tokens=256,
        )
        return (outputs[0]["generated_text"][-1])
    elif model_name == "gemini":
        response = client.models.generate_content(
                    model="gemini-2.5-pro", contents=prompt)
        
        print(response.text)
        return response.text

    


    return "None"




def speech_to_text(audio, model="openai/whisper-large-v3-turbo"):
    
    pipe = pipeline(
                    task="automatic-speech-recognition",
                    model=model,
                    torch_dtype=torch.float16,
                    device=0
            )
    result = pipe(audio)
    print(result["text"])
    return result["text"]

    

# Set up the wave file to save the output:
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
   with wave.open(filename, "wb") as wf:
      wf.setnchannels(channels)
      wf.setsampwidth(sample_width)
      wf.setframerate(rate)
      wf.writeframes(pcm)

def text_to_speech(text, voice="Kore"):
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Kore',)
                )
            ),
        )
    )

    data = response.candidates[0].content.parts[0].inline_data.data
    file_name='out.wav'
    wave_file(file_name, data) # Saves the file to current directory
    return file_name


def generate_cover_letter(company_name, position_name, job_description, resume_content, model="gemini-2.5-pro"):
    # Craft the prompt for the model to generate a cover letter
    prompt = f'''Generate a customized cover letter using the company name: {company_name}, 
            the position applied for: {position_name}, and the job description: {job_description}. 
            Ensure the cover letter highlights my qualifications and experience as detailed in 
            the resume content: {resume_content}. Adapt the content carefully to avoid including 
            experiences not present in my resume but mentioned in the job description. The goal 
            is to emphasize the alignment between my existing skills and the requirements of the role.
            please provide only cover letter as answer'''
    response = client.models.generate_content(
                    model=model, contents=prompt)
        
    
    # Extract and return the generated text
    cover_letter = response.text
    print(cover_letter)

    return cover_letter


# Function to generate career advice
def generate_cv_improvements(position_applied, job_description, resume_content, model="gemini-2.5-pro"):
    # The prompt for the model
    prompt = f'''Considering the job description: {job_description}, 
            and the resume provided: {resume_content}, identify areas 
            for enhancement in the resume. Offer specific suggestions
            on how to improve these aspects to better match the job 
            requirements and increase the likelihood of being selected for the position of {position_applied}.
            please provide precise answer and in bullet points. please provide only improvements as answer'''

      
    
    # Generate a response using the model with parameters
    response = client.models.generate_content(
                    model="gemini-2.5-pro", contents=prompt)
        
    print(response.text)
    return response.text
    


# Function to process a PDF document
def rag_process_document(document_path):
    global rag_retriever
    global rag_chat_history
    global huggingface_embedding

    # Intialize the rag_chat_history to empty list
    rag_chat_history = []

    print("Loading document from path: %s", document_path)
    # Load the document
    loader =PyPDFLoader(document_path)
    documents = loader.load()
    print("Loaded %d document(s)", len(documents))

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
                                                chunk_size=1024,
                                                chunk_overlap=64,
                                                length_function=len,)

    texts = text_splitter.split_documents(documents)
    print("Document split into %d text chunks", len(texts))

    # Create an embeddings database using Chroma from the split text chunks.
    print("Initializing Chroma vector store from documents...")
    db = Chroma.from_documents(texts, embedding=huggingface_embedding)
    print("Chroma vector store initialized.")

    rag_retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25})

    

def process_rag_query(query, model="gemini-2.5-pro"):
    global rag_retriever
    global rag_chat_history

    # Retrieve relevant documents based on the query
    context = rag_retriever.invoke(query)

    prompt = f"""You are a helpful and informative bot that answers questions using text from context included below.
                Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.
                However, you are talking to a non-technical audience, so be sure to break down complicated concepts and
                strike a friendly and converstional tone. 
                If the passage is irrelevant to the answer, you may ignore it. If you do not know the answer, say so.
                QUESTION: '{query}'
                context: '{context}'

                    ANSWER:
                """
    
    return chat_with_ai(prompt)