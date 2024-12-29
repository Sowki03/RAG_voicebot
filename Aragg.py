import vosk
import sounddevice as sd
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
from io import BytesIO
import pygame
import cohere

class AI_Assistant:
    
    
    def __init__(self):
        # Load Vosk model
        print("Loading Vosk model...")
        self.vosk_model = vosk.Model(r"E:\A Final Year Project\c\vosk-model-small-en-us-0.15")
        
        # Initialize Cohere client
        self.cohere_client = cohere.Client('9hAaW7LlSONXArJDLwcXscd6sxFoKX9wqs257Gil')

        # Initialize conversation history
        print("Setting up conversation history...")
        self.full_transcript = [
            {"role": "system", "content": "You are a receptionist at Amrita Vishwa Vidyapeetham. Provide accurate information and assistance on college-related queries."}
        ]

        # Load the FAQ data and generate embeddings
        print("Loading FAQs...")
        self.faqs = self.load_faqs_from_json("clean.json")
        self.faq_texts = [f["question"] + " " + f["answer"] for f in self.faqs]
        self.faq_embeddings = self.generate_faq_embeddings(self.faq_texts)

        # Initialize pygame for audio playback
        pygame.mixer.init()
        self.exit_sentences = [
            "thank you", 
            "that's all", 
            "I got the information", 
            "no more questions", 
            "I'm done", 
            "that's all I needed", 
            "all good", 
            "got it", 
            "thanks for the help", 
            "you've been helpful", 
            "I have no more questions", 
            "that will be all", 
            "goodbye", 
            "bye", 
            "see you later", 
            "I appreciate it"
        ]
        self.exit_embeddings = self.generate_faq_embeddings(self.exit_sentences)

    def load_faqs_from_json(self, file_path):
        """Load FAQs from a JSON file."""
        with open(file_path, 'r') as file:
            faqs = json.load(file)
        return faqs

    def generate_faq_embeddings(self, faq_texts):
        """Generate embeddings for each FAQ."""
        response = self.cohere_client.embed(texts=faq_texts)
        return response.embeddings

    def find_top_faqs(self, query, top_k=3):
        """Find the top k FAQs based on similarity to the query."""
        # Generate embedding for the query
        query_embedding = self.cohere_client.embed(texts=[query]).embeddings[0]

        # Compute cosine similarity between query and FAQ embeddings
        similarities = cosine_similarity([query_embedding], self.faq_embeddings)[0]

        # Get indices of top_k FAQs
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]

        # Retrieve the top FAQs based on similarity
        top_faqs = [{"question": self.faqs[i]["question"], "answer": self.faqs[i]["answer"], "similarity": similarities[i]} for i in top_k_indices]
        
        return top_faqs
    
    def check_exit_condition(self, query):
        """Check if the query indicates the end of the conversation."""
        query_embedding = self.cohere_client.embed(texts=[query]).embeddings[0]
        similarities = cosine_similarity([query_embedding], self.exit_embeddings)[0]
        return any(score > 0.8 for score in similarities)
    
    
    def generate_receptionist_response(self, query, top_faqs):
        """Generate a receptionist-like response based on top FAQs."""
        # Prepare context for generating a response
        context = f"Query: {query}\n\nHere are the top 3 FAQs that might help:\n"
        for i, faq in enumerate(top_faqs, 1):
            context += f"FAQ {i}: {faq['question']} - {faq['answer']}\n"
        context += "\nNow, using the above information, please provide precise 2-3 lines answer like a human receptionist at amrita.After providing the answer, end with: Let me know if you have any other queries.don't say hello or something in starting just give the answer"

        # Call Cohere to generate the response
        response = self.cohere_client.generate(
            model='command-xlarge',
            prompt=context,
            max_tokens=150,
            temperature=0.7
        )

        return response.generations[0].text.strip()

    def generate_ai_response(self, transcript_text):
        """Generate AI response based on the user's input."""
        print("Adding user message to full_transcript.")
        self.full_transcript.append({"role": "user", "content": transcript_text})
        
        if self.check_exit_condition(transcript_text):
            goodbye_message = "Thank you for reaching out to Amrita Vishwa Vidyapeetham. Have a great day!"
            self.generate_audio(goodbye_message)
            print(goodbye_message)
            exit(0)
        # Find top 3 relevant FAQs for the given query
        top_faqs = self.find_top_faqs(transcript_text)

        # Generate a response using the context from top FAQs
        ai_response = self.generate_receptionist_response(transcript_text, top_faqs)
        self.full_transcript.append({"role": "assistant", "content": ai_response})
        
        # Generate audio response
        self.generate_audio(ai_response)
        print(f"\nAI Receptionist: {ai_response}")

    def generate_audio(self, text):
        """Convert text to speech and play it."""
        print("Generating audio response...")
        tts = gTTS(text=text, lang='en')
        audio_data = BytesIO()
        tts.write_to_fp(audio_data)
        audio_data.seek(0)

        # Play audio
        pygame.mixer.music.load(audio_data)
        pygame.mixer.music.play()

        # Wait for audio to finish before allowing the next transcription
        while pygame.mixer.music.get_busy():  # Check if audio is still playing
            pygame.time.Clock().tick(10)  # Wait a short time before checking again

    def start_transcription(self):
        """Start real-time transcription with Vosk."""
        print("\nStarting real-time transcription...")
        duration =7   # Duration for each audio recording in seconds

        while True:
            print("Recording...")
            # Record audio
            audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='int16')
            sd.wait()
            
            print("Transcribing...")
            # Convert numpy array to bytes
            audio_bytes = audio.tobytes()

            # Transcribe audio using Vosk
            recognizer = vosk.KaldiRecognizer(self.vosk_model, 16000)
            recognizer.AcceptWaveform(audio_bytes)
            result = json.loads(recognizer.Result())
            text = result.get('text', '').strip()

            if text:
                print(f"User: {text}")
                self.generate_ai_response(text)

# Initialize AI Assistant
greeting = "Thank you for calling Amrita Vishwa Vidyapeetham. My name is dauna, how may I assist you?"
ai_assistant = AI_Assistant()
ai_assistant.generate_audio(greeting)

# Start real-time transcription
ai_assistant.start_transcription()
