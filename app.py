import nltk
from flask import Flask, request, render_template, session
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from data.questions import QUESTIONS  # Import the QUESTIONS dictionary

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session functionality

# Preprocess questions
stop_words = set(stopwords.words('english'))
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return ' '.join([word for word in tokens if word not in stop_words])

processed_questions = {q: preprocess(q) for q in QUESTIONS.keys()}
answers = list(QUESTIONS.values())

@app.route('/', methods=['GET', 'POST'])
def chatbot():
    # Initialize chat history in session if not present
    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == 'POST':
        action = request.form.get('action')  # Determine which button was clicked

        if action == 'submit':  # When submitting a new question
            user_question = request.form['question']
            preprocessed_user_question = preprocess(user_question)

            # Combine user question with stored questions for vectorization
            corpus = [preprocessed_user_question] + list(processed_questions.values())
            
            # Vectorize questions using TF-IDF
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(corpus)

            # Calculate cosine similarity
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            most_similar_index = similarities.argsort()[0, -1]  # Index of the most similar question
            similarity_score = similarities[0, most_similar_index]

            # Set a similarity threshold
            if similarity_score > 0.5:  # Adjust threshold as needed
                response = answers[most_similar_index]
            else:
                response = "Sorry, I don't understand that question."

            # Append question and response to chat history
            session['chat_history'].append({'question': user_question, 'answer': response})
            session.modified = True  # Mark session as modified to save changes

        elif action == 'clear':  # When clearing the chat
            session.pop('chat_history', None)  # Completely remove chat history from the session
            session.modified = True  # Mark session as modified to save changes

    return render_template('chatbot.html', chat_history=session.get('chat_history', []))

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5002)
