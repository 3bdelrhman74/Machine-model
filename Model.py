from flask import Flask, request, jsonify
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient('mongodb+srv://abdo:HYHguYtR7re8oz4s@cluster0.jqqtyhf.mongodb.net/lms?retryWrites=true&w=majority')
db = client['lms']
collection = db['courses']

def preprocess_skills(df):
    df['skills_combined'] = df['skills'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    return df

def create_csr_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['skills_combined'])
    return tfidf_matrix, tfidf

def knn_fit(csr_data):
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(csr_data)
    return knn

def recommend_courses(model, tfidf, data, course_title, n=5):
    course_idx = data.index[data['title'] == course_title].tolist()[0]
    csr_data = tfidf.transform([data.loc[course_idx, 'skills_combined']])
    distances, indices = model.kneighbors(csr_data, n_neighbors=n+1)
    recommended_courses = [data.iloc[i]['title'] for i in indices.flatten() if i != course_idx]
    return recommended_courses

@app.route('/api/v1/recommend', methods=['GET'])
def recommend():
    course_title = request.args.get('course_title')
    n = int(request.args.get('n', 5))

    data = pd.DataFrame(list(collection.find({}, {'title': 1, 'skills': 1})))
    data = preprocess_skills(data)
    csr_data, tfidf = create_csr_matrix(data)
    knn_model = knn_fit(csr_data)

    recommended_courses = recommend_courses(knn_model, tfidf, data, course_title, n)
    return jsonify(recommended_courses)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
