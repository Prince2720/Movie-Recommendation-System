# 🎬 Movie Recommendation System

A content-based movie recommendation engine that suggests similar movies based on genres, cast, crew, keywords, and plot descriptions. Built using machine learning techniques and natural language processing.

## ✨ Features

- 🎯 **Content-Based Filtering**: Recommends movies based on movie attributes and features
- 🔍 **Smart Similarity Matching**: Uses cosine similarity to find similar movies
- 🎭 **Multiple Parameters**: Considers genres, cast, director, keywords, and overview
- 📊 **Interactive UI**: User-friendly Streamlit interface (if applicable)
- 🎨 **Movie Posters**: Displays movie posters using TMDB API (if applicable)
- ⚡ **Fast Recommendations**: Get instant movie suggestions
- 📈 **Similarity Scores**: Shows how similar recommended movies are


## 🔧 How It Works

### Content-Based Filtering Algorithm

1. **Feature Extraction**: Combines movie metadata (genres, keywords, cast, crew, overview)
2. **Text Vectorization**: Converts text data into numerical vectors using CountVectorizer/TfidfVectorizer
3. **Similarity Calculation**: Computes cosine similarity between movie vectors
4. **Recommendation Generation**: Returns top N most similar movies based on similarity scores

### Architecture
```
Movie Data → Feature Engineering → Vectorization → Similarity Matrix → Recommendations
```

## 📋 Requirements

- Python 3.8 or higher
- Required libraries (see requirements.txt)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Download TMDB 5000 Movie Dataset from [Kaggle](https://www.kaggle.com/tmdb/tmdb-movie-metadata)
   - Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the project directory

4. **Run the preprocessing script** (if applicable)
   ```bash
   python preprocess.py
   ```

## 🎯 Usage

### Option 1: Streamlit Web App
```bash
streamlit run app.py
```
Then open your browser at `http://localhost:8501`

### Option 2: Jupyter Notebook
```bash
jupyter notebook movie_recommendation.ipynb
```

### Option 3: Python Script
```python
from movie_recommender import MovieRecommender

# Initialize recommender
recommender = MovieRecommender()

# Get recommendations
movies = recommender.recommend('The Dark Knight', n=5)
print(movies)
```



## 🛠️ Technologies Used

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms (CountVectorizer, cosine_similarity)
- **NLTK**: Natural language processing (stemming)
- **Streamlit**: Web application framework (if applicable)
- **Pickle**: Model serialization
- **Requests**: API calls for movie posters (if applicable)

## 📊 Dataset

This project uses the **TMDB 5000 Movie Dataset** which includes:
- **Movies Dataset**: 4,803 movies with 20 features
  - budget, genres, homepage, id, keywords, original_language, original_title, overview, popularity, production_companies, production_countries, release_date, revenue, runtime, spoken_languages, status, tagline, title, vote_average, vote_count
  
- **Credits Dataset**: Cast and crew information
  - movie_id, title, cast, crew

## 🎨 Feature Engineering

The recommendation system uses the following features:

1. **Genres**: Movie genres (Action, Comedy, Drama, etc.)
2. **Keywords**: Plot keywords and tags
3. **Cast**: Top 3 actors/actresses
4. **Crew**: Director name
5. **Overview**: Movie plot description

These features are combined into a single "tags" column and processed using:
- Text cleaning and lowercasing
- Stemming (reducing words to root form)
- Vectorization (CountVectorizer/TF-IDF)

## 📈 Algorithm Details

### Cosine Similarity

The system uses **cosine similarity** to measure how similar two movies are:

```
similarity = (A · B) / (||A|| × ||B||)
```

Where:
- A and B are movie feature vectors
- Values range from 0 (completely different) to 1 (identical)

### Recommendation Process

```python
1. User selects a movie
2. System retrieves the movie's feature vector
3. Calculates cosine similarity with all other movies
4. Sorts movies by similarity score (descending)
5. Returns top N recommendations (excluding the selected movie)
```

## 🎯 Performance

- **Dataset Size**: 4,803 movies
- **Feature Dimensions**: ~5,000 unique features (after vectorization)
- **Recommendation Time**: < 1 second per query
- **Accuracy**: Based on content similarity (subjective evaluation)

## 🔮 Future Enhancements

- [ ] Implement collaborative filtering
- [ ] Add hybrid recommendation (content + collaborative)
- [ ] Include user ratings and reviews
- [ ] Add filtering by year, genre, rating
- [ ] Implement deep learning models (Neural Collaborative Filtering)
- [ ] Add movie trailer integration
- [ ] Create user accounts for personalized recommendations
- [ ] Deploy on cloud platform (AWS, Heroku, Streamlit Cloud)
- [ ] Add multilingual support
- [ ] Implement A/B testing for recommendations

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## 👤 Author

**SARVESH KUMAR SHUKLA**
- GitHub: [@Prince2720](https://github.com/Prince2720)
- LinkedIn: [Sarvesh Kumar Shukla](https://www.linkedin.com/in/sarvesh-kumar-shukla/)
- Email: princeshukla2720@gmail.com

## 🙏 Acknowledgments

- [TMDB](https://www.themoviedb.org/) for providing the movie dataset
- [Kaggle](https://www.kaggle.com/) for hosting the dataset
- [Streamlit](https://streamlit.io/) for the web framework
- [Scikit-learn](https://scikit-learn.org/) for machine learning tools

## 📚 References

- [Content-Based Recommendation Systems](https://en.wikipedia.org/wiki/Recommender_system#Content-based_filtering)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [TF-IDF Vectorization](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

## 📧 Contact

For any queries or suggestions, feel free to reach out:
- Email: princeshukla2720@gmail.com
- LinkedIn: [Sarvesh Kumar Shukla](https://www.linkedin.com/in/sarvesh-kumar-shukla/)


## ⭐ Show Your Support

Give a ⭐ if you like this project and found it helpful!

---

**Made with ❤️ and Python**