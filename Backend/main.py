from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
import re
from model_utils import preprocess_text, TfidfVectorizer, ChiSquareSelector, MultiClassSVM, stop_words, stemmer, JUNK_TOKENS, nlp

app = FastAPI(title="Topic Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorizer = None
selector = None
svm_model = None

def load_models():
    global vectorizer, selector, svm_model
    try:
        if os.path.exists('vectorizer.joblib') and os.path.exists('selector.joblib') and os.path.exists('svm_model.joblib'):
            vectorizer = joblib.load('vectorizer.joblib')
            selector = joblib.load('selector.joblib')
            svm_model = joblib.load('svm_model.joblib')
            print("Model artifacts loaded successfully")
        else:
            print("Model files not found. Please verify that the joblib files exist.")
    except Exception as e:
        print(f"Error loading model: {e}")

load_models()

class PredictionRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "online", "model_loaded": svm_model is not None}

@app.post("/predict")
async def predict(request: PredictionRequest):
    if svm_model is None or vectorizer is None or selector is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")
    
    try:
        original_text = request.text
        tokens = preprocess_text(original_text)
        
        # Word alignment for preprocessing visualization using Spacy
        doc = nlp(original_text)
        person_tokens = set()
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                for token in ent:
                    person_tokens.add(token.text)
                    
        aligned_words = []
        prev_person = False
        
        for token in doc:
            text_str = token.text
            
            # Check if space or punctuation
            if token.is_space or token.is_punct:
                aligned_words.append({
                    "text": text_str,
                    "is_word": False,
                    "status": "ignored",
                    "reason": "space/punctuation"
                })
                continue
                
            is_person = (
                token.ent_type_ == "PERSON"
                or token.text in person_tokens
                or "'" in token.text
            )
            
            if is_person:
                if not prev_person:
                    aligned_words.append({
                        "text": text_str,
                        "is_word": True,
                        "status": "replaced",
                        "replacement": "PERSON",
                        "reason": "person entity"
                    })
                else:
                    aligned_words.append({
                        "text": text_str,
                        "is_word": True,
                        "status": "removed",
                        "reason": "merged into PERSON"
                    })
                prev_person = True
                continue
            prev_person = False
            
            CURRENCY_SYMBOLS = {"$", "£", "€"}
            MAGNITUDE_SUFFIXES = {"m", "bn", "b", "k"}
            
            if text_str in CURRENCY_SYMBOLS:
                aligned_words.append({
                    "text": text_str,
                    "is_word": True,
                    "status": "replaced",
                    "replacement": "MONEY",
                    "reason": "currency symbol"
                })
                continue
                
            if re.match(r"^\d+(?:\.\d+)?(?:bn|m|b|k)$", text_str, re.I):
                aligned_words.append({
                    "text": text_str,
                    "is_word": True,
                    "status": "replaced",
                    "replacement": "NUM",
                    "reason": "magnitude number"
                })
                continue
                
            if text_str.lower() in MAGNITUDE_SUFFIXES:
                aligned_words.append({
                    "text": text_str,
                    "is_word": True,
                    "status": "removed",
                    "reason": "magnitude suffix"
                })
                continue
                
            if token.like_num:
                aligned_words.append({
                    "text": text_str,
                    "is_word": True,
                    "status": "replaced",
                    "replacement": "NUM",
                    "reason": "numeric token"
                })
                continue
                
            lemma = token.lemma_.strip().lower()
            if not lemma:
                aligned_words.append({
                    "text": text_str,
                    "is_word": True,
                    "status": "removed",
                    "reason": "empty lemma"
                })
                continue
            if lemma in JUNK_TOKENS:
                aligned_words.append({
                    "text": text_str,
                    "is_word": True,
                    "status": "removed",
                    "reason": "junk token"
                })
                continue
            if lemma in stop_words:
                aligned_words.append({
                    "text": text_str,
                    "is_word": True,
                    "status": "removed",
                    "reason": "stopword"
                })
                continue
            if not re.search(r"[a-z0-9]", lemma):
                aligned_words.append({
                    "text": text_str,
                    "is_word": True,
                    "status": "removed",
                    "reason": "non-alphanumeric"
                })
                continue
            if len(lemma) < 2:
                aligned_words.append({
                    "text": text_str,
                    "is_word": True,
                    "status": "removed",
                    "reason": "too short"
                })
                continue
                
            aligned_words.append({
                "text": text_str,
                "is_word": True,
                "status": "kept",
                "processed": lemma
            })

        # TF-IDF Vectorization (raw document vectorization)
        tfidf_vec = vectorizer.transform([original_text])
        
        # Tokenizing & Vocabulary Alignment (using build_analyzer to get ngrams)
        analyzer = vectorizer.build_analyzer()
        raw_ngrams = analyzer(original_text)
        
        tokenizing_content = []
        active_features = {}
        
        for ng in raw_ngrams:
            if ng in vectorizer.vocabulary_:
                col = vectorizer.vocabulary_[ng]
                weight = float(tfidf_vec[0, col])
                in_vocab = True
                if weight > 0:
                    active_features[ng] = {
                        "vocab_idx": col,
                        "weight": weight,
                        "chi_score": float(selector.scores_[col]),
                        "selected": bool(col in selector.selected_features_)
                    }
            else:
                weight = 0.0
                in_vocab = False
                
            tokenizing_content.append({
                "ngram": ng,
                "weight": weight,
                "in_vocab": in_vocab
            })
            
        # Feature selection step
        chi_vec = selector.transform(tfidf_vec)
        
        # Sort and gather active features
        sorted_active_features = []
        for ngram, info in active_features.items():
            sorted_active_features.append({
                "ngram": ngram,
                "weight": info["weight"],
                "chi_score": info["chi_score"],
                "selected": info["selected"]
            })
        # Sort by weight or chi_score
        sorted_active_features = sorted(sorted_active_features, key=lambda x: x["chi_score"], reverse=True)

        # Classification Inference
        probs = svm_model.predict_confidence(chi_vec)[0]
        class_name = str(svm_model.predict(chi_vec)[0])
        score = float(np.max(probs))
        class_id = int(np.where(svm_model.classes_ == class_name)[0][0])
        
        classes_data = []
        for c, p in zip(svm_model.classes_, probs):
            classes_data.append({
                "label": str(c),
                "confidence": float(p)
            })
            
        classes_data = sorted(classes_data, key=lambda x: x["confidence"], reverse=True)
        
        return {
            "prediction": {
                "label": class_name,
                "class_id": class_id,
                "score": score,
                "classes": classes_data
            },
            "steps": [
                {
                    "name": "Original",
                    "content": original_text,
                    "type": "text"
                },
                {
                    "name": "Preprocessing",
                    "content": " ".join(tokens),
                    "aligned_words": aligned_words,
                    "type": "preprocessing"
                },
                {
                    "name": "Tokenizing",
                    "content": tokenizing_content,
                    "type": "tokenizing"
                },
                {
                    "name": "Selecting (Chi-Square)",
                    "content": sorted_active_features,
                    "type": "selector"
                }
            ]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
