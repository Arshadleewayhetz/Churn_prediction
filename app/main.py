"""
Churn Prediction API
-------------------
FastAPI application for churn prediction inference.

Author: Codegen
Date: 2025-05-26
"""

import logging
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Any, Optional
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta

from app.schemas.input import ChurnPredictionInput, ChurnPredictionBatchInput
from app.schemas.output import ChurnPredictionOutput, ChurnPredictionBatchOutput, ChurnReason
from app.schemas.user import User, UserInDB, Token
from app.models.model import ChurnModel, get_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting customer churn and identifying churn reasons",
    version="1.0.0",
)

# Security
SECRET_KEY = "your-secret-key"  # Change this!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Dummy user database
users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    }
}

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = User(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint to verify the API is running."""
    return {"status": "healthy", "message": "Churn Prediction API is running"}

@app.post("/predict", response_model=ChurnPredictionOutput, tags=["Prediction"])
async def predict_churn(
    input_data: ChurnPredictionInput,
    model: ChurnModel = Depends(get_model)
):
    """
    Predict churn for a single customer.
    
    Returns:
        - Churn probability
        - Binary churn prediction
        - Top churn reasons with impact percentages
    """
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        
        # Make prediction
        prediction_result = model.predict(input_df)
        
        return prediction_result
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=ChurnPredictionBatchOutput, tags=["Prediction"])
async def predict_churn_batch(
    input_data: ChurnPredictionBatchInput,
    model: ChurnModel = Depends(get_model)
):
    """
    Predict churn for multiple customers.
    
    Returns:
        - List of predictions with churn probabilities and reasons
        - Summary statistics
    """
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([item.dict() for item in input_data.customers])
        
        # Make batch predictions
        predictions = []
        for i in range(len(input_df)):
            single_prediction = model.predict(input_df.iloc[[i]])
            predictions.append(single_prediction)
        
        # Calculate summary statistics
        churn_count = sum(1 for p in predictions if p.churn_prediction)
        churn_rate = churn_count / len(predictions) if predictions else 0
        
        # Aggregate top reasons
        all_reasons = []
        for p in predictions:
            all_reasons.extend(p.churn_reasons)
        
        # Count reason frequencies
        reason_counts = {}
        for reason in all_reasons:
            if reason.feature_name in reason_counts:
                reason_counts[reason.feature_name] += 1
            else:
                reason_counts[reason.feature_name] = 1
        
        # Sort by frequency
        top_reasons = [
            ChurnReason(
                feature_name=feature,
                impact_percentage=sum(r.impact_percentage for r in all_reasons if r.feature_name == feature) / reason_counts[feature]
            )
            for feature in sorted(reason_counts, key=reason_counts.get, reverse=True)[:5]
        ]
        
        return ChurnPredictionBatchOutput(
            predictions=predictions,
            total_customers=len(predictions),
            churn_count=churn_count,
            churn_rate=churn_rate,
            top_reasons=top_reasons
        )
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/model/info", tags=["Model"])
async def model_info(model: ChurnModel = Depends(get_model)):
    """Get information about the currently loaded model."""
    return {
        "model_type": model.model_type,
        "features": model.feature_names,
        "model_metrics": model.model_metrics,
        "last_trained": model.last_trained,
    }

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

