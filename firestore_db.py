import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
cred = credentials.Certificate("trek-ai-firebase-adminsdk-m8l32-db5180792f.json")
firebase_admin.initialize_app(cred)

db = firestore.client()