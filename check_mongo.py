# check_mongo.py
from pymongo import MongoClient
import os
import json

MONGO_CONNECTION_STRING = os.environ.get('MONGO_CONNECTION_STRING')
DB_NAME = os.environ.get('MONGODB_DATABASE_NAME')
COLLECTION_NAME = os.environ.get('MONGODB_COLLECTION_NAME')

if not MONGO_CONNECTION_STRING:
    print('❌ Connection String ENV variable not set.')
    exit(1)

try:
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Check for authentication success
    client.admin.command('ismaster')

    count = collection.count_documents({})

    print(f'✅ Connection Successful. DB: {DB_NAME}, Collection: {COLLECTION_NAME}')
    print(f'💬 Documents found in collection: {count}')

    if count > 0:
        document = collection.find_one({})
        print(f'📝 Example Document (First Message): {json.dumps(document, indent=2, default=str)}')
    else:
        print('⚠️ No chat history documents found.')

except Exception as e:
    if 'Authentication failed' in str(e) or 'not authorized' in str(e):
        print(f'❌ Authentication Failed. Error: {e}')
    else:
        print(f'❌ Unknown Connection/Query Failed: {e}')
        exit(1)