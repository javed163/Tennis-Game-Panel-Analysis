Padel Analytics — Deployment Guide
How to deploy the pipeline as a live API on 4 different platforms.

Option 1 — Run locally (quickest)
# From project root
pip install fastapi uvicorn python-multipart
uvicorn deploy.app:app --host 0.0.0.0 --port 8000 --reload
Open http://localhost:8000/docs — interactive Swagger UI loads automatically.

Test with curl:

curl -X POST http://localhost:8000/analyze \
  -F "file=@data/raw/match.mp4" \
  -F "max_frames=300"
Option 2 — Docker (recommended for any cloud)
Build image
# From project root
docker build -t padel-analytics -f deploy/Dockerfile .
Run container
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  padel-analytics
Push to Docker Hub
docker tag padel-analytics YOUR_DOCKERHUB_USERNAME/padel-analytics:latest
docker push YOUR_DOCKERHUB_USERNAME/padel-analytics:latest
Option 3 — Render.com (free tier, no credit card)
Push your code to GitHub
Go to https://render.com → New → Web Service
Connect your GitHub repo
Set these values:
Field	Value
Name	padel-analytics
Environment	Python 3
Build Command	pip install -r requirements.txt -r deploy/requirements_deploy.txt && pip install -e .
Start Command	uvicorn deploy.app:app --host 0.0.0.0 --port $PORT
Instance Type	Free
Click Deploy — Render gives you a public URL automatically.
Note: Free tier spins down after 15 minutes of inactivity. First request after sleep takes ~30 seconds to wake up.

Option 4 — Hugging Face Spaces (best for demo / sharing)
Hugging Face Spaces supports FastAPI apps natively.

Step 1 — Create a Space
Go to https://huggingface.co/spaces
Click New Space
Name: padel-analytics
SDK: Docker
Visibility: Public (or Private)
Step 2 — Push your code
# Install Git LFS for large model files
git lfs install

# Clone the Space repo
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/padel-analytics

# Copy project files into it
cp -r src/ deploy/ main.py requirements.txt setup.py \
    huggingface.co/spaces/YOUR_HF_USERNAME/padel-analytics/

# Add the Dockerfile (HF uses it automatically)
cp deploy/Dockerfile padel-analytics/Dockerfile

cd padel-analytics/
git add .
git commit -m "Initial deploy"
git push
Step 3 — Space builds automatically
HF builds the Docker image and gives you a URL like: https://your-username-padel-analytics.hf.space

Step 4 — Test it
Visit https://your-username-padel-analytics.hf.space/docs

Option 5 — Railway.app
Install Railway CLI: npm install -g @railway/cli
Login: railway login
From project root:
railway init
railway up
Railway auto-detects the Dockerfile and deploys it.
Get your URL: railway open
API Endpoints
Once deployed, your API has these endpoints:

Method	URL	Description
GET	/	Redirects to Swagger docs
GET	/health	Health check
POST	/analyze	Upload video → start analysis
GET	/results/{job_id}	Poll status / get results
GET	/results/{job_id}/shots.json	Download shots JSON
GET	/results/{job_id}/shots.csv	Download shots CSV
GET	/jobs	List all jobs
How to use the API
Step 1 — Upload video
import requests

with open("match.mp4", "rb") as f:
    response = requests.post(
        "https://YOUR_DEPLOY_URL/analyze",
        files={"file": f},
        data={"max_frames": 300},
    )

job = response.json()
job_id = job["job_id"]
print(f"Job ID: {job_id}")
Step 2 — Poll for results
import time

while True:
    r = requests.get(f"https://YOUR_DEPLOY_URL/results/{job_id}")
    data = r.json()
    print(f"Status: {data['status']}")

    if data["status"] == "done":
        print(f"Total shots: {data['total_shots']}")
        print(f"Shots: {data['shots'][:3]}")
        break
    elif data["status"] == "error":
        print(f"Error: {data.get('detail')}")
        break

    time.sleep(3)
Step 3 — Download files
# Download shots.csv
csv_data = requests.get(
    f"https://YOUR_DEPLOY_URL/results/{job_id}/shots.csv"
)
with open("shots.csv", "wb") as f:
    f.write(csv_data.content)
print("Saved shots.csv")
Model files (Google Drive upload)
After training the custom classifier (notebook 02_model_training.ipynb), upload these two files to Google Drive and paste the links in README.md:

File	Path
shot_clf.pkl	models/classification/shot_clf.pkl
label_encoder.pkl	models/classification/label_encoder.pkl
yolov8n.pt	Auto-downloaded — no upload needed
How to make a Google Drive link public
Right-click the file → Share
Change to "Anyone with the link"
Copy link
Paste in README under Model links section
Environment variables (for production)
Create a .env file in the project root:

YOLO_DEVICE=cpu
YOLO_CONF_THRESHOLD=0.4
SHOT_VELOCITY_THRESHOLD=8.0
LOG_LEVEL=INFO
These override config.py values in production.