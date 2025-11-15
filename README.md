

✔ How to build & run the Docker image
✔ Instructions for docker-compose
✔ Example GET /summary and POST /inference requests
✔ Notes about x86 architecture
✔ Directory expectations
✔ How to test using the provided grader

Copy/paste directly into your project’s README.md.

---

# **Hurricane Harvey Damage Classification – Inference Server README**

## **Overview**

This repository contains an inference server for the Hurricane Harvey building-damage classification model developed in Project 3. The server loads a trained TensorFlow model and exposes two HTTP endpoints:

* **GET /summary** – returns metadata about the deployed model
* **POST /inference** – accepts a raw image (binary payload) and returns a JSON prediction:

  ```json
  { "prediction": "damage" } 
  ```

  or

  ```json
  { "prediction": "no_damage" }
  ```

The server runs inside a Docker container using **x86 architecture**, as required by the assignment.

---

## **Project Structure**

Your directory should look like this:

```
harvey_destruction/
│
├── app.py
├── requirements_api.txt
├── Dockerfile
├── docker-compose.yml
├── best_hurricane_model.h5
├── best_hurricane_model.keras
├── best_model_info.json
└── data/
      ├── damage/
      └── no_damage/
```

The server expects the model files to be in the same directory as `app.py`.

---

## **1. Building the Docker Image**

Make sure you are on the VM (x86 architecture) before building:

```bash
cd ~/nb-data/harvey_destruction
docker compose build
```

This builds an image named:

```
harvey_destruction-harvey-inference:latest
```

---

## **2. Running the Server Using docker-compose**

Start the inference server:

```bash
docker compose up
```

You should see logs such as:

```
Loading model...
Model loaded successfully
Running on http://0.0.0.0:5000
```

Stop the server with:

```bash
docker compose down
```

---

## **3. Endpoints**

### **GET /summary**

Returns model metadata (architecture, input shape, preprocessing, etc.)

Example request:

```bash
curl http://localhost:5000/summary
```

Example response:

```json
{
  "model_name": "Alternate_LeNet5",
  "input_shape": [128, 128, 3],
  "preprocessing": {
    "resize": [128, 128],
    "scale": 0.00392156862
  },
  "prediction_classes": ["no_damage", "damage"],
  "test_accuracy": 0.8668
}
```

---

### **POST /inference**

You must send the **raw binary image payload**.

Example request (from the project data folder):

```bash
curl -X POST \
     --data-binary "@data/damage/-93.66109_30.212114.jpeg" \
     http://localhost:5000/inference
```

Example response:

```json
{ "prediction": "damage", "probability": 0.9873 }
```

Example with no_damage image:

```bash
curl -X POST \
     --data-binary "@data/no_damage/-95.6302_29.768889.jpeg" \
     http://localhost:5000/inference
```

---

## **4. Running the Official Grader**

The project provides an automated grader to validate:

✔ Correct server format
✔ Correct JSON structure
✔ Correct predictions
✔ End-to-end functionality

Use the included `start_grader.sh` script:

```bash
chmod +x start_grader.sh
./start_grader.sh
```

If everything is correct, you will see:

```
GET /summary format correct
POST /inference format correct
Final results:
Total correct: 6
Accuracy: 1.0
```

---

## **5. Pushing the Docker Image to Docker Hub (Required)**

1. Log in:

```bash
docker login
```

2. Tag your built image:

```bash
docker tag harvey_destruction-harvey-inference <your-dockerhub-username>/harvey-inference:latest
```

3. Push it:

```bash
docker push <your-dockerhub-username>/harvey-inference:latest
```

---

## **6. Notes About Architecture**

* **Important:** The image must be built on the class VM because it uses **x86 architecture**.
* Images built on Apple Silicon (M1/M2/M3) will be ARM-based and **will not work** for grading.

---

## **7. Example Usage Summary**

Once the server is running:

**Summary:**

```bash
curl http://localhost:5000/summary
```

**Inference (damage):**

```bash
curl -X POST --data-binary "@data/damage/example.jpeg" http://localhost:5000/inference
```

**Inference (no_damage):**

```bash
curl -X POST --data-binary "@data/no_damage/example.jpeg" http://localhost:5000/inference
```

---

## **8. Troubleshooting**

* If you see `Permission denied` when running the grader:

```bash
chmod +x start_grader.sh
```

* If Docker says it cannot bind to port 5000, stop any old containers:

```bash
docker ps
docker stop <container-id>
```

* If inference returns 413 errors, the server was not configured properly—use raw binary uploads exactly as shown.

---
