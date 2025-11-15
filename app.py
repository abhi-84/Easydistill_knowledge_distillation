import subprocess
import os
import sys
import threading
import time
import queue
import io
import contextlib
import glob
import json
# Add this import at the top of your file
from werkzeug.utils import secure_filename

from flask import Flask, render_template, request, Response, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Global Variables for State Management ---
app = Flask(__name__)
CORS(app)
training_thread = None
stop_event = threading.Event()
model = None # Global variable to hold the loaded model

# --- Utility Function to Find Latest Model ---
def find_latest_model_path():
    """Finds the directory of the latest trained model."""
    search_path = os.path.join(os.getcwd(), 'result_stage1', '*')
    list_of_dirs = glob.glob(search_path)
    if not list_of_dirs:
        return None
    latest_dir = max(list_of_dirs, key=os.path.getmtime)
    return latest_dir

# --- Training Logic (in a separate thread) ---
def run_training_in_thread(train_script_path, config_file_path, output_queue):
    """Executes the training script and puts output in a queue."""
    command = [
        sys.executable,
        train_script_path,
        "--config", config_file_path
    ]

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        while process.poll() is None and not stop_event.is_set():
            line = process.stdout.readline()
            if line:
                output_queue.put(f"data: {line}\n")
            time.sleep(0.1) # Prevents high CPU usage

        if stop_event.is_set():
            process.terminate()
            output_queue.put("data: --- TRAINING INTERRUPTED ---\n\n")
            output_queue.put("data: --- END OF STREAM ---\n\n")
        else:
            process.wait()
            output_queue.put("data: --- TRAINING COMPLETE ---\n\n")
            output_queue.put("data: --- END OF STREAM ---\n\n")


    except Exception as e:
        output_queue.put(f"data: Error: {e}\n\n")
    finally:
        output_queue.put("data: --- END OF STREAM ---\n\n")

# --- Flask Routes ---
@app.route('/')
def landing_page():
    """Renders the landing page with stage buttons."""
    return render_template('landing.html')

# Existing route, renamed to /stage1
@app.route('/stage1')
def stage1():
    """Renders the Stage 1 training web page."""
    return render_template('index.html')

# New route for Stage 2
@app.route('/stage2')
def stage2():
    """Renders the Stage 2 training web page."""
    # Assuming you will have a separate HTML file for Stage 2
    return render_template('index2.html')


@app.route('/start_training', methods=['POST'])
def start_training():
    global training_thread, stop_event, training_output_queue

    if training_thread and training_thread.is_alive():
        return jsonify({"status": "error", "message": "Training already in progress."}), 409

    data = request.json or {}
    dataset_filename = data.get("dataset")  # just the filename, not full path

    # Hardcoded dataset base path
    DATASET_BASE_PATH = "/home/rakeshl/abhishek/your_project/datasets"
    dataset_path = os.path.join(DATASET_BASE_PATH, dataset_filename) if dataset_filename else None

    config_file_path = os.path.join(
        os.getcwd(),
        "easydistill", "recipes", "distilqwen_series", "distillqwen2.5", "distilqwen2.5_stage1.json"
    )

    # Check files
    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({"status": "error", "message": f"Dataset not found: {dataset_path}"}), 400
    if not os.path.exists(config_file_path):
        return jsonify({"status": "error", "message": f"Config not found: {config_file_path}"}), 500

    # Update config
    try:
        import json
        with open(config_file_path, "r") as f:
            config = json.load(f)
        config["dataset"]["labeled_path"] = dataset_path
        with open(config_file_path, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to update config: {e}"}), 500

    # Start training
    project_root = os.getcwd()
    train_script_path = os.path.join(project_root, 'easydistill', 'easydistill', 'kd', 'train.py')
    stop_event.clear()
    training_output_queue = queue.Queue()
    training_thread = threading.Thread(
        target=run_training_in_thread,
        args=(train_script_path, config_file_path, training_output_queue)
    )
    training_thread.daemon = True
    training_thread.start()

    return jsonify({"status": "success", "message": "Training started successfully."})

'''@app.route('/stream_training_output')
def stream_training_output():
    """Streams the output from the training process."""
    def generate():
        while True:
            try:
                line = training_output_queue.get(timeout=1)
                yield line
                if line == "data: --- END OF STREAM ---\n\n":
                    break
            except queue.Empty:
                if training_thread and not training_thread.is_alive():
                    yield "data: --- END OF STREAM ---\n\n"
                    break
                continue
    return Response(generate(), mimetype='text/event-stream')
'''
@app.route("/stream_training_output")
def stream_training_output():
    def generate():
        try:
            log_path = "training.log"  # adjust as per your actual path
            with open(log_path, "r") as f:
                f.seek(0, 2)  # Move to end of file
                while True:
                    line = f.readline()
                    if not line:
                        time.sleep(1)
                        continue
                    yield f"data: {line}\n\n"
        except GeneratorExit:
            # Client disconnected — normal behavior
            print("[INFO] Stream connection closed by client")
        except Exception as e:
            # When any error occurs during streaming
            print(f"[ERROR] Stream error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"
        finally:
            # Always end stream properly
            yield "data: [END]\n\n"

    return Response(generate(), mimetype="text/event-stream")


# Add this new route to handle file uploads
@app.route('/upload_file', methods=['POST'])
def upload_file():
    """Endpoint to handle dataset file uploads."""
    # Check if the 'file' part is in the request
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request."}), 400

    file = request.files['file']

    # Check if a file was actually selected
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file."}), 400

    if file:
        # Create a secure filename to prevent malicious directory traversal
        filename = secure_filename(file.filename)
        
        # Define a directory to save the uploaded files
        upload_folder = os.path.join(os.getcwd(), 'easydistill', 'datasets')
        # Create the directory if it doesn't exist
        os.makedirs(upload_folder, exist_ok=True)
        
        # Save the file to the new folder
        file.save(os.path.join(upload_folder, filename))
        
        return jsonify({"status": "success", "message": f"File '{filename}' uploaded successfully."})

    return jsonify({"status": "error", "message": "An unknown error occurred."}), 500



@app.route('/stop_training', methods=['POST'])
def stop_training():
    global stop_event, training_output_queue

    if training_thread and training_thread.is_alive():
        stop_event.set()
        # Notify the stream generator immediately
        if training_output_queue:
            training_output_queue.put("data: --- TRAINING STOPPED BY USER ---\n\n")
            training_output_queue.put("data: --- END OF STREAM ---\n\n")
        return jsonify({"status": "success", "message": "Training is stopped..."})
    else:
        return jsonify({"status": "error", "message": "No training process is currently running."}), 400


@app.route('/load_model', methods=['POST'])
def load_model():
    """Loads the latest trained model into memory."""
    global model, tokenizer
    try:
        model_path = find_latest_model_path()
        if not model_path:
            return jsonify({"status": "error", "message": "No trained model found."}), 404

        # Assume the model and tokenizer can be loaded with Auto... methods
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        return jsonify({"status": "success", "message": f"Model loaded from {model_path}"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to load model: {e}"}), 500

@app.route('/converse', methods=['POST'])
def converse():
    """Endpoint for conversation with the loaded model."""
    global model, tokenizer
    if model is None:
        return jsonify({"status": "error", "message": "No model loaded. Please load a model first."}), 400
    
    data = request.json
    user_question = data.get('question')

    try:
        inputs = tokenizer(user_question, return_tensors='pt')
        outputs = model.generate(
            **inputs,
            max_length=100,
            temperature=0.7,
            do_sample=True
        )
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"status": "success", "response": response_text})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error generating response: {e}"}), 500

if __name__ == '__main__':
    # Adjust the Python path to include the correct easydistill directory
    sys.path.append(os.path.join(os.getcwd(), 'easydistill'))
#    app.run(debug=True)
    app.run(host="0.0.0.0", port=5000, debug=True)

