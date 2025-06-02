import os
import io
import uuid
import torch
import numpy as np
import cv2
import matplotlib
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from torchvision import transforms
from flask_cors import CORS
# from video_processing import extract_frames_from_video, predict_frame, predict_video
from detector import DeepfakeDetector
from utils import load_model, generate_confidence_graph, get_gradcam_heatmap  # Import Grad-CAM from utils.py
from report_generator import generate_pdf_report

temp_video_path = os.path.join("temp_dir", f"{uuid.uuid4()}.mp4")
cap = cv2.VideoCapture(temp_video_path)

matplotlib.use('Agg')  # Use non-GUI backend for matplotlib

# Flask app setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Directories setup
overlay_images_dir = './overlay_images'
reports_dir = './reports'
graphs_dir = './graphs'
os.makedirs(overlay_images_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepfakeDetector()
model = load_model(model, "../saved_models/deepfake_detector.pth", device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def preprocess_image(pil_image):
    return transform(pil_image).unsqueeze(0).to(device)

def generate_gradcam_map(img_tensor, model):
    # Now using get_gradcam_heatmap from utils.py
    return get_gradcam_heatmap(model, img_tensor)

def overlay_gradcam(pil_image, gradcam_map):
    original = np.array(pil_image)
    original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    gradcam_resized = cv2.resize(gradcam_map, (original.shape[1], original.shape[0]))
    heatmap = np.uint8(255 * gradcam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay)

@app.route('/overlay_images/<path:filename>')
def serve_overlay_images(filename):
    return send_from_directory(overlay_images_dir, filename)

@app.route('/graphs/<path:filename>')
def serve_graphs(filename):
    return send_from_directory(graphs_dir, filename)

@app.route('/reports/<path:filename>')
def serve_report(filename):
    return send_from_directory(reports_dir, filename)



@app.route('/predict-image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    pil_image = Image.open(file.stream).convert('RGB')
    img_tensor = preprocess_image(pil_image)

    with torch.no_grad():
        output = model(img_tensor)
        # Apply sigmoid to logits
        prediction = torch.sigmoid(output)
        label = 'fake' if prediction.item() > 0.5 else 'real'
        confidence = prediction.item()

    gradcam = generate_gradcam_map(img_tensor, model)
    overlay = overlay_gradcam(pil_image, gradcam)

    image_id = uuid.uuid4().hex
    overlay_filename = f'gradcam_{image_id}.png'
    overlay_path = os.path.join(overlay_images_dir, overlay_filename)
    overlay.save(overlay_path)

    graph_filename = f'confidence_graph_{image_id}.png'
    graph_path = os.path.join(graphs_dir, graph_filename)
    frame_ids = [0]
    real_conf = [confidence * 100 if label == 'real' else 0]
    fake_conf = [confidence * 100 if label == 'fake' else 0]
    generate_confidence_graph(frame_ids, real_conf, fake_conf, graph_path)

    explanation = (
        f"The model classified the image as '{label.upper()}' with {round(confidence * 100, 2)}% confidence. "
        "This decision was based on regions highlighted in the Grad-CAM overlay."
    )

    return jsonify({
        'label': label,
        'confidence': round(confidence * 100, 2),
        'overlay_image': f'/overlay_images/{overlay_filename}',
        'graph_image': f'/graphs/{graph_filename}',
        'explanation': explanation
    })
# @app.route('/predict-video', methods=['POST'])
# def predict_video():
#     if 'video' not in request.files:
#         return jsonify({'error': 'No video uploaded'}), 400

#     file = request.files['video']
#     temp_video_path = f'temp_input_{uuid.uuid4().hex}.mp4'
    
#     with open(temp_video_path, 'wb') as f:
#         f.write(file.read())

#     cap = cv2.VideoCapture(temp_video_path)

#     frame_results = []
#     reference_paths, gradcam_paths = [], []
#     real_confidences, fake_confidences, all_confidences = [], [], []

#     frame_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret or frame_count >= 60:
#             break

#         if frame_count % 2 == 0:  # Changed to every 2nd frame
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             pil_frame = Image.fromarray(rgb)
#             img_tensor = preprocess_image(pil_frame)

#             with torch.no_grad():
#                 output = model(img_tensor)
#                 probs = torch.nn.functional.softmax(output[0], dim=0)
#                 label_idx = torch.argmax(probs).item()
#                 label = 'real' if label_idx == 0 else 'fake'
#                 confidence = probs[label_idx].item()

#             all_confidences.append(confidence * 100)
#             real_confidences.append(confidence * 100 if label == 'real' else 0)
#             fake_confidences.append(confidence * 100 if label == 'fake' else 0)

#             gradcam = generate_gradcam_map(img_tensor, model)
#             overlay = overlay_gradcam(pil_frame, gradcam)

#             ref_path = os.path.join(overlay_images_dir, f'ref_{frame_count}.png')
#             gradcam_path = os.path.join(overlay_images_dir, f'gradcam_{frame_count}.png')
#             pil_frame.save(ref_path)
#             overlay.save(gradcam_path)

#             reference_paths.append(ref_path)
#             gradcam_paths.append(gradcam_path)

#             frame_results.append({
#                 'frame': frame_count,
#                 'label': label,
#                 'confidence': round(confidence * 100, 2)
#             })
#         frame_count += 1

#     cap.release()
#     os.remove(temp_video_path)

#     graph_filename = f'confidence_graph_{uuid.uuid4().hex}.png'
#     graph_path = os.path.join(graphs_dir, graph_filename)
#     generate_confidence_graph(
#         frame_ids=[r['frame'] for r in frame_results],
#         real_confidences=real_confidences,
#         fake_confidences=fake_confidences,
#         output_path=graph_path
#     )

#     # Final label by majority voting instead of confidence total
#     real_count = sum(1 for r in frame_results if r['label'] == 'real')
#     fake_count = sum(1 for r in frame_results if r['label'] == 'fake')
#     final_label = 'real' if real_count >= fake_count else 'fake'


#     prediction_explanation = (
#         f"The video is classified as '{final_label}' because the majority of the frames were predicted as '{final_label}'. "
#         f"Average confidence: {round(avg_conf, 2)}%."
#     )

#     graph_explanation = (
#         "The red line (fake) is dominant." if final_label == "fake"
#         else "The green line (real) remains consistently high."
#     )

#     report_filename = f"deepfake_report_{uuid.uuid4().hex}.pdf"
#     report_path = os.path.join(reports_dir, report_filename)

#     generate_pdf_report(
#         report_path=report_path,
#         prediction_label=final_label,
#         confidence=avg_conf,
#         reference_images=reference_paths,
#         gradcam_images=gradcam_paths,
#         graph_image=graph_path,
#         prediction_explanation=prediction_explanation,
#         graph_explanation=graph_explanation
#     )

#     for path in reference_paths + gradcam_paths + [graph_path]:
#         os.remove(path)

#     return jsonify({
#         'label': final_label,
#         'confidence': round(avg_conf, 2),
#         'report': report_path,
#         'frames_analyzed': len(frame_results),
#         'report_path': f'/reports/{report_filename}',
#         'prediction_explanation': prediction_explanation,
#         'graph_explanation': graph_explanation
#         })


@app.route('/predict-video', methods=['POST'])
def predict_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    file = request.files['video']
    temp_video_path = f'temp_input_{uuid.uuid4().hex}.mp4'
    
    with open(temp_video_path, 'wb') as f:
        f.write(file.read())

    cap = cv2.VideoCapture(temp_video_path)
    frame_indices = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 6:
        frame_indices = list(range(total_frames))
    else:
        step = total_frames // 6
        frame_indices = [i * step for i in range(12)]

    frame_results = []
    reference_paths, gradcam_paths = [], []
    real_confidences, fake_confidences, all_confidences = [], [], []

    frame_count = 0
    selected_index = 0

    while True:
        ret, frame = cap.read()
        if not ret or selected_index >= len(frame_indices):
            break

        if frame_count == frame_indices[selected_index]:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb)
            img_tensor = preprocess_image(pil_frame)

            with torch.no_grad():
                output = model(img_tensor)
        # Apply sigmoid to logits
            prediction = torch.sigmoid(output)
            label = 'fake' if prediction.item() > 0.5 else 'real'
            confidence = prediction.item()

            all_confidences.append(confidence )
            real_confidences.append(confidence if label == 'real' else 0)
            fake_confidences.append(confidence  if label == 'fake' else 0)

            gradcam = generate_gradcam_map(img_tensor, model)
            overlay = overlay_gradcam(pil_frame, gradcam)

            ref_path = os.path.join(overlay_images_dir, f'ref_{frame_count}.png')
            gradcam_path = os.path.join(overlay_images_dir, f'gradcam_{frame_count}.png')
            pil_frame.save(ref_path)
            overlay.save(gradcam_path)

            reference_paths.append(ref_path)
            gradcam_paths.append(gradcam_path)

            frame_results.append({
                'frame': frame_count,
                'label': label,
                'confidence': round(confidence *100 , 2)
            })

            selected_index += 1

        frame_count += 1

    cap.release()
    os.remove(temp_video_path)

    # Generate confidence graph
    graph_filename = f'confidence_graph_{uuid.uuid4().hex}.png'
    graph_path = os.path.join(graphs_dir, graph_filename)
    generate_confidence_graph(
        frame_ids=[r['frame'] for r in frame_results],
        real_confidences=real_confidences,
        fake_confidences=fake_confidences,
        output_path=graph_path
    )

    # Final label by majority voting
    real_count = sum(1 for r in frame_results if r['label'] == 'real')
    fake_count = sum(1 for r in frame_results if r['label'] == 'fake')
    final_label = 'real' if real_count <= fake_count else 'fake'

    # Average confidence of predicted frames
    avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0

    # Prediction explanation
    prediction_explanation = (
        f"The video is classified as '{final_label}' because the majority of the frames were predicted as '{final_label}'. "
        f"Average confidence: {round(avg_conf, 2)}%."
    )

    # Graph explanation
    graph_explanation = (
        "The red line (fake) is dominant." if final_label == "fake"
        else "The green line (real) remains consistently high."
    )

    # Generate PDF report
    report_filename = f"deepfake_report_{uuid.uuid4().hex}.pdf"
    report_path = os.path.join(reports_dir, report_filename)
    generate_pdf_report(
        report_path=report_path,
        prediction_label=final_label,
        confidence=avg_conf,
        reference_images=reference_paths,
        gradcam_images=gradcam_paths,
        graph_image=graph_path,
        prediction_explanation=prediction_explanation,
        graph_explanation=graph_explanation
    )

    # Clean up generated files
    for path in reference_paths + gradcam_paths + [graph_path]:
        os.remove(path)

    return jsonify({
        'label': final_label,
        'confidence': round(avg_conf, 2),
        'report': report_path,
        'frames_analyzed': len(frame_results),
        'report_path': f'/reports/{report_filename}',
        'prediction_explanation': prediction_explanation,
        'graph_explanation': graph_explanation
    })

if __name__ == '__main__':
    app.run(debug=True)