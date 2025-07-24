from flask import Flask, render_template, request, jsonify, send_from_directory
import os

from services.file_processor import process_pdf
from services.summarizer import generate_summary
from services.rag import ask_question

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Создаем папку для загрузок
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Загрузка PDF файла"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Invalid file type"}), 400
    
    try:
        # Сохраняем файл
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Обрабатываем PDF
        doc_id = process_pdf(file_path)
        
        return jsonify({
            "status": "success",
            "filename": file.filename,
            "doc_id": doc_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/summarize/<doc_id>', methods=['GET'])
def summarize(doc_id):
    """Суммаризация документа"""
    try:
        summary = generate_summary(doc_id)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    """Ответ на вопрос по документам"""
    data = request.json
    question = data.get('question')
    doc_id = data.get('doc_id')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        answer = ask_question(question, doc_id)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Доступ к загруженным файлам"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)