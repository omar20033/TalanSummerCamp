from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import tempfile

app = Flask(__name__)

def preprocess_content(text):
    text = text.replace('\n', ' ')
    return text

@app.route('/process_pdf', methods=['POST','GET'])
def process_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({"success": False, "message": "No file part"}), 400
    
    file = request.files['pdf_file']
    
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"}), 400
    
    # Save the file temporarily
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)
    
    try:
        # Load and process the PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Apply preprocessing
        for doc in docs:
            doc.page_content = preprocess_content(doc.page_content)
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=50
        )
        
        # Split documents into chunks
        doc_splits = text_splitter.split_documents(docs)

        # Clean up the temporary file
        os.remove(temp_path)
        
        # Return success response
        return jsonify({"success": True, "message": "PDF processed successfully"})
    
    except Exception as e:
        os.remove(temp_path)
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


