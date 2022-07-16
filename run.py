
from flask import Flask, request, jsonify
import werkzeug
from human_action_reg import *
app = Flask(__name__)

@app.route('/upload', methods=["POST"])
def upload():
    if request.method == "POST" :
        file = request.files['file']
        filename = werkzeug.utils.secure_filename(file.filename)
        print("\nReceived File name : " + file.filename)
        file.save("./uploadedFiles/" + filename)

        #human_recog(filename)

        return jsonify({
            "message": "File Uploaded Successfully ",
        })

if __name__ == "__main__":
    app.run(debug=True, port=9000)