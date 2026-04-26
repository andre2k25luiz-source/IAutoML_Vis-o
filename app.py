import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from routes.main_routes import main_routes
from routes.train_routes import train_routes
from routes.predict_routes import predict_routes
from routes.metrics_routes import metrics_routes

app = Flask(__name__)

app.register_blueprint(main_routes)
app.register_blueprint(train_routes)
app.register_blueprint(predict_routes)
app.register_blueprint(metrics_routes)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)