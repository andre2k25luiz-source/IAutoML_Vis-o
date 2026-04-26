from flask import Blueprint, jsonify
from services.metrics_services import get_metrics, get_metrics_plot

metrics_routes = Blueprint("metrics", __name__)


# =========================
# METRICS
# =========================
@metrics_routes.route('/metrics', methods=['GET'])
def metrics_route():
    return jsonify(get_metrics())


# =========================
# METRICS PLOT
# =========================
@metrics_routes.route('/metrics_plot', methods=['GET'])
def metrics_plot_route():
    return jsonify(get_metrics_plot())