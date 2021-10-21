from app import app
from flask import jsonify


@app.errorhandler(400)
def not_allowed(error):
    return jsonify({
        "message": "error by {}".format(str(error)),
        "error_type": "not allowed",
        "status_code": 400
    })


@app.errorhandler(401)
def unauthorized(error):
    return jsonify({
        "message": "error by {}".format(str(error)),
        "error_type": "unauthorized",
        "status_code": 401
    })


@app.errorhandler(403)
def forbidden(error):
    return jsonify({
        "message": "error by {}".format(str(error)),
        "error_type": "forbidden",
        "status_code": 403
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "message": "error by {}".format(str(error)),
        "error_type": "not found",
        "status_code": 404
    })


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "message": "error by {}".format(str(error)),
        "error_type": "method not allowed",
        "status_code": 405
    })


@app.errorhandler(406)
def not_acceptable(error):
    return jsonify({
        "message": "error by {}".format(str(error)),
        "error_type": "not acceptable",
        "status_code": 406
    })


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        "message": "error by {}".format(str(error)),
        "error_type": "internal server error",
        "status_code": 500
    })
