from flask_testing import TestCase
from app import app
from flask import current_app, url_for


class MainTest(TestCase):

    def create_app(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        return app

    def test_app_exists(self):
        self.assertIsNotNone(current_app)

    def test_app_in_test_mode(self):
        self.assertTrue(current_app.config["TESTING"])

    """def test_bacteria_growth_get(self):
        response = self.client.get(url_for("bacteria_growth"))
        self.assert200(response)"""

    def test_lact_pred_get(self):
        response = self.client.get(url_for("lact_pred"))
        self.assert200(response)

    def test_list_strep_pred_get(self):
        response = self.client.get(url_for("list_strep_pred"))
        self.assert200(response)

    def test_strep_pred_get(self):
        response = self.client.get(url_for("strep_pred"))
        self.assert200(response)

    def test_classification_get(self):
        response = self.client.get(url_for("classification_single"))
        self.assert200(response)
