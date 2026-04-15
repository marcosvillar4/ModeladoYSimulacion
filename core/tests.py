from django.test import TestCase
from django.urls import reverse


class CoreViewsTest(TestCase):
    def test_home_page_is_available(self):
        response = self.client.get(reverse("core:home"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Plataforma de simulaciones")

    def test_about_page_is_available(self):
        response = self.client.get(reverse("core:about"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "MVP")
