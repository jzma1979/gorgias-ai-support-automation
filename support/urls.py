from django.urls import path

from support.views import gorgias_webhook, health

urlpatterns = [
    path("health/", health, name="health"),
    path("api/webhooks/gorgias/", gorgias_webhook, name="gorgias-webhook"),
]
