from django.db import models

class Prediction(models.Model):

    experience = models.FloatField()
    predicted_salary = models.FloatField()

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.experience} yrs → ₹{self.predicted_salary}"