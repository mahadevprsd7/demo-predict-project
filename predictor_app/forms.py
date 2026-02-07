from django import forms

class SalaryForm(forms.Form):
    experience = forms.FloatField(
        label="Years of Experience",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter years of experience'
        })
    )