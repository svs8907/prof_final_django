from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm
from django import forms
from django.contrib.auth.models import User

class CreateUserForm(UserCreationForm):
    phone_number = forms.CharField(max_length=15, required=True, help_text='Enter your phone number')
    class Meta:
        model = User
        fields = ["username", "email", "password1", "password2","phone_number"]
        
    def save(self, commit=True):
        user = super().save(commit=False)
        user.phone_number = self.cleaned_data['phone_number']  # Storing phone number
        if commit:
            user.save()
        return user    