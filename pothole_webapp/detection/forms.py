from django import forms

class UploadImageForm(forms.Form):
    image = forms.ImageField()

class UploadVideoForm(forms.Form):
    video = forms.FileField()
