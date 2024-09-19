from django.shortcuts import render
import requests

def upload_view(request):
    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']
        files = {'file': file}
        response = requests.post('http://localhost:8000/api/transcribe', files=files)
        transcription = response.json().get('transcription', '')
        return render(request, 'result.html', {'transcription': transcription})
    return render(request, 'upload.html')
