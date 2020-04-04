
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators import csrf
import os

IMAGE_DIR='./project_web/pictures'

def upload(request):
    context = {}
    if request.POST:
        img_file = request.FILES.get("image")
        img_name = img_file.name
 
        f = open(os.path.join(IMAGE_DIR,img_name), 'wb')
        for chunk in img_file.chunks(chunk_size=1024):
            f.write(chunk)
        
        # call image captioning machine
        context['rlt'] = 'this is a image captioning result'
    return render(request, 'upload.html', context)