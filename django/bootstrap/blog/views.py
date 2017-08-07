from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

from .models import Location

def index(request):

	# 중복 뺸 유저아이디만 
	locations = Location.objects.values('member_id').distinct()

	context = {'locations' : locations}

	return render(request, 'blog/index.html', context)


def member(request, member_id):

	locations = Location.objects.filter(member_id = member_id)

	context = {'locations' : locations}

	return render(request, 'blog/member.html', context)