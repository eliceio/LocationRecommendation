from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

from .models import Location


def index(request):

	# 중복 뺸 유저아이디만 
	locations = Location.objects.values('member_id', 'member_nickname').distinct()

	context = {'locations' : locations}

	return render(request, 'blog/index.html', context)


# # 이렇게 만들면 현재 로케이션을 넣을 수가 없어져...
# def recommendataion(member_id):

# 	member_id = member_id

# 	# recommendate place list
# 	r_list = ['a','b','c','d','e']

# 	return r_list

def member(request, member_id):

	locations = Location.objects.filter(member_id = member_id)

	# recommend = recommendation(member_id)

	context = {'locations' : locations}

	return render(request, 'blog/member.html', context)