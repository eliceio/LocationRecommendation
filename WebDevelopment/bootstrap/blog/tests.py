from django.test import TestCase

# Create your tests here.
from .models import Location

location = Location.objects.values()
print(location)