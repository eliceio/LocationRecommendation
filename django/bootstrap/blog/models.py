from django.db import models

# Create your models here.
class Location(models.Model):
	member_id = models.IntegerField()
	member_nickname = models.CharField(max_length=10)
	rating = models.IntegerField()
	restaurant_address = models.CharField(max_length=30)
	restaurant_id = models.IntegerField()
	restaurant_latitude = models.FloatField()
	restaurant_longitude = models.FloatField()
	restaurant_name = models.CharField(max_length=20)
	restaurant_code = models.IntegerField()
	restaurant_subcode = models.IntegerField()

	def __str__(self):
		#return str(self.member_id) + ":" + str(self.restaurant_id)
		return str(self.member_id)

