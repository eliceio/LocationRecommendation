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
	register_time = models.DateField()

	def __str__(self):
		#return str(self.member_id) + ":" + str(self.restaurant_id)
		return str(self.member_id)

# class Recommendation(models.Model):
# 	user_number = models.ForeignKey(Location)
# 	place_1 = models.CharField(max_length=20)
# 	place_2 = models.CharField(max_length=20)
# 	place_3 = models.CharField(max_length=20)
# 	place_4 = models.CharField(max_length=20)
# 	place_5 = models.CharField(max_length=20)

# 	def __str__(self):
# 		return str(self.user_number)


