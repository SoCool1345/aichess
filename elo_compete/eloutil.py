import elo

from elo import Rating,rate_1vs1

r1 = Rating(100)
r2 = Rating(5)
print(rate_1vs1(r1,r2,drawn=True))
print(elo.adjust_1vs1(r1,r2))
print(elo.quality_1vs1(r1,r2))