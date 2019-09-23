from datetime import timedelta, date
import random

def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

start_dt = date(2000, 1, 1)
end_dt = date(2050, 1, 1)
with open("date_time.txt", 'w') as f:
    for i in range(100):
        for dt in daterange(start_dt, end_dt):
            f.write("{},{},{},{},{},{}\n".format(random.randint(0, 23), random.randint(0, 59), random.randint(0, 59), dt.year, dt.month, dt.day))