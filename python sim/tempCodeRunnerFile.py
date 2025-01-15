dt = time_end - time_start
print(f" tooK : {int(dt/3600)%60} hours. {int(dt/60)%60} min. {dt%60} sec.")