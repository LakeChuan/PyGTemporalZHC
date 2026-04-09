with open("cq_rain_rate.asc", "w") as f:
    f.write("# CQ urban pluvial flood test - rainfall rate (test intensity)\n")
    f.write("10  seconds\n")
    for i in range(1170):  # 39*30=1170
        f.write(f"864000  {i}\n")