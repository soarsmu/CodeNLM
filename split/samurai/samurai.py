# Ronin
from spiral import ronin
print("The results of ronin")
for s in [ 'mStartCData', 'nonnegativedecimaltype', 'getUtf8Octets', 'GPSmodule', 'savefileas', 'nbrOfbugs']:
    print(ronin.split(s))

# Samurai
print("The results of samurai")
from spiral import samurai
for s in [ 'mStartCData', 'nonnegativedecimaltype', 'getUtf8Octets', 'GPSmodule', 'savefileas', 'nbrOfbugs']:
    print(samurai.split(s))
