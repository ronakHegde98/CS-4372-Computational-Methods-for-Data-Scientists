"""


"""

from config.creds import authenticate
import pprint
import time
import preprocessor as p
import json


API = authenticate("creds.ini")
results = API.search(q = "USA -filter:retweets",lang="en", result_type = "recent", count = 50)

print(type(results))
print(type(results[0]))

for result in results:
  print(p.clean(result.text))