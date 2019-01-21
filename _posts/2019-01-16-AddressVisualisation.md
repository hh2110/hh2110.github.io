---
title: "Address Visualisation"
date: 2019-01-16
tags: [data science]
header:
  image: "/images/addressvisualisation/hospital_front.jpg"
excerpt: "Visualisation, Data Science"
mathjax: "true"
---

I volunteer for a charity based in Peshawar Pakistan, called [Abaseen 
Institute for Medical Sciences](http://aimspk.abaseen.com) - AIMS. This 
is a charity that strives to create a quality assured and affordable 
diabetes healthcare institute. I am involved in the IT-based activities, 
like helping with the hospital software or maintaining the website.

The hospital has agreed with the local government to run bi-monthly medical 
camps throughout the city of Peshawar. The city can be split into Union 
Councils, of which there are 92. By analysing where our patients live, we can 
strategically choose which Union councils of the city to visit first, to 
maximise the number of patients visiting from that region.

After isolating the addresses list and cleaning it up, by dealing with blank
elements and characters such as '\r', we can use a library, googlemaps, to 
search for longitude and latitude values for each address (this also requires
and apikey, for more info visit[here](https://github.com/googlemaps/google-maps-services-python):

```python
	import pandas as pd
	import googlemaps

	data = pd.read_csv("./data/CleanAddData.csv", index_col=0)
	with open('api.txt') as f:
		apiKey=f.readline()
	gmaps_key=googlemaps.Client(key = apiKey)
	
	# create new empty columns for google maps search results
	# geocode_object should be included too 
	data["LAT"] = None
	data["LON"] = None
	data["GMAPS_NAME"] = None # result returned after the search
	data["GEOCODE_OBJECT"] = None # object returned afer serch 
	data["PARTIAL_RESULT"] = None # boolean to show perfect or partial match

	# this is the loop that will search for the information on each address
	chosen=range(0,len(data))
	for i in chosen:
	    geocode_result = gmaps_key.geocode(data['pntAddress'][i]+', Pakistan')
	    if i%100==0: # a tracker to see how much is done
	        print(i)
	        data.to_csv('./data/GMapsAddress_'+str(i)+'.csv')
	    try:
	        lat = geocode_result[0]["geometry"]["location"]["lat"]
	        lon = geocode_result[0]["geometry"]["location"]["lng"]
	        gname = geocode_result[0]["formatted_address"]
	        gdata = geocode_result[0]
	        partial = checkPartialMatch(geocode_result) 
	        data.iat[i, data.columns.get_loc("LAT")] = lat
	        data.iat[i, data.columns.get_loc("LON")] = lon
	        data.iat[i, data.columns.get_loc("GMAPS_NAME")] = gname
	        data.iat[i, data.columns.get_loc("GEOCODE_OBJECT")] = gdata
	        data.iat[i, data.columns.get_loc("PARTIAL_RESULT")] = partial
	    except:
	        gname = None
	        lat = None 
	        lon = None
	        print("couldn't find address", i, data['pntAddress'][i])
	
	#save the new dataframe that holds all the information
	from datetime import datetime
	time=datetime.now().strftime('%H%M')
	data.to_csv('./data/GMapsAddress_'+time+'.csv')
```	
	




This is a map of Peshawar:
<img src="{{ site.url }}{{ site.baseurl }}/images/addressvisualisation/peshmap.png" 
alt="Map of Peshawar, KPK, Pakistan">


