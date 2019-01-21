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
an apikey, for more info visit [here](https://github.com/googlemaps/google-maps-services-python)):

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

Then we can use a really cool library called folium to plot a heatmap of the 
lat and lon values superposed onto a map of Peshawar:

```python
	import folium
	from folium.yyplugins import HeatMap
	
	# create dataframe
	data = pd.read_csv('./data/GMapsAddressesAccurate_1848.csv', index_col=0)

	# isloate the lon and lat data
	latList=data["LAT"].values;
	lonList=data["LON"].values;

	# remove any None values for where lon and lat data could not be found
	latList=latList[latList != np.array(None)]
	lonList=lonList[lonList != np.array(None)]
	
	# use folium to add heat map to map of peshawar 
	hmap = folium.Map(location=[33.99, 71.52], zoom_start=12, )
	
	heat = HeatMap( list(zip(data.LAT.values, data.LON.values)),
    	               min_opacity=0.2,
    	               max_val=100,
        	           radius=20, blur=15, 
            	       max_zoom=10, 
                	 )

	hmap.add_child(heat)
	
	time=datetime.now().strftime('%H%M')
	hmap.save('./data/'+'folium_'+time+'.html')
```

A screenshot of the resulting map is shown below, but the interactive map can
be accessed [here](./iteration0_folium.html)	




This is a map of Peshawar:
<img src="{{ site.url }}{{ site.baseurl }}/images/addressvisualisation/peshmap.png" 
alt="Map of Peshawar, KPK, Pakistan">


