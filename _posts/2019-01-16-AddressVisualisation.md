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
	latList=latList[np.isfinite(latList)]
	lonList=lonList[np.isfinite(lonList)]
	
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
be accessed [here](/extrahtml/initial_folium.html):

<img src="{{ site.url }}{{ site.baseurl }}/images/addressvisualisation/initial.jpg" 
alt="Initial heatmap">
	
The hospital in the map above is 
labelled by the red letter H. One can see there are 
many addresses near that location. Additioanlly, there is also a high density of 
points in the region labelled by a dashed white circle. 
When we investigate these points, there are 2 scenarios: 1)they are in 
fact partial matches from the address search by Google Maps and in turn, those
searches have returned the general location as 'Peshawar' OR 2) partial matches have
lead to incorrect lon and lat values. 

To fix this, we can run a test over the search results to see if they match the 
addresses from the data - if there is a match then we can plot the result and if 
not then we must mark it accordingly. This involves comparing the two sets of 
strings: 1) the input address and 2) the search result. To compare the strings, we 
can use the [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) 
paramter which can be measured for two strings, using the Levenshtein library.

```python
	import Levenshtein as lev
	
	# levenshtein distance funtion
	# to measure difference between two strings which could be multiple words
	def SimilarityBtwStrings(inp, res):
	    # only consider whole of input 
        # but part of result - first 2 elements (separated by commas)
	    
	    # first get a list of result words
	    rWordList=[]
	    #list1 is based on commas
	    list1=res.split(', ')[0:2] #this number 2 gets us the first 2 elements
	    for rPhrase in list1:
	        if ' ' in rPhrase:
	            list2=rPhrase.split(' ')
	            for rWord in list2:
	                rWordList.append(rWord)
	        else:
	            rWordList.append(rPhrase)
	    rWordList=[i.lower() for i in rWordList]
	    
	    iWordList=inp.strip().split(' ')
	    iWordList=[i.lower() for i in iWordList]
	
		# compare the words in each List of strings
		# as soon as we get a match (except for numbers, 'peshawar', 'phase)
		# we return 1 which denotes a positive match
	    for i in iWordList:
	        for r in rWordList:
	            if r!='peshawar' and i!='peshawar' and \
	               r!='phase' and i!='phase' and \
	               r.isdigit()!=True and i.isdigit()!=True:
	                   if lev.distance(i,r) < 2: # tolerance level is 0 or 1
	                       return 1
	      
	    return 0
	
	data = pd.read_csv("./data/GMapsAddress_1329.csv", index_col=0)

	#will need a new column measuring precision
	data['MATCHING']=None
	
	counter=0
	for index, row in data.iterrows():
	    inp1=row['pntAddress']
	    res1=row['GMAPS_NAME']
	    if type(res1)==str: #ignoring NaN values
	        sim=SimilarityBtwStrings(inp1, res1)
	        counter+=sim
	        if sim==1:
	            data.iat[index, data.columns.get_loc('MATCHING')]=1
	        else:
	            data.iat[index, data.columns.get_loc('MATCHING')]=0
	
	accuracy=round((counter/len(data))*100, 4)
	print('accuracy of the results', accuracy) 
	
	data=data.loc[data['MATCHING']==1]
	
	#save the new dataframe that holds all the information
	time=datetime.now().strftime('%H%M')
	data.to_csv('./data/GMapsAddressesAccurate_'+time+'.csv')
```

The accuracy of the list of addresses is 50% when one considers a positive match
to be perfect. A perfect match will result in a Levenshtein distance of 0. If we 
increase this tolerance to 1 rather than 0, then our accuracy jumps up to 66%.
Plotting only the 66% of so called accurate results yields the following heatmap (the
interactive version can be accessed [here](/extrahtml/final_folium.html)):

<img src="{{ site.url }}{{ site.baseurl }}/images/addressvisualisation/final.jpg" 
alt="Initial heatmap">

This heatmap can now be used by the management of AIMS to start targetting certain 
areas to maximise the return of the charitable health camps.

But what about the 34% of addresses that did not get a match. These addresses 
mostly inldude strings with spelling mistakes. A clustering algorithm could be 
used to group together similar strings. This algroithm could be informed by the 
results of the 66% of addresses too (in the pipeline, so please stay tuned). 
Additonally, the common areas within the city could form part of a drop down 
list to increase the speed with which the hospital's reception records the 
addresses of the patients.





