import folium

m = folium.Map(location=[37.53897093698831, 127.05461953077439], 
               zoom_start=14)
# icon basic
folium.Marker([37.54706945947954, 127.04740975332888],
              icon=folium.Icon(color='black', icon='info-sign')).add_to(m)
# icon color
folium.Marker(
    [37.54461957910074, 127.05590699103249], 
    popup="<a href='https://zero-base.co.kr/' target=_'blink'>마커</a>",
    tooltip='Zerobase',
    icon=folium.Icon(
        color='red',
        icon_color='blue',
        icon='cloud'
    )
).add_to(m)

# icon custom
folium.Marker(
    [37.54041716624373, 127.06914637466906], 
    popup='<b>subway</b>',
    tooltip='Icon custom',
    icon=folium.Icon(
        color='purple',
        icon_color='white',
        icon='glyphicon glyphicon-thumbs-up', # googledp glyphicon 검색해서 관련된 코드 제공해주는 사이트들 참고해서 원하는 것 가져오기
        prefix='glyphicon'
    )
).add_to(m)

m