// Function to color the polygons by total number of migrants
function total_mig_color(d) {

    return d > 5000 ? '#800026' :
           d > 2500  ? '#BD0026' :
           d > 500  ? '#E31A1C' :
           d > 250  ? '#FC4E2A' :
           d > 100   ? '#FD8D3C' :
           d > 50   ? '#FEB24C' :
           d > 10   ? '#FED976' :
                      '#FFEDA0';

}

// Function to style the polygons
function total_mig_style(feature) {
    return {
        fillColor: total_mig_color(feature.properties.num_migrants),
        weight: .2,
        opacity: 1,
        color: 'white',
        fillOpacity: 0.5
    };
}


// Function to color the polygons by percentage change in migrants
function perc_change_color(d) {

    return  d > .00001        ? '#440154FF' :
            d > -.00001       ? '#FDE725FF' :
            d > -100000  ? '#73D055FF':
                           '#FDE725FF';
}

// Function to style the polygons
function perc_change_style(feature) {
    return {
        fillColor: perc_change_color(feature.properties.num_migrants),
        weight: .2,
        opacity: 1,
        color: 'white',
        fillOpacity: 0.5
    };
}



// Function to color the polygons by change in migrants
function abs_change_color(d) {

    return  d > .00001        ? '#440154FF' :
            d > -.00001       ? '#FDE725FF' :
            d > -100000  ? '#73D055FF':
                           '#FDE725FF';
}

// Function to style the polygons
function abs_change_style(feature) {
    return {
        fillColor: abs_change_color(feature.properties.num_migrants),
        weight: .2,
        opacity: 1,
        color: 'white',
        fillOpacity: 0.5
    };
}



function change_map_var(variable) {

    console.log("here!!", variable)

    var to_send = {'variable': variable}


    axios.post('http://127.0.0.1:5000/update_map', to_send)

        .then(response => {


            // Remove the current migration data layer from the map
            mymap.removeLayer(window.poly);
            mymap.removeControl(legend);

            console.log("removed legend");



            legend = L.control({position: 'bottomleft'});
            legend.onAdd = function (map) {

                var div = L.DomUtil.create('div', 'info legend');

                if (variable == "sum_num_intmig") {
                    var grades = [0, 10, 50, 100, 250, 500, 2500, 5000];
                } else if (variable == "perc_migrants") {
                    var grades = [0, 0.02, .04, .06, .08, .1, .5, .8, .8];
                } else if (variable == "absolute_change") {
                    var grades = [-100000, -.00001, .00001];
                } else {
                    var grades = [-100000, -.00001, .00001];
                }

                var labels = [];

                // loop through our density intervals and generate a label with a colored square for each interval
                for (var i = 0; i < grades.length; i++) {
                    div.innerHTML +=
                        '<i style="background:' + getColor(grades[i] + 1) + '"></i> ' +
                        grades[i] + (grades[i + 1] ? ' to ' + grades[i + 1] + '<br>' : '+');
                }

                return div;
            };
            legend.addTo(mymap);





            window.poly = [];

            // Remove the drawn polygons from the map and re-initalize the drawnPolys group as empty
            mymap.removeLayer(window.drawnPolys);
            window.drawnPolys = new L.featureGroup().addTo(mymap);

            // Convert the new migration data into a leaflet geoJSON

            if (variable == "sum_num_intmig") {
                var polys = L.geoJSON(response.data, {style: total_mig_style, onEachFeature: onEachFeature})//.addTo(mymap);
            } else if (variable == "perc_migrants") {
                var polys = L.geoJSON(response.data, {style: polygon_style, onEachFeature: onEachFeature})//.addTo(mymap);
            } else if (variable == "absolute_change") {
                var polys = L.geoJSON(response.data, {style: abs_change_style, onEachFeature: onEachFeature})//.addTo(mymap);
            } else {
                var polys = L.geoJSON(response.data, {style: perc_change_style, onEachFeature: onEachFeature})//.addTo(mymap);
            }

            console.log("POLYS: ", polys.getLayers().length)

            // Update the global window.poly variable & add it to the map
            window.poly = polys;
            window.poly.addTo(mymap);

            window.selected_polys = [];

            // Zoom the map back out to all of Mexico                
            mymap.setView(new L.LatLng(23.6345, -102.5528), 6);



        })

}



