// MIGRATION MAP
// Create the map and add the basemap tiles
var mymap = L.map('mapid').setView([23.6345, -102.5528], zoom_start = 7);
L.tileLayer('https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png', {
	maxZoom: 20,
	attribution: '&copy; <a href="https://stadiamaps.com/">Stadia Maps</a>, &copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors'
}).addTo(mymap);


// Create the drawnPolys group
window.drawnPolys = new L.featureGroup().addTo(mymap);

// Add the draw control to the map
var drawControl = new L.Control.Draw({ 

    draw: {
            circle: false, 
            circlemarker: false, 
            polyline : false,
            marker   : false,
            polygon  : true,
            rectangle: true
        },
    edit: {
           featureGroup: drawnPolys, 
           remove: true}
        });

mymap.addControl(drawControl);


// Make an empty global variable to store the shapeID's of the polygons that overlap COMPLETLEY with any of the polygons a user draws
window.selected_polys = [];


// What happens when a shape is drawn/created
mymap.on(L.Draw.Event.CREATED, e => {

    // Save the layer as an object
    var layer = e.layer;

    // Add the drawn layer to the map
    mymap.addLayer(layer);
    
    // Add the layer to the drawnPolys LayerGroup
    window.drawnPolys.addLayer(layer);

    console.log(window.drawnPolys);
    
    // Zoom in to the drawn poly
    mymap.fitBounds(layer.getBounds());
   
   // Get the layers in the geoJSON layer (no need rn but leaving in for reference)
    var polygons = window.poly.getLayers();

    for (i = 0; i < polygons.length; i++) {
        if (layer.getBounds().contains(polygons[i].getBounds())) {
            // console.log(polygons[i].feature.properties.shapeID)
            window.selected_polys.push(polygons[i].feature.properties.shapeID)
        }
    }

    console.log("Municipalities within selected area: ");
    console.log(window.selected_polys);

});


// Function to remove a given shapeID from a list
function arrayRemove(arr, value) { 
    return arr.filter(function(ele){  return ele != value; });
}


// What happens when a shape is deleted
mymap.on(L.Draw.Event.DELETED, e => {

    // Get a list of the layers that are being deleted
    var poly_being_removed = e.layers;

    // Get a list of the polygons on the map
    var polygons = window.poly.getLayers();

    // For each of the polygons being removed...
    poly_being_removed.eachLayer(function(layer) {

        // Make an empty list to store the polgyon ID's that are no longer overlapping with a drawn polgyon
        ids_to_be_removed = [];

        // For every polygon currently on the map...
        for (i = 0; i < polygons.length; i++) {

            // If it intersects with any of the polgyons being removed then add that shapeID to the ids_to_be_removed list
            if (layer.getBounds().contains(polygons[i].getBounds())) {
                ids_to_be_removed.push(polygons[i].feature.properties.shapeID)
            }
        }

    });


    // Remove all of the ID's in ids_to_be_removed from the list of selected polygons
    for (i = 0; i < ids_to_be_removed.length; i++) {
        window.selected_polys = arrayRemove(window.selected_polys, ids_to_be_removed[i])
    }

    // CELEBRATE GOOD TIMES
    console.log("NOW SELECTED POLYS: ");
    console.log(window.selected_polys);

});




// Function to color the polygons by number of migrants
function getColor(d) {
    return d > 1000 ? '#800026' :
           d > 500  ? '#BD0026' :
           d > 200  ? '#E31A1C' :
           d > 100  ? '#FC4E2A' :
           d > 50   ? '#FD8D3C' :
           d > 20   ? '#FEB24C' :
           d > 10   ? '#FED976' :
                      '#FFEDA0';
}


// Function to style the polygons
function polygon_style(feature) {
  return {
    fillColor: getColor(feature.properties.num_migrants),
    weight: .2,
    opacity: 1,
    color: 'white',
    fillOpacity: 0.5
  };
}


// Removes highlight from polygons when cursor is not over them
function resetHighlight(e) {
    window.poly.resetStyle(e.target);
}


// Zooms to a clicked on polygon
function zoomToFeature(e) {
    mymap.fitBounds(e.target.getBounds());
}


// Highlights a polygon when hovered over
function highlightFeature(e) {

    var layer = e.target;

    layer.setStyle({
        weight: 5,
        color: '#666',
        dashArray: '',
        fillOpacity: 0.7
    });

    if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {
        layer.bringToFront();
    }

}


// On each feature, highlight/remove highlight when hovered over, zoom when clicked and add popup
function onEachFeature(feature, layer) {

    layer.bindPopup('<h2>Municipality: ' + feature.properties.shapeName + '</h2>' + 
                    '<h2>Number of migrants: ' + feature.properties.num_migrants + '</h2>');

    layer.on({
        mouseover: highlightFeature,
        mouseout: resetHighlight,
        click: zoomToFeature
    });

}


// Create the window.poly global variable
window.poly;

// Function to get the data from the Flask function/URL (TO-DO: REMOVE ALL OF THE FUNCTIONS FROM HERE AND USE WINDOW.POLY TO EDIT THEM)
axios.get('http://127.0.0.1:5000/geojson-features')

    .then(response => {

        var polys = L.geoJSON(response.data, {style: polygon_style, onEachFeature: onEachFeature})//.addTo(mymap);
        window.poly = polys;
        window.poly.addTo(mymap);

    })
