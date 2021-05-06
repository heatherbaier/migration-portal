function predict_migration() {

    // Make two new variables to store column name sand their associated percentage change values
    var column_names = [],
        percent_changes = [];

    // Grab all of the percent change inputs
    var inputs = document.getElementsByTagName('input');

    // For all of the inputs, append the ID and value to their respective lists
    for (i = 0; i < inputs.length; ++i) {
        column_names.push(inputs[i].id);
        percent_changes.push(inputs[i].value);
    }

    // If there are no municipalities slected, provide the user with the option to either go back and pick some or add the inputted increases to all of them
    if (window.selected_polys.length == 0) {
        if (confirm("\nYou haven't selected any municipalities on the map.\n\nPress OK to apply your variable changes to all of the municipalities in Mexico or Cancel to go back and choose a subset of municipalities.") != true) {
            return 
        }
    }

    // Post all of the variable ID's and their percent changes back to Flaks
    fetch('/predict_migration', {

        // When the data gets POSTed back to Flask, it'll be in JSON format
        headers: {
            'Content-Type': 'application/json'
        },

        // Send the data back as a POST request
        method: 'POST',

        // Here's where you construct the JSON
        body: JSON.stringify({

            "selected_municipalities": window.selected_polys,
            "column_names": column_names,
            "percent_changes": percent_changes

        })

        // ...and then send it off
        }).then(function (response) {
            return response.text();
        }).then(function (text) {

            // Remove the current migration data layer from the map
            mymap.removeLayer(window.poly);

            // Remove the drawn polygons from the map and re-initalize the drawnPolys group as empty
            mymap.removeLayer(window.drawnPolys);
            window.drawnPolys = new L.featureGroup().addTo(mymap);

            // Convert the new migration data into a leaflet geoJSON
            var polys = L.geoJSON(JSON.parse(text), {style: polygon_style, onEachFeature: onEachFeature})//.addTo(mymap);

            // Update the global window.poly variable & add it to the map
            window.poly = polys;
            window.poly.addTo(mymap);

            // if button for keeping previous polygons is
            window.selected_polys = [];

            // Zoom the map back out to all of Mexico                
            // mymap.fitBounds(window.poly.getBounds());
            mymap.setView(new L.LatLng(23.6345, -102.5528), 7);
            
            // Function to get the data from the Flask function/URL (TO-DO: REMOVE ALL OF THE FUNCTIONS FROM HERE AND USE WINDOW.POLY TO EDIT THEM)
            axios.get('http://127.0.0.1:5000/update_stats')

            .then(response => {

                // Update all of the HTML text that doesn't involve the trending icon
                document.getElementById("total_migrants").innerHTML = response.data['predicted_migrants'];
                document.getElementById("change_migrants").innerHTML = response.data['change'];
                document.getElementById("avg_age").innerHTML = response.data['avg_age'];
                document.getElementById("avg_age_change").innerHTML = response.data['avg_age_change'];
                document.getElementById("pchange_migrants").innerHTML = response.data['p_change'].toString().concat("%");
                document.getElementById("pavg_age_change").innerHTML = response.data['pavg_age_change'].toString().concat("%");

                // if p_change is greater than 1, make icon green & trending_up and vice versa
                if (response.data['p_change'] > 0) {
                    document.getElementById("pchange_icon").innerHTML = 'trending_up'
                    document.getElementById("pchange_icon").style.color = 'green'
                } else {
                    // document.getElementById("pchange_migrants").innerHTML = response.data['p_change'].toString().concat("%");
                    document.getElementById("pchange_icon").innerHTML = 'trending_down'
                    document.getElementById("pchange_icon").style.color = 'red'
                }

                // if pavg_age_change is greater than 1, make icon green & trending_up and vice versa
                if (response.data['pavg_age_change'] > 0) {
                    document.getElementById("page_change_icon").innerHTML = 'trending_up'
                    document.getElementById("page_change_icon").style.color = 'green'
                } else {
                    // document.getElementById("pavg_age_change").innerHTML = response.data['pavg_age_change'].toString().concat("%");
                    document.getElementById("page_change_icon").innerHTML = 'trending_down'
                    document.getElementById("page_change_icon").style.color = 'red'
                }

            });

            // document.getElementById("status").innerHTML = "Done."

        });

        // window.status = ''


        // while (window.status != 'Done.') {

        //     axios.get('http://127.0.0.1:5000/status_update')

        //     .then(response => {

        //         console.log('here!')

        //         document.getElementById("status").innerHTML = response.data['status'];
        //         window.status = response.data['status']
            
        //     });

        // }   


}