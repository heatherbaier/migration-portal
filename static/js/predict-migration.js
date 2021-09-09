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
        if (confirm("\nNo municipalities have been selected on the map.\n\nPress 'OK' to apply your variable changes to all of the municipalities in Mexico or 'Cancel' to go back and choose a subset of municipalities.") != true) {
            return 
        }
    }

    // If the user hasn't made any changes to the data, don't allow them to run the model
    var num_changes = percent_changes.filter(x => x != '100').length
    if (num_changes == 0) {
        if (alert("\nNo changes have been made to the input variables. Please edit the desired variables before predicting again.") != true) {
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

            window.poly = [];

            // Remove the drawn polygons from the map and re-initalize the drawnPolys group as empty
            mymap.removeLayer(window.drawnPolys);
            window.drawnPolys = new L.featureGroup().addTo(mymap);

            // Convert the new migration data into a leaflet geoJSON
            var polys = L.geoJSON(JSON.parse(text), {style: polygon_style, onEachFeature: onEachFeature})//.addTo(mymap);

            console.log("POLYS: ", polys.getLayers().length)

            // Update the global window.poly variable & add it to the map
            window.poly = polys;
            window.poly.addTo(mymap);

            window.selected_polys = [];

            // Zoom the map back out to all of Mexico                
            // mymap.fitBounds(window.poly.getBounds());
            mymap.setView(new L.LatLng(23.6345, -102.5528), 7);
            
            // Function to get the data from the Flask function/URL (TO-DO: REMOVE ALL OF THE FUNCTIONS FROM HERE AND USE WINDOW.POLY TO EDIT THEM)
            axios.get('http://127.0.0.1:5000/update_stats')

            .then(response => {

                // Update all of the HTML text that doesn't involve the trending icon
                document.getElementById("total_migrants").innerHTML = response.data['predicted_migrants'].toLocaleString();
                document.getElementById("change_migrants").innerHTML = response.data['change'].toLocaleString().concat(" migrants");
                document.getElementById("avg_age").innerHTML = response.data['avg_age'];
                document.getElementById("avg_age_change").innerHTML = response.data['avg_age_change'].toString().concat(" years");
                document.getElementById("pchange_migrants").innerHTML = response.data['p_change'].toString().concat("%");
                document.getElementById("pavg_age_change").innerHTML = response.data['pavg_age_change'].toString().concat("%");

                // if p_change is greater than 1, make icon green & trending_up and vice versa
                if (response.data['p_change'] > 0) {
                    document.getElementById("pchange_icon").innerHTML = 'trending_up'
                    document.getElementById("pchange_icon").style.color = 'red'
                } else {
                    // document.getElementById("pchange_migrants").innerHTML = response.data['p_change'].toString().concat("%");
                    document.getElementById("pchange_icon").innerHTML = 'trending_down'
                    document.getElementById("pchange_icon").style.color = 'green'
                }

                // if pavg_age_change is greater than 1, make icon green & trending_up and vice versa
                if (response.data['pavg_age_change'] > 0) {
                    document.getElementById("page_change_icon").innerHTML = 'trending_up'
                    document.getElementById("page_change_icon").style.color = 'black'
                } else {
                    // document.getElementById("pavg_age_change").innerHTML = response.data['pavg_age_change'].toString().concat("%");
                    document.getElementById("page_change_icon").innerHTML = 'trending_down'
                    document.getElementById("page_change_icon").style.color = 'black'
                }

                // Update the status so the user knows everything is done
                document.getElementById("status").innerHTML = "Done."

                // window.selected_polys = [];

                console.log("Municipalities within selected area post update: ");
                console.log(window.selected_polys.length);


                var analytics_div = document.getElementById("analytics")

                var bs_analytics = document.createElement('canvas');
                bs_analytics.id = 'border-sector-analytics';
                // bs_analytics.style.color = "black";
                // bs_analytics.width = "50%";
                // bs_analytics.style.width = "1000px";

                // bs_analytics.style.height = "1000px";

                bs_analytics.style.fontSize = "50px";
                // div.className = 'border pad';

                analytics_div.appendChild(bs_analytics);

                var ctx = document.getElementById('border-sector-analytics').getContext('2d');


                var myChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: ['Economic', 'Demographic', 'Family Unit', 'Health', 'Occupational', 'Education', 'Household'],
                        datasets: [{
                            label: '% change',
                            data: [12, -10, 3, -4, 2, 3, -14],
                            backgroundColor: [
                                'rgba(70, 109, 29, 0.5)',
                                'rgba(54, 162, 235, 0.5)',
                                'rgba(255, 206, 86, 0.5)',
                                'rgba(208, 49, 45, 0.5)',
                                'rgba(153, 102, 255, 0.5)',
                                'rgba(255, 159, 64, 0.5)',
                                'rgba(40, 30, 93, 0.5)'
                            ],
                            borderColor: [
                                'rgba(70, 109, 29, 1)',
                                'rgba(54, 162, 235, 1)',
                                'rgba(255, 206, 86, 1)',
                                'rgba(208, 49, 45, 1)',
                                'rgba(153, 102, 255, 1)',
                                'rgba(255, 159, 64, 1)',
                                'rgba(40, 30, 93, 1)'
                            ],
                            borderWidth: 1
                        }]
                    },
                    options: {
                        legend: { display: false },
                        title: {
                            display: true,
                            text: "Relative changes in sociodemographic variables"
                        },
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });
                



            });

        });

}