function var_drilldown(elem) {

    console.log(elem.id)

    // Make edits to elements that don't have to be deleted first
    document.getElementById("nav").style.width = "25%"
    document.getElementById("drill").style.display = "block"
    document.getElementById("drill").style.width = "25%"
    document.getElementById("data").style.width = "50%"
    document.getElementById("drill-muni").innerHTML = elem.id.slice(5).concat(" Analytics");    

    // Go through and remove anything that might be in the column already
    if (document.getElementById("plot1") != null) {
        document.getElementById("dd-plot1").removeChild(document.getElementById("plot1"))
    }

    var to_delete = ["unc_title", "unc_level", "unc_hr"]
    var to_delete_p = document.getElementById("drill");

    for (i = 0; i < to_delete.length; ++i) {
        if (document.getElementById(to_delete[i]) != null) {
            to_delete_p.removeChild(document.getElementById(to_delete[i]))
        }
    }


    fetch('/var_drilldown', {

        // When the data gets POSTed back to Flask, it'll be in JSON format
        headers: {
            'Content-Type': 'application/json'
        },

        // Send the data back as a POST request
        method: 'POST',

        // Here's where you construct the JSON
        body: JSON.stringify({
            "info_var": elem.id.slice(5)
        })

        // ...and then send it off
        }).then(function (response) {
            return response.text();
        }).then(function (text) {

            data = JSON.parse(text);

            console.log("done with fetch!!")

            document.getElementById("mig_perc").style.fontSize = "28px";
            document.getElementById("mig_perc").style.textAlign = "center";
            document.getElementById("mig_perc").innerHTML = elem.id.slice(5) + " is ranked <br><h1>" + data['var_rank'] + "</h1>in feature importance out of XX variables."


        })


}