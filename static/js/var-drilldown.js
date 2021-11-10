function var_drilldown(elem) {

    console.log(elem.id)

    // Make edits to elements that don't have to be deleted first
    document.getElementById("nav").style.width = "25%"
    document.getElementById("drill").style.display = "block"
    document.getElementById("drill").style.width = "25%"
    document.getElementById("data").style.width = "50%"

    document.getElementById("drill-muni").innerHTML = elem.id.slice(5).concat(" Analytics");    

    // Go through and remove anything that might be in the column already

}