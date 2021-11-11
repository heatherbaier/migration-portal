function change_year(direc) {

    var cur_year = parseInt(document.getElementById("ts-year").innerHTML)

    if (direc == "increase") {
        cur_year += 5
    } else {
        cur_year -= 5
    }

    if (cur_year == 2010) {
        document.getElementById("lc").style.visibility = "hidden";
        document.getElementById("rc").style.visibility = "visible";
    } else {
        document.getElementById("rc").style.visibility = "hidden";
        document.getElementById("lc").style.visibility = "visible";
    }

    document.getElementById("ts-year").innerHTML = cur_year

}