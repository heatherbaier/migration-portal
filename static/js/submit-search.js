const submit_button = document.getElementById("submit").addEventListener("click", check);

// Function to briefly pause the update function so it doesn't go crazy
function sleep(milliseconds) {

    const date = Date.now();
    var currentDate = null;

    do {
        currentDate = Date.now();
    } while (currentDate - date < milliseconds);

}


function check(event) {

    // 1. Create a new XMLHttpRequest object
    let xhr = new XMLHttpRequest();

    // 2. Configure it: GET-request for the URL /article/.../load
    xhr.open('GET', "http://127.0.0.1:5000/status_update");

    // 3. Send the request over the network
    xhr.send();

    // 4. This will be called after the response is received
    xhr.onload = function() {
        var result = JSON.parse(xhr.responseText);
        if (result['status'] == "Rendering new migration map...") {
            console.log('Finished!');
            document.getElementById("status").innerHTML = result['status']
        } else {
            console.log('Not done!')
            document.getElementById("status").innerHTML = result['status']
            sleep(500);
            xhr.open('GET', "http://127.0.0.1:5000/status_update");
            xhr.send();
        }
    }

}
