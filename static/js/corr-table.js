function make_corr_tables(data) {

    var canvasL = document.getElementById("canvas-divL")
    var canvasR = document.getElementById("canvas-divR")
    var scenario_title = document.getElementById('scenario-title')

    scenario_title.innerHTML = "Scenario breakdown";

    var bs_analytics = document.createElement('canvas');
    bs_analytics.id = 'corr-by-category';
    canvasL.appendChild(bs_analytics); // append the left chart to the left canvas


    var corr_var = document.createElement('canvas');
    corr_var.id = 'corr-by-variable';
    canvasR.appendChild(corr_var); // append the left chart to the left canvas

    var backgroundColors = [
        'rgba(70, 109, 29, 0.5)',
        'rgba(54, 162, 235, 0.5)',
        'rgba(255, 206, 86, 0.5)',
        'rgba(208, 49, 45, 0.5)',
        'rgba(153, 102, 255, 0.5)',
        'rgba(255, 159, 64, 0.5)',
        'rgba(40, 30, 93, 0.5)'
    ];

    var borderColors = [
        'rgba(70, 109, 29, 1)',
        'rgba(54, 162, 235, 1)',
        'rgba(255, 206, 86, 1)',
        'rgba(208, 49, 45, 1)',
        'rgba(153, 102, 255, 1)',
        'rgba(255, 159, 64, 1)',
        'rgba(40, 30, 93, 1)'
    ];


    var ctx1 = document.getElementById('corr-by-variable').getContext('2d');
    var myChart1 = new Chart(ctx1, {
        type: 'bar',
        data: {
            labels: data["corr_category_dict"]['Deomographic'][0],
            datasets: [{
                label: '% change',
                data: data["corr_category_dict"]['Deomographic'][1],
                backgroundColor: backgroundColors[1],
                borderColor: borderColors[1],
                borderWidth: 1
            }]
        },
        options: {
            maintainAspectRatio: false,
            legend: { display: false },
            title: {
                display: true,
                text: "Relative changes in sociodemographic variables",
                fontSize: 20
            },
            scales: {
                    y: {
                        beginAtZero: true
                    },
                    yAxes: [{
                        ticks: {
                            fontSize: 20
                        }
                    }],
                    xAxes: [{
                        ticks: {
                            fontSize: 15
                        }
                    }]
            }
        }
    });


    var ctx = document.getElementById('corr-by-category').getContext('2d');
    var myChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Economic', 'Demographic', 'Family Unit', 'Health', 'Occupational', 'Education', 'Household'],
            datasets: [{
                label: '% change',
                data: data['corr_means'],
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
            maintainAspectRatio: false,
            legend: { display: false },
            title: {
                display: true,
                text: "Relative changes in sociodemographic variables",
                fontSize: 20
            },
            scales: {
                    y: {
                        beginAtZero: true
                    },
                    yAxes: [{
                        ticks: {
                            fontSize: 20
                        }
                    }],
                    xAxes: [{
                        ticks: {
                            fontSize: 20
                        }
                    }]
            },
            'onClick' : function (evt, item) {
                // console.log ('legend onClick', evt);
                console.log('bar index: ', item[0]["_index"]);
                edit_corr_table(data, item[0]["_index"]);
            }
        }
    });

}




function edit_corr_table(data, index) {

    var categories = ["Economic", "Deomographic", "Family", "Employment", "Health", "Education", "Household"]


    var backgroundColors = [
        'rgba(70, 109, 29, 0.5)',
        'rgba(54, 162, 235, 0.5)',
        'rgba(255, 206, 86, 0.5)',
        'rgba(208, 49, 45, 0.5)',
        'rgba(153, 102, 255, 0.5)',
        'rgba(255, 159, 64, 0.5)',
        'rgba(40, 30, 93, 0.5)'
    ];

    var borderColors = [
        'rgba(70, 109, 29, 1)',
        'rgba(54, 162, 235, 1)',
        'rgba(255, 206, 86, 1)',
        'rgba(208, 49, 45, 1)',
        'rgba(153, 102, 255, 1)',
        'rgba(255, 159, 64, 1)',
        'rgba(40, 30, 93, 1)'
    ];


    var canvasR = document.getElementById("canvas-divR");
    var oldcanv = document.getElementById('corr-by-variable');
    canvasR.removeChild(oldcanv);

    var corr_var = document.createElement('canvas');
    corr_var.id = 'corr-by-variable';
    canvasR.appendChild(corr_var); // append the left chart to the left canvas
    



    var ctx1 = document.getElementById('corr-by-variable').getContext('2d');

    var myChart1 = new Chart(ctx1, {
        type: 'bar',
        data: {
            labels: data["corr_category_dict"][categories[index]][0],
            datasets: [{
                label: '% change',
                data: data["corr_category_dict"][categories[index]][1],
                backgroundColor: backgroundColors[index],
                borderColor: borderColors[index],
                borderWidth: 1
            }]
        },
        options: {
            maintainAspectRatio: false,
            legend: { display: false },
            title: {
                display: true,
                text: "Relative changes in sociodemographic variables",
                fontSize: 20
            },
            scales: {
                    y: {
                        beginAtZero: true
                    },
                    yAxes: [{
                        ticks: {
                            fontSize: 20
                        }
                    }],
                    xAxes: [{
                        ticks: {
                            fontSize: 15
                        }
                    }]
            }
        }
    });

}



