
function stock_info_click() {
    var stock_name = document.getElementById("stock_name").value;
    var dps1 = [], dps2 = [];
    var stockChart = new CanvasJS.StockChart("chartContainer", {
        title: {
            text: stock_name
        },
        /*subtitles: [{
            text: "Simple Moving Average"
        }],*/
        charts: [{
            axisY: {
                prefix: "$"
            },
            legend: {
                verticalAlign: "top",
                cursor: "pointer",
                itemclick: function (e) {
                    if (typeof (e.dataSeries.visible) === "undefined" || e.dataSeries.visible) {
                        e.dataSeries.visible = false;
                    } else {
                        e.dataSeries.visible = true;
                    }
                    e.chart.render();
                }
            },
            toolTip: {
                shared: true
            },
            data: [{
                type: "candlestick",
                showInLegend: true,
                name: "Stock Price",
                yValueFormatString: "$#,###.00",
                xValueType: "dateTime",
                dataPoints: dps1
            }],
        }],
        navigator: {
            data: [{
                dataPoints: dps2
            }],
            slider: {
                minimum: new Date(2018, 03, 01),
                maximum: new Date(2018, 05, 01)
            }
        }
    });

    document.getElementById("chartContainer").style.display = "block";

    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function () {
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {
            //$.getJSON("https://canvasjs.com/data/docs/ethusd2018.json", function (data) {
            var data = xmlHttp.responseText;
            data = JSON.parse(data)
            console.log(data);
            console.log(data[Object.keys(data).length-1]);
            for (var i = 0; i < data.length; i++) {
                dps1.push({ x: new Date(data[i].date), y: [Number(data[i].open), Number(data[i].high), Number(data[i].low), Number(data[i].close)] });
                dps2.push({ x: new Date(data[i].date), y: Number(data[i].close) });
            }
            stockChart.render();
            var sma = calculateSMA(dps1, 7);
            stockChart.charts[0].addTo("data", { type: "line", dataPoints: sma, showInLegend: true, yValueFormatString: "$#,###.00", name: "Simple Moving Average" })
            //});
        }
    }
    xmlHttp.open("GET", window.location.origin + "/load_dataset?name=" + stock_name, true); // true for asynchronous
    xmlHttp.send(null);
}

function stock_signals_click() {
    var stock_name = document.getElementById("stock_name").value;

    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function () {
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {
            console.log(":)")
        }
    }
    window.open(window.location.origin + "/technical_signals?name=" + stock_name, '_blank').focus();
    xmlHttp.send(null);

    //document.getElementById("imageSignals").src = window.location.origin + "/load_technical_signals?name=" + stock_name

    //var xmlHttp = new XMLHttpRequest();
    //xmlHttp.onreadystatechange = function () {
    //    if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {
    //
    //        //var data = xmlHttp.responseText;
    //        //console.log(data)
    //        //data =
    //        console.log(xmlHttp.respose);
    //        var imgSrc = URL.createObjectURL(xmlHttp.respose);
    //
    //        var $img = $( '<img/>', {
    //            "alt": "test image",
    //            "src": imgSrc
    //        } ).appendTo( $( '#imageContainer' ) );
    //        window.URL.revokeObjectURL( imgSrc );
    //
    //    }
    //}
    //xmlHttp.open("GET", window.location.origin + "/load_technical_signals?name=" + stock_name, true); // true for asynchronous
    //xmlHttp.send(null);
}


function calculateSMA(dps, count) {
    var avg = function (dps) {
        var sum = 0, count = 0, val;
        for (var i = 0; i < dps.length; i++) {
            val = dps[i].y[3]; sum += val; count++;
        }
        return sum / count;
    };
    var result = [], val;
    count = count || 5;
    for (var i = 0; i < count; i++)
        result.push({ x: dps[i].x, y: null });
    for (var i = count - 1, len = dps.length; i < len; i++) {
        val = avg(dps.slice(i - count + 1, i));
        if (isNaN(val))
            result.push({ x: dps[i].x, y: null });
        else
            result.push({ x: dps[i].x, y: val });
    }
    return result;
}

function loadDataset(name) {
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open("GET", window.location.origin + "/load_dataset?"+name, false); // false for synchronous request
    xmlHttp.send(null);
    return xmlHttp.responseText;
}