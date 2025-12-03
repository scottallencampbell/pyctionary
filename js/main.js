var currentLabel = undefined;
var model = undefined;
var interval = undefined;
var mouseMoved = false;
var canvas = undefined;
var ctx = undefined;
var canvasDownscaled = undefined;
var ctxDownscaled = undefined;
var canvasPreprocessed = undefined;
var ctxPreprocessed = undefined;
var chartData = undefined;
var chart = undefined;
var sketchpad = undefined;
var correctGuessTimer = undefined;
var strokes = [];
const WIDTH = 28;
const HEIGHT = 28;
const MODEL_PATH = './models/dist/model.json';

const hashString = (str) => {
    return str.split('').reduce((prevHash, currVal) =>
      (((prevHash << 5) - prevHash) + currVal.charCodeAt(0))|0, 0);
}

const registerMouseMove = () => {
    mouseMoved = true;
}

const guess = async (force) => {
    if (mouseMoved === false && force !== true) { return; }

    var results = await getPredictions();
    
    renderPredictions(results);    
}

const getPredictions = async (force) => {
    await pica().resize(canvas, canvasDownscaled);
    const centered = recenterImage(ctxDownscaled, ctxPreprocessed);

    if (centered === false && force !== true) { return null; }

    const tensor = tf.browser.fromPixels(canvasPreprocessed)
        .mean(2)
        .expandDims(2)
        .expandDims()
        .toFloat();

    const pixels = tensor.div(255.0);
    const predictions = await model.predict(pixels).data();
    const results = Array.from(predictions);

    return results;
}

const renderPredictions = (results) => {
    if (results !== null) {
        const len = results.length;
        const indices = new Array(len);

        for (let i = 0; i < len; ++i) {
            indices[i] = i;
        }

        let map = results.map((el) => { return { value: el }});    
        const labelIndex = LABELS_LOOKUP[currentLabel];
        map[labelIndex].itemStyle = {color: '#ccccff'};
    
        chartData.series[0].data = map;
        chart.setOption(chartData);    

        const topIndex = results.indexOf(Math.max.apply(Math, results));

        if(LABELS[topIndex] === currentLabel) {
            if (correctGuessTimer === undefined) {
                correctGuessTimer = setTimeout(celebrateWin, 2000);
            }
        } else if (correctGuessTimer !== undefined) {
            clearTimeout(correctGuessTimer);
            correctGuessTimer = undefined;
        }       
    }

    $('#output').fadeTo(200, results === null ? 0 : 1);
    
    mouseMoved = false;
}

const celebrateWin = () => {
    sketchpad.mode = 'disabled';
    $('#blind').fadeIn();
    
    setTimeout(() => {
        const jsConfetti = new JSConfetti();

        jsConfetti.addConfetti({ 
            emojis: ['🌈', '🦄', '🐶', '🚀', '❤️', '😊', '🥇', '🙌'],
            emojiSize: 50,
            confettiNumber: 80
        });
    }, 1500);

    setTimeout(() => {
        $('#container').fadeOut();
    }, 3000);

    setTimeout(() => {
        $('body').fadeOut();
    }, 3500);

    setTimeout(() => {        
        reload();
    }, 4000);
}

const recenterImage = (ctxFrom, ctxTo) => {
    let center = getCenterOfMass(ctxFrom, WIDTH, HEIGHT);
    if (center[0] == -1) { return false; }

    const offsetX = parseInt((WIDTH / 2 - center[0]));
    const offsetY = parseInt((HEIGHT / 2 - center[1]));

    const image = ctxFrom.getImageData(-offsetX, -offsetY, WIDTH, HEIGHT);
    ctxTo.putImageData(image, 0, 0);

    return true;
}

const getCenterOfMass = (context, width, height) => {
    let sumX = 0;
    let sumY = 0;
    let sum = 0;
    const image = context.getImageData(0, 0, width, height);

    for (let x = 0; x < height; x++) {
        for (let y = 0; y < height; y++) {
            const signal = image.data[(x + y * height) * 4 + 3] / 255;

            if (signal > 0) {
                sumX = sumX + signal * x;
                sumY = sumY + signal * y;
                sum += signal;
            }
        }
    }

    if (sum == 0) {
        return [-1, -1];
    }
    else {
        centerX = Math.floor(sumX / sum);
        centerY = Math.floor(sumY / sum);

        return [centerX, centerY];
    }
}

const specifyCanvases = () => {
    canvas = document.querySelector('div#sketchpad canvas');
    ctx = canvas.getContext('2d');

    canvasDownscaled = document.querySelector('div#downscaled canvas');
    ctxDownscaled = canvasDownscaled.getContext('2d');

    canvasPreprocessed = document.querySelector('div#preprocessed canvas');
    ctxPreprocessed = canvasPreprocessed.getContext('2d');

    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

const buildChart = () => {
    chart = echarts.init(document.getElementById('output'));
   
    let labelMap = LABELS.map(function(el){ return { value: el }});
    const labelIndex = LABELS_LOOKUP[currentLabel];
    labelMap[labelIndex].textStyle = {color: '#ccccff'};
   
    chartData = {
        grid: { left: '0', right: '200px' },
        xAxis: {
            max: 1,
            show: false
        },
        yAxis: {
            type: 'category',
            data: labelMap,
            inverse: true,
            animationDuration: 200,
            animationDurationUpdate: 200,
            max: 10,
            offset: 10,
            axisLine: {
                show: false
            },
            axisTick: {
                show: false
            },
            position: 'right',
            axisLabel: {
                fontSize: 18,
                fontWeight: 'bold',
                color: '#cccccc',
                fontFamily: 'Roboto'
            }
        },
        series: [
            {
                barWidth: 34,
                barCategoryGap: 20,
                itemStyle: {
                    color: '#ffffff'
                },
                realtimeSort: true,
                type: 'bar',
                stack: 'x',
                data: [],
                label: {
                    show: false,
                    position: 'left',
                    valueAnimation: true
                },
                showBackground: true,
                backgroundStyle: {
                    color: 'rgba(250, 250, 250, .3)'
                }
            }
        ],
        legend: {
            show: true
        },
        animationDuration: 0,
        animationDurationUpdate: 200,
        animationEasing: 'linear',
        animationEasingUpdate: 'linear'
    };
}

const enableSketchpad = () => {
    sketchpad = new Atrament(canvas, { 
        width: 500, 
        height: 500, 
        color: 'white', 
        weight: 5, 
        smoothing: .7
    });

    sketchpad.recordStrokes = true;
    sketchpad.addEventListener('strokerecorded', ({ stroke }) => {
        if (!sketchpad.recordPaused) {
            strokes.push(stroke);
        }
    }); 

    strokes = [];
}

const clearCanvas = () => {
    sketchpad.clear();
    strokes = [];
    guess(true);
}

const undoLastStroke = () => {
    strokes.pop();
    sketchpad.clear();
    sketchpad.recordPaused = true;
    
    for (let stroke of strokes) {
        sketchpad.mode = stroke.mode;
        sketchpad.weight = stroke.weight;
        sketchpad.smoothing = stroke.smoothing;
        sketchpad.color = stroke.color;
        sketchpad.adaptiveStroke = stroke.adaptiveStroke;

        const points = stroke.points.slice();
        const firstPoint = points.shift().point;

        sketchpad.beginStroke(firstPoint.x, firstPoint.y);

        let prevPoint = firstPoint;
        while (points.length > 0) {
            const point = points.shift();
            const { x, y } = sketchpad.draw(point.point.x, point.point.y, prevPoint.x, prevPoint.y);
            prevPoint = { x, y };
        }
        sketchpad.endStroke(prevPoint.x, prevPoint.y);
    }

    sketchpad.recordPaused = false;
    guess(true);
}

const selectLabel = () => {
    const cookie = getCookie('labelIndex');
    
    let labelIndex = 0;

    if (cookie === null || parseInt(cookie) === NaN) {
        labelIndex = Math.floor(Math.random() * LABELS.length);      
    } else {
        labelIndex = (parseInt(cookie) + 1) % LABELS.length;
    }

    setCookie('labelIndex', labelIndex.toString(), 365);

    currentLabel = LABELS[SHUFFLED[labelIndex]];
    console.log('index/label: ' + labelIndex + '/' + currentLabel );
}

const displayLabel = () => {
    let prettified = '';
    
    if (PLURALS.indexOf(currentLabel) >= 0) {
        prettified = currentLabel;
    }
    else if (['a','e','i','o','u'].indexOf(currentLabel[0].toLowerCase()) !== -1) {
        prettified = 'an ' + currentLabel;
    }
    else {
        prettified = 'a ' + currentLabel;
    }

    $('#question #label').html(prettified);
}

const renderPage = () => {
    $('body').delay(2000).fadeIn();
    $('#question').delay(3000).fadeIn();
    $('#question #label').delay(4000).fadeIn();
    $('#main').delay(5000).fadeIn();
}

const load = () => {
    // get references to all three canvases and contexts
    specifyCanvases();

    // load the sketchpad control
    enableSketchpad();
    
    // load the model
    tf.loadLayersModel(MODEL_PATH).then(function (result) { 
        model = result;
        // run the model prediction once on init, some latency in first load don't want a hiccup while drawing
        getPredictions(true);
     });

    // select label to be drawn
    selectLabel();
    displayLabel();

    // set up barchart display of predictions
    buildChart();

    // progressively render page controls
    renderPage();

    canvas.addEventListener('mousemove', registerMouseMove, false);
    canvas.addEventListener('touchmove', registerMouseMove, false);

    interval = setInterval(guess, 500);

    console.log(screen.width);
}

const reload = () => {
    $('#blind, #question, #question #label, #main').hide();
    $('#container').show();
    clearCanvas();
    selectLabel();
    displayLabel();
    buildChart();
    renderPage();
    
    sketchpad.mode = 'draw';
}

if (location.protocol !== 'https:' && !location.href.includes('localhost') && !location.href.includes('127.0.0.1')) {
    location.protocol = 'https:';
}

$(document).ready(load);



