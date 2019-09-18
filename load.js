let canvas,canvas2,ctx,ctx2;
async function loadModel() {
    console.log('Loading model');
    const model = await tf.loadLayersModel('http://127.0.0.1:8080/my-model.json');
    console.log('Model loaded');
    console.log(model);
	return model;
};


function prepCanvas() {
    canvas = document.getElementById('mycanvas');
    //canvas2 = document.getElementById('mycanvas2');
    ctx = canvas.getContext('2d');
    //ctx2 = canvas2.getContext('2d');
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.lineWidth = 12;
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    //ctx2.fillStyle = 'black';
    //ctx2.fillRect(0, 0, canvas2.width, canvas2.height);
    ctx.strokeStyle = "white";;
    let flag = false;

    function start(e) {
        //console.log(e.clientX,e.clientY);
        flag = true;
    }

    function stop() {
        flag = false;
        ctx.beginPath();
    }
    
    function draw(e) {
        if(flag) {
            //console.log(e.clientX - canvas.offsetLeft,e.clientY - canvas.offsetTop);
            ctx.lineTo(e.clientX - canvas.offsetLeft,e.clientY - canvas.offsetTop);
            ctx.stroke();
        }
    }

    
    

    canvas.addEventListener('mousedown',start);
    canvas.addEventListener('mouseup',stop);
    canvas.addEventListener('mousemove',draw);
    canvas.addEventListener('mouseout',stop);
    canvas.addEventListener('touchstart',start);
    canvas.addEventListener('touchend',stop);
    canvas.addEventListener('touchmove',draw);
}

document.addEventListener('DOMContentLoaded', loadModel);
document.addEventListener('DOMContentLoaded', prepCanvas);
//console.log(model)

async function predict(flag){
    grayscale();
    
    console.log('performing resize');
    tf_tensor = tf.browser.fromPixels(canvas,1);
    if(flag){
        tf_tensor = detectDigit(tf_tensor);
    }
    
    resized_arr = tf.image.resizeBilinear(tf_tensor, [20, 20]);
    console.log(resized_arr.shape);
    resized_arr = resized_arr.pad([[4,4],[4,4],[0,0]]); 
    //tf.browser.toPixels(resized_arr,canvas);
    resized_arr = tf.cast(resized_arr,'float32');
    //ctx2.drawImage(canvas,0,0,200,200,0,0,28,28);
    
    resized_arr = resized_arr.expandDims(0);
    resized_arr = resized_arr.div(tf.scalar(255.0));
    //tf_tensor = tensor.div(tf.scalar(255));
    model = await loadModel();
    prediction = model.predict(resized_arr);
    prediction.print()
    prediction = prediction.dataSync();
    output = document.getElementById('output');
    output.innerHTML = 'prediction: ' + tf.argMax(prediction).arraySync()
    console.log(tf.argMax(prediction).arraySync());
    //tf.argMax(prediction).print();
}

function grayscale() {
    console.log('here');
    let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let data = imageData.data;
    for (var i = 0; i < data.length; i += 4) {
      var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
      data[i]     = avg; // red
      data[i + 1] = avg; // green
      data[i + 2] = avg; // blue
    }
    ctx.putImageData(imageData, 0, 0);
};

function detectDigit(tensor) {
    let arr = tensor.arraySync();
    let xflag = true,yflag = true;
    let x1 = null,x2 = null,y1 =null,y2 = null;
    for(i=0;i<200;i++){
        if(tf.max(arr[i]).dataSync()[0] === 255){
            //console.log('setting x1 with i:' + i);
            //console.log(xflag);
            if(xflag){
                //console.log('setting x1 with i:' + i);
                //console.log('value: ' + tf.max(arr[i]).dataSync()[0])
                x1 = i;
                xflag = false;
            }
            else{
                x2 = i;
            }
        }
        if(tf.max(tensor.slice([0,i],[200,1]).dataSync()).dataSync()[0] === 255){
            if(yflag){
                y1 = i;
                yflag = false;
            }
            else{
                y2 = i;
            }
        }
    }

    let height = x2 - x1;
    let width = y2 - y1;
    ctx.strokeStyle = "blue";
    ctx.lineWidth = 2;
    ctx.strokeRect(y1 - 8, x1 - 8, width + 16, height + 16);

    ctx.strokeStyle = 'white';
    ctx.lineWidth = 12;


    console.log('X1:' + x1);
    console.log('X2:' + x2);
    console.log('Y1:' + y1);
    console.log('Y2:' + y2);
    console.log('width:' + width);
    console.log('height:' + height);
    tensor = tensor.slice([y1-8,x1-8],[height+16,width+16,1]);
    return tensor;
}

function reset() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

