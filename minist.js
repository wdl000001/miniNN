const Matrix = require('./Matrix');
const NArray = require('./NArray');
const Activity = require('./Activity');
const Model = require('./Model');
const Layer = require('./BaseLayer');
const Loss = require('./Loss');

const IMAGE_H = 28;
const IMAGE_W = 28;
const INPUT_DIMS = IMAGE_H * IMAGE_W;
const TYPE_COUNT = 10;
var fs = require('fs');


function readImg(url) {
    let buffer = fs.readFileSync(url);
    let len = buffer.length;
    let numImgs = (len - 16) / INPUT_DIMS;
    let data = new NArray([numImgs, IMAGE_H, IMAGE_W]);
    let ix = 16;
    let array = data.array;
    for (let i = 0; i < array.length; i++) {
        array[i] = buffer[i + 16];
    }
    return data;
}

function readLabel(url) {
    let buffer = fs.readFileSync(url);
    let len = buffer.length;
    let numLabels = (len - 8);
    let data = new NArray([numLabels, TYPE_COUNT]);
    for (let i = 0; i < numLabels; i++) {
        const ix = buffer[i + 8];
        data.set([i, ix], 1);
    }
    return data;
}
console.log('read data');

let trainX = readImg('./data/train-images.idx3-ubyte');
let testX = readImg('./data/t10k-images.idx3-ubyte');


var TrainCount = trainX.shape[0];
var TestCount = testX.shape[0];

let trainY = readLabel('./data/train-labels.idx1-ubyte');
let testY = readLabel('./data/t10k-labels.idx1-ubyte');
// trainX.reshape([trainX.shape[0], INPUT_DIMS])
// testX.reshape([testX.shape[0], INPUT_DIMS])
trainX.reshape([TrainCount, IMAGE_H, IMAGE_W, 1]);
trainX.scale(1 / 255.0);

testX.reshape([TestCount, IMAGE_H, IMAGE_W, 1]);
testX.scale(1 / 255.0);

console.log('init model');

// let model = new Model()
// model.addLayer(new Layer.NN(INPUT_DIMS, 128)) 
// model.addLayer(new Activity.relu())
// // model.addLayer(new Layer.NN(256, 128))
// // model.addLayer(new Activity.relu())
// model.addLayer(new Layer.NN(128, 10))
// model.addLayer(new Activity.softmax())
// model.setLoss(new Loss.CrossEntropy())
// model.setLearnRate(0.5)
// model.setL2(1e-6)
// model.init()

let model = new Model();
model.addLayer(new Layer.Conv2d(3, 3, 1, 8, 2, 2)); //layer 1 [k, 14,14,16]
model.addLayer(new Activity.relu());//layer 2
model.addLayer(new Layer.Conv2d(3, 3, 8, 16, 2, 2));//[k, 7,7,32]
model.addLayer(new Activity.relu());//
model.addLayer(new Layer.Flatten());// [k, 7*7*32] 1568
model.addLayer(new Layer.NN(7 * 7 * 16, 10));//
model.addLayer(new Activity.softmax());
model.setLoss(new Loss.CrossEntropy());//loss
model.setLearnRate(0.01);
model.init();


// let X = testX.get([[0,16]], true)
// let Y = testY.get([[0,16]])

let ix = 0;
let batch_size = 32;
let i = 0;

while (true) {
    if (ix + batch_size > TrainCount) {
        ix = TrainCount - batch_size;
    }
    X = trainX.get([[ix, ix + batch_size]], true);
    Y = trainY.get([[ix, ix + batch_size]]);
    let loss = model.train([X, Y]);
    console.log('===========================train i:', i, ' loss: ', loss);
    i++;
    if (i == 100) {
        model.setLearnRate(0.002);
    } else if (i == 200) {
        model.setLearnRate(0.0005);
    } else if (i == 300) {
        model.setLearnRate(0.0001);
    }
    ix = (ix + batch_size) % TrainCount;
}

