const Matrix = require('./Matrix');
const NArray = require('./NArray');
const Activity = require('./Activity');
const Model = require('./Model');
const Layer = require('./BaseLayer');
const Loss = require('./Loss');
const Evaluate = require('./Evaluate');

const IMAGE_H = 28;
const IMAGE_W = 28;
const INPUT_DIMS = IMAGE_H * IMAGE_W;
const TYPE_COUNT = 10;
var fs = require('fs');


function readImg(url) {
    let buffer = fs.readFileSync(url);
    let len = buffer.length;
    let numImgs = (len - 16) / INPUT_DIMS;
    let data = new NArray([numImgs, IMAGE_H, IMAGE_W, 1]);
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

// trainX.reshape([trainX.shape[0], INPUT_DIMS]);
// testX.reshape([testX.shape[0], INPUT_DIMS]);


trainX.scale(1 / 255.0);
testX.scale(1 / 255.0);

console.log('init model');

// let model = new Model();
// model.addLayer(new Layer.NN(INPUT_DIMS, 256));
// model.addLayer(new Activity.relu());
// model.addLayer(new Layer.NN(256, 128));
// model.addLayer(new Activity.relu());
// model.addLayer(new Layer.NN(128, 10));
// model.addLayer(new Activity.softmax());
// model.setLoss(new Loss.CrossEntropy());
// model.setLearnRate(0.01);
// model.setL2(1e-6);
// model.init();

let model = new Model();
model.addLayer(new Layer.Conv2d(5, 5, 1, 8, 2, 2)); //layer 1 [k, 14,14,16]
model.addLayer(new Activity.relu());//layer 2
model.addLayer(new Layer.Conv2d(5, 5, 8, 16, 2, 2));//[k, 7,7,32]
model.addLayer(new Activity.relu());//
// model.addLayer(new Layer.Conv2d(5, 5, 32, 64, 2, 2));//[k, 4,4,32]
// model.addLayer(new Activity.relu());//
model.addLayer(new Layer.Flatten());// [k, 7*7*32] 1568
model.addLayer(new Layer.NN(7 * 7 * 16, 10));//
// model.addLayer(new Activity.relu());//
// model.addLayer(new Layer.NN(512, 10));//
model.addLayer(new Activity.softmax());
model.setLoss(new Loss.CrossEntropy());//loss
model.setLearnRate(0.1);
model.init();


// let X = testX.get([[0,16]], true)
// let Y = testY.get([[0,16]])

let ix = 0;
let batch_size = 60;
let i = 0;
let epoch = 0;
while (true) {
    if (ix + batch_size > TrainCount) {
        ix = TrainCount - batch_size;
    }
    X = trainX.get([[ix, ix + batch_size]], true);
    Y = trainY.get([[ix, ix + batch_size]], true);
    let loss = model.train([X, Y]);
    console.log(`epoch: ${epoch}, i: ${i}, loss: ${loss}`);
    i++;
    if (epoch == 10) {
        model.setLearnRate(0.02);
    } else if (epoch == 20) {
        model.setLearnRate(0.005);
    } else if (epoch == 30) {
        model.setLearnRate(0.001);
    }
    ix += batch_size;
    if (ix >= TrainCount) {
        ix = 0;
        epoch++;
        i = 0;
        ////////////////////////////////计算准确率
        let totalAcc = 0;
        let n = Math.floor(TestCount / batch_size) - 1;
        let testIx = 0;
        for (let k = 0; k < n; k++) {
            X = testX.get([[testIx, testIx + batch_size]], true);
            Y = testY.get([[testIx, testIx + batch_size]], true);
            let p = model.front(X);
            let acc = Evaluate.ClassificationAccuracy(Y, p);
            totalAcc += acc * batch_size;
            testIx += batch_size;
        }
        let lastBatchSize = TestCount - testIx;
        X = testX.get([[testIx, TestCount]], true);
        Y = testY.get([[testIx, TestCount]], true);
        let p = model.front(X);
        let acc = Evaluate.ClassificationAccuracy(Y, p);
        totalAcc += acc * lastBatchSize;
        totalAcc /= TestCount;
        console.log('Evaluate acc: ', totalAcc);
    }
}

