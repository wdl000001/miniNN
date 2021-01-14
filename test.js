const Matrix = require('./Matrix');
const NArray = require('./NArray');
const Activity = require('./Activity');
const Model = require('./Model');
const Layer = require('./BaseLayer');
const Loss = require('./Loss');
const Evaluate = require('./Evaluate');
// let arr = new NArray([4, 3, 2]);
// for (let i = 0; i < arr.array.length; i++) {
//   arr.array[i] = i;
// }

// console.log(arr.toString());
// let brr = arr.transpose(true);
// console.log(brr.toString());
// let crr = brr.add(brr);
// let drr = brr.sub(brr);
// console.log('++++++-----------');
// console.log(crr.toString());
// console.log(drr.toString());
/////test softmax
// const softmax = Activity.softmax

// let arr = new NArray([3, 5])
// for(let i = 0; i < 15; i++){
//   arr.array[i] = Math.random() * 50
// }
// console.log(arr.toString())

// let fun =  new softmax()
// let sx = fun.func(arr)
// console.log(sx.toString())

//test model

function createRandom(shape, max = 1) {
  let narr = new NArray(shape);
  let { array, numEle } = narr;
  for (let i = 0; i < numEle; i++) {
    array[i] = Math.random() * max - max * 0.5;
  }
  return narr;
}


// let model = new Model()
// model.addLayer(new Layer.NN(32, 64)) //layer 0
// model.addLayer(new Activity.relu())//layer 1
// model.addLayer(new Layer.NN(64, 32)) //layer 0
// model.addLayer(new Activity.relu())//layer 1
// model.addLayer(new Layer.NN(32, 16))//layer 2
// model.setLoss(new Loss.MSE())//loss
// model.init()

// let X = createRandom([32, 32])
// let Y = createRandom([32, 16])

// let pred = model.front(X)
// console.log(pred.toString())


// let i = 0;
// while(true){
//   let loss = model.train([X, Y])
//   console.log('train i:', i, ' loss: ', loss)
//   i++
// }

//model2 
// let model = new Model()
// model.addLayer(new Layer.NN(32, 256)) //layer 1
// model.addLayer(new Activity.sigmoid())//layer 2
// model.addLayer(new Layer.NN(256, 10))//layer 3
// model.addLayer(new Activity.softmax())//layer 4
// model.setLoss(new Loss.CrossEntropy())//loss
// model.setLearnRate(0.01)
// model.init()

// let X = createRandom([16, 32])
// let Y = new NArray([16, 10])
// for (let i = 0; i < X.shape[0]; i++) {
//   let ix = Math.floor(Math.random() * Y.shape[1])
//   Y.set([i, ix], 1)
// }

// let i = 0;
// while (true) {
//   let loss = model.train([X, Y])
//   console.log('train i:', i, ' loss: ', loss)
//   i++
// }

// model3 conv2d
let model = new Model();
model.addLayer(new Layer.Conv2d(5, 5, 1, 8, 2, 2)); //layer 1 [k, 14,14,8]
model.addLayer(new Activity.relu());//layer 2
model.addLayer(new Layer.Conv2d(5, 5, 8, 16, 2, 2));//[k, 7,7,16]
model.addLayer(new Activity.relu());//
// model.addLayer(new Layer.Conv2d(5, 5, 16, 16, 2, 2));//[k, 4,4,16]
// model.addLayer(new Activity.relu());//
model.addLayer(new Layer.Flatten());// 
model.addLayer(new Layer.NN(7 * 7 * 16, 4));//
model.addLayer(new Activity.softmax());
model.setLoss(new Loss.CrossEntropy());//loss
model.setLearnRate(0.1);
model.init();

let X = createRandom([12, 28, 28, 1]);
let Y = new NArray([12, 4]);
for (let i = 0; i < X.shape[0]; i++) {
  let ix = Math.floor(Math.random() * Y.shape[1]);
  Y.set([i, ix], 1);
}

let i = 0;
while (true) {
  let loss = model.train([X, Y]);
  console.log('train i:', i, ' loss: ', loss);
  i++;
  if(i%10==0){
    let pred = model.front(X)
    let acc = Evaluate.ClassificationAccuracy(Y, pred);
    console.log('acc: ', acc)
  }
}