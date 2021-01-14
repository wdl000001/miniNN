const NArray = require('./NArray');

/**
 * 
 * @param {} vec (m,n)*(n,p) = (m,p)
 */
function mtxDot(mn, np) {
  let mnShape = mn.shape;
  let npShape = np.shape;
  if (mnShape[1] !== npShape[0] || mnShape.length != 2 || npShape.length != 2) {
    throw new Error('mtxDot:input error: ' + mnShape + '*' + npShape);
  }
  let [m, n] = mn.shape;
  let p = np.shape[1];
  let rt = new NArray([m, p]);

  for (let i = 0; i < m; i++) { //row
    for (let j = 0; j < p; j++) { //col
      sum = 0;
      for (let k = 0; k < n; k++) {
        sum += mn.get([i, k]) * np.get([k, j]);
      }
      rt.set([i, j], sum);
    }
  }
  return rt;
};

function mtxDot2(mn, np) {
  let mnShape = mn.shape;
  let npShape = np.shape;
  if (mnShape[1] !== npShape[0] || mnShape.length != 2 || npShape.length != 2) {
    throw new Error('mtxDot:input error: ' + mnShape + '*' + npShape);
  }
  let [m, n] = mn.shape;
  let p = np.shape[1];
  let rt = new NArray([m, p]);
  let mnarray = mn.array;
  let nparray = np.array;
  let col = new Array(n);
  for (let j = 0; j < p; j++) {
    for (let i = 0; i < n; i++) {
      col[i] = nparray[i * p + j];
    }
    for (let i = 0; i < m; i++) {
      let sum = 0;
      let row_start = i * n;
      for (let k = 0; k < n; k++) {
        sum += col[k] * mnarray[row_start + k];
      }
      rt.array[i * p + j] = sum;//[m,p]
    }
  }
  return rt;
};

module.exports = {
  mtxDot: mtxDot2,
  mtxDotSlow: mtxDot
};


if (require.main === module) {
  // done;
  let n = 256;
  let t0 = new Date().getTime();
  var a = new NArray([n, n]);

  for (let i = 0; i < a.array.length; i++) {
    a.array[i] = Math.random();
  }
  var b = new NArray([n, n]);
  for (let i = 0; i < b.array.length; i++) {
    b.array[i] = Math.random();
  }
  let t1 = new Date().getTime();
  console.log('init time:', (t1 - t0) / 1000);

  var d = mtxDot2(a, b);
  let t2 = new Date().getTime();
  console.log('dot2 time:', (t2 - t1) / 1000);

  var c = mtxDot(a, b);
  let t3 = new Date().getTime();
  console.log('dot time:', (t3 - t2) / 1000);

  // console.log(c.sub(d));


  // //done
  // var a = new NArray([3, 2]);
  // let k = 1;
  // for (let i = 0; i < 3; i++) {
  //   for (let j = 0; j < 2; j++) {
  //     a.set([i, j], k++);
  //   }
  // }
  // console.log(a.toString());
  // var b = new NArray([2, 4]);
  // for (let i = 0; i < b.numEle; i++) {
  //   b.array[i] = i;
  // }
  // console.log(b.toString());
  // console.log('（3,2）*（2,4）--------------------------------');
  // console.log(mtxDot(a, b).toString());
  // console.log('b--------------------------------');

  // console.log('（3,2）*（2,4）--------------------------------');
  // console.log(mtxDot2(a, b).toString());
  // console.log('b--------------------------------');

  // b = new NArray([2, 6, 4, 3]);
  // for (let i = 0; i < b.numEle; i++) {
  //   b.array[i] = i;
  // }
  // console.log(b.toString());
  // console.log('b[1,:]--------------------------------');
  // var c = b.get([1,]);
  // console.log(c.toString());

  // console.log('b[1, 0:3, 0:3]--------------------------------');
  // c = b.get([1, [0, 3], [0, 3]]);
  // console.log(c.toString());
}